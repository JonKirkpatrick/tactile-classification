module int8_vector_core (
    input  logic clk,
    input  logic rst,
    input  logic start,

    input  logic signed [7:0] input_a [256],
    input  logic signed [7:0] input_b [256],

    input  logic [3:0] tap_level,          // 2 to 8
    input  logic [3:0] shift_amt,          // 0 to 15
    input  logic partial_product,          // 1 = return raw 32-bit sum at root
    input  logic relu_active,              // 1 = apply ReLU activation

    output logic signed [7:0]  output_packed [64],   // 8-bit signed values
    output logic signed [31:0] partial_raw_output,   // full raw value if partial_product = 1
    output logic valid
);

    // === Module Description ===
    // int8_vector_core.sv
    // This module performs a 256-element dot product operation with configurable quantization and packing.
    // By tapping the tree at different levels, we are able to perform several dot products simultaneously.
    // This allows for efficient computation of multiple dot products in parallel.  The one major caveat
    // is that inputs must be internally padded to power of two segments to ensure that each individual dot 
    // product vector is aligned with the adder tree structure.
    // It is designed for deployment on FPGA hardware, specifically the DE10-Lite.
    // The pipeline stages include multiplication, adder tree reduction, quantization, and output packing.
    // Control signals are managed via a delay line to synchronize operations across pipeline stages.

    // === Inputs ===
    // clk: Clock signal.
    // rst: Reset signal (active high).
    // start: Start signal to initiate the pipeline.
    // input_a, input_b: Arrays of signed 8-bit inputs for the dot product operation.
    // tap_level: Specifies the level of the adder tree to sample.
    // shift_amt: Specifies the amount to shift during quantization.
    // partial_product: Flag to output the raw 32-bit sum at the root of the adder tree.
    // relu_active: Flag to apply ReLU activation on the quantized output.

    // === Outputs ===
    // output_packed: Array of 32 quantized 8-bit signed values.
    // partial_raw_output: Raw 32-bit sum if partial_product is set.
    // valid: Indicates when the output is ready.

    // === Internal Signals ===
    // multiply: Flag to begin multiplication.
    // ready: The whole adder tree has been populated.
    // quantize: Flag to begin quantization.
    // pack: Flag to begin output packing.
    // delay_line: Pipeline control signals.

    // === Pipeline Stages ===
    // Stage 1: Multiplication
    // - Multiplies corresponding elements of input_a and input_b.
    // - Results are stored in the first layer of tap_outputs as signed int32.

    // Stage 2: Adder Tree Reduction
    // - Reduces the results in tap_outputs across 8 layers.
    // - Each layer halves the number of elements by summing pairs.

    // Stage 3: Output Handling
    // - Selects values from the specified tap_level.
    // - Quantizes the selected values by shifting and clamping.
    // - Applies ReLU activation if relu_active is set.
    // - Packs the quantized values into output_packed.

    // === Control Pipeline ===
    // - delay_line manages control signals for each pipeline stage.
    // - valid, pack, quantize, ready, and multiply are derived from delay_line.
    // - The pipeline depth is set to 14 stages to accommodate the adder tree and control signals.
    // - The toggle signal is used to alternate control signals across clock cycles.
    // - This ensures that new products are placed into the adder tree every second clock cycle.

    // === Reset Behavior ===
    // - The partial raw output is initialized to zero.
    // - delay_line is cleared.

    // === Deployment Notes ===
    // - Designed for FPGA deployment with efficient resource utilization.
    // - Ensure timing constraints are met during synthesis.
    // - Testbench validation covers edge cases and typical use scenarios.

    logic multiply; // Flag to indicate the registers are loaded and multiplication can happen
    logic ready; // Ready signal to indicate when to collect the output from the adder tree
    logic quantize; // Flag to indicate that we're ready to quantize
    logic pack; // Flag to indicate we're ready to pack the output

    // === Stage 1: 256 signed multipliers ===

    // Layer 0
    logic signed [31:0] tap_outputs [9][256];
    always_ff @(posedge clk) begin
        if (multiply) begin
            for (int i = 0; i < 256; i++)
                tap_outputs[0][i] <= input_a[i] * input_b[i];
        end
    end

    // === Stage 2: Tappable Adder Tree ===

    // Layers 1 to 8
    genvar level, i;
    generate
        for (level = 1; level < 9; level++) begin : adder_layers
            for (i = 0; i < (256 >> level); i++) begin : tree_nodes
                always_ff @(posedge clk) begin
                    tap_outputs[level][i] <= tap_outputs[level-1][2*i] + tap_outputs[level-1][2*i+1];
                end
            end
        end
    endgenerate

    // === Stage 3: Output Handling ===

    logic signed [31:0] selected_values [64];
    logic signed [7:0]  quantized [64];

    always_ff @(posedge clk) begin
        if (ready) begin
            for (int j = 0; j < 64; j++) begin
                if (j < (256 >> tap_level))
                    selected_values[j] <= tap_outputs[tap_level][j];
                else
                    selected_values[j] <= 32'sd0; // pad with zeros beyond the tap level
            end
        end
    end

    // Shift, clamp, and apply ReLU activation
    always_ff @(posedge clk) begin
        if (quantize) begin
            if (partial_product) begin
                for (int j = 0; j < 64; j++)
                    quantized[j] <= 8'sd0;
            end else begin
                for (int j = 0; j < 64; j++) begin
                    automatic logic signed [31:0] shifted;
                    shifted = selected_values[j] >>> shift_amt;
                    if (shifted > 127)        quantized[j] <= 8'sd127;
                    else if (shifted < -128) quantized[j] <= -8'sd128;
                    else                     quantized[j] <= shifted[7:0];

                    // Apply ReLU activation if enabled
                    if (relu_active && quantized[j] < 0)
                        quantized[j] <= 8'sd0;
                end
            end
        end
    end

    // Final packing
    always_ff @(posedge clk) begin
        if (pack) begin
            for (int j = 0; j < 64; j++)
                output_packed[j] <= quantized[j];
        end
    end

    // Partial product raw output
    always_ff @(posedge clk or posedge rst) begin
        if (rst)
            partial_raw_output <= 32'sd0;
        else if (pack && partial_product)
            partial_raw_output <= tap_outputs[8][0];
        else if (pack && !partial_product)
            partial_raw_output <= 32'sd0;
    end

    // === Control Pipeline ===

    localparam PIPELINE_DEPTH = 14;
    logic [PIPELINE_DEPTH-1:0] delay_line;
    logic toggle;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            delay_line <= '0;
            toggle <= 1'b0;
        end else if (start) begin
            delay_line <= {delay_line[PIPELINE_DEPTH-2:0], 1'b1};
            toggle <= 1'b0;
        end else begin
            delay_line <= {delay_line[PIPELINE_DEPTH-2:0], toggle};
            toggle <= ~toggle;
        end
    end

    assign valid = delay_line[PIPELINE_DEPTH-1-8+tap_level];
    assign pack = delay_line[PIPELINE_DEPTH-2-8+tap_level];
    assign quantize = delay_line[PIPELINE_DEPTH-3-8+tap_level];
    assign ready = delay_line[PIPELINE_DEPTH-4-8+tap_level];
    assign multiply = delay_line[1];

endmodule