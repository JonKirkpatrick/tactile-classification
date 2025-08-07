module int8_vector_core (
    input  logic clk,
    input  logic rst,

    input  logic signed [7:0] input_a [256],
    input  logic signed [7:0] input_b [256],
    input  logic signed [7:0] input_c [256],

    input  logic [3:0] tap_level,                   // 2 to 8
    input  logic [3:0] shift_amt,                   // 0 to 15
    input  logic partial_product,                   // 1 = return raw 32-bit sum at root
    input  logic relu_active,                       // 1 = apply ReLU activation
    input  logic pool_active,                       // 1 = pooling is active
    input  logic gap_active,                        // 1 = gap pooling is active
    input  logic new_vector_rise,                   // Rising edge of new vector signal

    output logic signed [7:0]  output_packed [128], // 8-bit signed values
    output logic signed [31:0] partial_raw_output,  // full raw value if partial_product = 1
    output logic valid_rise,
    output logic idle
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
    // output_packed: Array of 128 quantized 8-bit signed values.
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

    logic multiply;         // Flag to indicate the registers are loaded and multiplication can happen
    logic multiply_d;       // Delayed multiply signal for control synchronization
    logic multiply_rise;    // Rising edge of the multiply signal
    logic ready;            // Ready signal to indicate when to collect the output from the adder tree
    logic ready_d;          // Delayed ready signal for control synchronization
    logic ready_rise;       // Rising edge of the ready signal
    logic quantize;         // Flag to indicate that we're ready to quantize
    logic quantize_d;       // Delayed quantize signal for control synchronization
    logic quantize_rise;    // Rising edge of the quantize signal
    logic pack;             // Flag to indicate we're ready to pack the output
    logic pack_d;           // Delayed pack signal for control synchronization
    logic pack_rise;        // Rising edge of the pack signal
    logic valid;            // Valid signal to indicate when the output is ready
    logic valid_d;          // Delayed valid signal to indicate when the output is ready

    assign multiply_rise = multiply & ~multiply_d;  // Detect rising edge of multiply signal
    assign ready_rise = ready & ~ready_d;           // Detect rising edge of ready signal
    assign quantize_rise = quantize & ~quantize_d;  // Detect rising edge of quantize signal
    assign pack_rise = pack & ~pack_d;              // Detect rising edge of pack signal
    assign valid_rise = valid & ~valid_d;           // Detect rising edge of valid signal

    // === Stage 1: 256 signed multipliers ===

    // Layer 0
    logic signed [31:0] tap_outputs [9][256];

    always_ff @(posedge clk) begin
        if (multiply_rise && ~pool_active) begin
            for (int i = 0; i < 256; i++) begin
                tap_outputs[0][i] <= input_a[i] * input_b[i];
            end
        end else if (multiply_rise && pool_active) begin
            for (int i = 0; i < 256; i++)
                tap_outputs[0][i] <= input_c[i];
        end
    end

    // === Stage 2: Tappable Adder Tree ===

    // Layers 1 to 8
    genvar level, i;
    generate
        for (level = 1; level < 9; level++) begin : adder_layers
            for (i = 0; i < (256 >> level); i++) begin : tree_nodes
                always_ff @(posedge clk) begin
                    if (~pool_active || gap_active) begin
                        tap_outputs[level][i] <= tap_outputs[level-1][i<<1] + tap_outputs[level-1][(i<<1)+1];
                    end else begin
                        if (tap_outputs[level-1][i<<1] > tap_outputs[level-1][(i<<1)+1]) begin
                            tap_outputs[level][i] <= tap_outputs[level-1][i<<1];
                        end else begin
                            tap_outputs[level][i] <= tap_outputs[level-1][(i<<1)+1];
                        end
                    end
                    
                end
            end
        end
    endgenerate

    // === Stage 3: Output Handling ===

    logic signed [31:0] selected_values [128];
    logic signed [7:0]  quantized [128];

    always_ff @(posedge clk) begin
        if (ready_rise) begin
            for (int j = 0; j < 128; j++) begin
                if (j < (256 >> tap_level)) begin
                    selected_values[j] <= tap_outputs[tap_level][j];
                end else
                    selected_values[j] <= 32'sd0; // pad with zeros beyond the tap level
            end
        end
    end

    // Shift, clamp, and apply ReLU activation
    always_ff @(posedge clk) begin
        if (quantize_rise) begin
            if (partial_product) begin
                for (int j = 0; j < 128; j++)
                    quantized[j] <= 8'sd0;
            end else begin
                for (int j = 0; j < 128; j++) begin
                    automatic logic signed [31:0] shifted;
                    shifted = selected_values[j] >>> shift_amt;

                    // Clip to int8 range
                    if (shifted > 127)        shifted = 127;
                    else if (shifted < -128) shifted = -128;

                    // Apply ReLU before assigning
                    if (relu_active && shifted < 0)
                        quantized[j] <= 8'sd0;
                    else
                        quantized[j] <= shifted[7:0];
                end
            end
        end
    end

    // Final packing
    always_ff @(posedge clk) begin
        if (pack_rise) begin
            for (int j = 0; j < 128; j++) begin
                output_packed[j] <= quantized[j];
            end
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

    localparam PIPELINE_DEPTH = 13;
    logic [PIPELINE_DEPTH-1:0] delay_line;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            delay_line <= 0;
        end else begin
            if (new_vector_rise) begin
                delay_line <= {delay_line[PIPELINE_DEPTH-2:0], 1'b1};
            end else begin
                delay_line <= {delay_line[PIPELINE_DEPTH-2:0], 1'b0};
            end
        end
    end

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            valid_d <= 0;
            quantize_d <= 0;
            pack_d <= 0;
            ready_d <= 0;
            multiply_d <= 0;
        end else begin
            valid_d <= valid;
            quantize_d <= quantize;
            pack_d <= pack;
            ready_d <= ready;
            multiply_d <= multiply;
            idle <= (delay_line == 0);
        end
    end

    assign valid = delay_line[PIPELINE_DEPTH-1-8+tap_level];
    assign pack = delay_line[PIPELINE_DEPTH-2-8+tap_level];
    assign quantize = delay_line[PIPELINE_DEPTH-3-8+tap_level];
    assign ready = delay_line[PIPELINE_DEPTH-4-8+tap_level];
    assign multiply = delay_line[0];

endmodule