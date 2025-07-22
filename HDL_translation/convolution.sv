`include "int8_vector_core.sv"

// === Struct Definition ===
// In order to treat every convolution layer in our model as a depthwise 1d convolution,
// we define a struct to hold the context and semantics of the synthetic convolution layer.

typedef struct packed {
    logic        first_block;       // Indicates if this is the first block in the model
    logic        data_context;      // 0 for input/intermediate tensor, 1 for weights in param_mem
    logic        pong;              // Used to toggle between two sets of intermediate results
    logic [31:0] weight_pointer;    // Pointer to the first weight in layer's parameters

    logic [8:0]  kernel_size;       // Size of the kernel for the convolution
    logic [4:0]  padding;           // Padding size
    logic [8:0]  stride;            // Stride for the convolution

    logic [31:0] tensor_size;       // Number of elements in the input tensor
    logic [8:0]  in_chan;           // Number of input channels
    logic [8:0]  group_size;        // Size of the group for the convolution
    logic [15:0] temporal_length;   // Length of the temporal dimension

    logic [8:0]  pack_size;         // Size of each vector to be packed
    logic [7:0]  pack_count;        // Number of packed vectors per batch
    logic [31:0] pack_batch;        // Number of batches per convolution layer

    logic [8:0]  current_channel;   // Current channel being processed
    logic [15:0] temporal_position; // Current position in the temporal dimension
    logic [8:0]  current_group;     // Current group being processed
    logic [8:0]  kernel_position;   // Current position in the kernel
    logic [15:0] dots_per_channel;  // The number of dot products per channel
} conv_context_t;

// === Convolution Module ===
//
// This module implements a flexible convolution engine for CNN models.
// It supports depthwise, pointwise, and fully connected layers, with configurable parameters.
// The module uses an instruction-driven approach to define layer properties and operations.
//
// Key Features:
// - Instruction-based configuration for convolution layers.
// - Supports depthwise, pointwise, and fully connected layers.
// - Handles input tensors and weights in various formats.
// - Integrates with the int8_vector_core module for efficient dot product computation.
// - Designed for FPGA deployment, specifically targeting the DE10-Lite.
//
// Limitations:
// - Assumes input tensors and weights are preloaded into memory.
// - Requires explicit padding and stride values for convolution layers.
// - Fixed maximum tensor and parameter memory sizes.
//
// Instruction Format:
// - 32-bits[63:32]: Experimental instruction block.
// - 4-bits [63:60]: Relationship between batch and channels (log2 of batches per channel or literal of channels per batch).
// - 1-bit  [59]: First pointwise convolution (1) or not (0).
// - 1-bit  [31]: First block (1) or not (0).
// - 1-bit  [30]: Apply ReLU (1) or not (0).
// - 1-bit  [29]: Global Average Pooling (1) or not (0).
// - 3-bits [28:26]: MaxPool log2 size (0 for no pooling).
// - 4-bits [25:22]: Padding (Implies a max kernel size of 31).
// - 4-bits [21:18]: Temporal length (log2 of time steps).
// - 6-bits [17:12]: Channels in (log2 of input channels or literal if first block).
// - 4-bits [11:8]: Channels out (log2 of output channels).
// - 4-bits [7:4]: Shift amount for quantization to int8.
// - 4-bits [3:0]: Packet size (ceiling of the (log2 of the kernel size) - 1).
//
// Context Struct:
// - Holds semantic information for the convolution operation.
// - Includes parameters like padding, tensor size, kernel size, stride, and more.
// - Used to manage state and offsets during vector packing and dot product computation.
//
// Integration:
// - The module interfaces with the int8_vector_core for dot product operations.
// - Results are stored in intermediate buffers for further processing.
// - Designed to work seamlessly with BRAM for parameter and instruction storage.
//
// Simulation Notes:
// - Includes initial blocks for test data population.
// - These should be removed in production deployments.
//
// Usage:
// - Configure the instruction memory with layer descriptions.
// - Load input tensors and weights into memory.
// - Trigger the module with the start signal.
// - Monitor the busy and done signals for operation status.
// - Retrieve results from the intermediate buffers after processing.
//


// This signature will be changed once development is complete.
// Inferred class will be driven by ArgMax and thus will not be signed.
// Also, the inferred class will likely be smaller than 8 bits.
module convolution (
    input logic clk,
    input logic reset,
    input logic start,
    input logic signed [7:0] input_tensor [0:8191],
    output logic signed [7:0] inferred_class, 
    output logic busy,
    output logic done
);

    // These figures are based on the intended target hardware: DE10-Lite FPGA
    localparam int PARAM_MEM_DEPTH = 131072;
    localparam int INSTR_MEM_DEPTH = 256;

    // These are the memory blocks for parameters and instructions.
    // In their current form, they are only used for simulation purposes.
    // We will need to replace them with actual BRAM in production.
    //initial $readmemh("parameters.hex", param_mem);
    initial $readmemh("instructions.hex", instr_mem);

    conv_context_t ctx;  // Semantically rich context for the convolution operation
    logic temporal_axis; // Indicates if the tensor is aligned in memory in temporal-major order
    logic partial_product; // Flag to indicate the dot product being computed is larger than 256 elements wide.
    
    logic [63:0] instr_mem [0:INSTR_MEM_DEPTH-1]; // Instruction memory for CNN layers
    logic [63:0] current_instr;  // Current instruction being processed
    logic [7:0] instruction_pointer; // Pointer to the current instruction being processed

    logic [15:0] current_batch; // Current batch of 256 inputs and parameters being processed
    logic [15:0] batches_processed; // Number of batches processed so far

    logic signed [7:0] param_mem [0:PARAM_MEM_DEPTH-1]; // Parameter memory for weights
    logic [31:0] pointer_advance; // The amount to advance the parameter pointer after current layer
    
    logic signed [7:0] intermediate_results [0:16383][2]; // Intermediate results buffer
    logic signed [7:0] vector_a [0:255]; // Vector A for dot product
    logic signed [7:0] vector_b [0:255]; // Vector B for dot product
    logic signed [7:0] dot_product_return [0:63]; // Dot product results
    logic signed [31:0] raw_output; // Raw output for partial products

    logic layer_complete; // Indicates if the current layer processing is complete


    // Raw Instruction Fields
    logic first_pw; // First pointwise convolution in the model
    logic relu_active;
    logic global_avg_pool;
    logic [2:0] maxpool;
    logic [3:0] padding;
    logic [3:0] time_steps;
    logic [5:0] in_channels;
    logic [3:0] out_channels;
    logic [3:0] shift_amt;
    logic [3:0] packet;
    logic [3:0] batch_chan_ref; // log2 of batches per channel or literal of channels per batch

    // Interpretted Instruction Fields
    logic [8:0] in_chan;
    logic [8:0] out_chan;
    logic [15:0] temporal_length;
    logic [4:0] pool_amt [2];
    logic [3:0] tap_level;
    
    // === These are just for simulation and must be removed in production ===
    initial begin
        // Populate weight memory with some test data
        for (int i = 0; i < 1001; i++) begin
            param_mem[i] = (i - 84) % 128;
        end
    end


    int8_vector_core int8_core (
        .clk(clk),
        .rst(reset),
        .start(start),
        .input_a(vector_a),
        .input_b(vector_b),
        .tap_level(tap_level),
        .shift_amt(shift_amt),
        .partial_product(partial_product),
        .relu_active(relu_active),
        .output_packed(dot_product_return),
        .partial_raw_output(raw_output),
        .valid(valid),
        .toggle(toggle)
    );

    always_comb begin

        // Populate Raw Instruction Fields
        ctx.first_block = current_instr[31];
        relu_active = current_instr[30];
        global_avg_pool = current_instr[29];
        maxpool = current_instr[28:26];
        ctx.padding = current_instr[25:22];
        time_steps = current_instr[21:18];
        in_channels = current_instr[17:12];
        out_channels = current_instr[11:8];
        shift_amt = current_instr[7:4];
        packet = current_instr[3:0];
        batch_chan_ref = current_instr[63:60];
        first_pw = current_instr[59];

        // Interpret Raw Instruction Fields
        if (ctx.first_block) begin
            // First Block: Use literal for in_channels
            // We assume this is also a depthwise convolution
            // and out_channels is equal to in_channels
            // This is a special case for the first layer
            in_chan = in_channels;
            out_chan = in_channels;
        end else if (first_pw) begin
            in_chan = in_channels;
            out_chan = 1 << out_channels;
        end else begin
            // Subsequent Blocks: Use log2 of channels
            in_chan = 1 << in_channels;
            out_chan = 1 << out_channels;
        end

        temporal_length = 1 << time_steps;
        pool_amt[0] = 1 << maxpool;
        pool_amt[1] = 1;
        tap_level = packet + 1;

        // Deduce Convolution Type and Assign Contextual Parameters
        if (in_chan == out_chan && ctx.padding != 0) begin

            // This is a depthwise convolution
            ctx.kernel_size = 2 * ctx.padding + 1;
            ctx.group_size = in_chan;
            ctx.stride = 1;
            ctx.tensor_size = temporal_length * in_chan;
            ctx.in_chan = in_chan;
            ctx.data_context = 0;
            temporal_axis = 0;
            ctx.dots_per_channel = temporal_length;
            ctx.temporal_length = temporal_length;
            pointer_advance = ctx.kernel_size * in_chan;

        end else if (ctx.padding == 0 && temporal_length > 1) begin

            // This is a pointwise convolution
            ctx.kernel_size = in_chan;
            ctx.group_size = temporal_length;
            ctx.stride = in_chan;
            ctx.padding = 0;
            ctx.tensor_size = in_chan * out_chan;
            ctx.in_chan = 1;
            ctx.data_context = 1;
            temporal_axis = 1;
            ctx.dots_per_channel = out_chan;
            ctx.temporal_length = in_chan * out_chan;
            pointer_advance = in_chan * out_chan;

        end else begin

            // This is a fully connected layer
            ctx.kernel_size = in_chan;
            ctx.group_size = out_chan;
            ctx.stride = in_chan;
            ctx.padding = 0;
            ctx.tensor_size = in_chan;
            ctx.in_chan = 1;
            ctx.data_context = 0;
            temporal_axis = 1;
            ctx.dots_per_channel = 1;
            ctx.temporal_length = in_chan;
            pointer_advance = in_chan * out_chan;

        end

        // Determine Optimal Vector Packing Strategy
        begin
            ctx.pack_size = 2 << packet; // The size of each vector to be packed
            ctx.pack_count = 1 << (7 - packet); // The number of packed vectors per batch
            ctx.pack_batch = out_chan * temporal_length >> (7 - packet); // The number of batches per convolution layer
        end

        // We use this to determine the correct offsets to apply when packing vectors for the dot product unit.
        begin
            if (ctx.pack_count < ctx.dots_per_channel) begin
                if (ctx.in_chan > 1) begin
                    ctx.current_channel = current_batch >> batch_chan_ref;
                end else begin
                    ctx.current_channel = 0;
                end
                ctx.current_group = current_batch >> batch_chan_ref;
                ctx.temporal_position = (current_batch * ctx.pack_count - ctx.current_channel * ctx.dots_per_channel) * ctx.stride;
            end else begin
                ctx.current_channel = current_batch * batch_chan_ref;
                ctx.current_group = current_batch * batch_chan_ref;
                ctx.temporal_position = 0;
            end
        end
    end

    // === Vector Packing Routine ===
    // This routine packs the input tensor and weights into vectors for the dot product unit.
    // It uses the context defined above to determine the correct offsets and packing strategy.

    always_comb begin
    automatic int tensor_index;
    automatic int weight_index;
    automatic int kernel_position = 0;
    automatic int pack_position   = 0;
    automatic int kernel_index = ctx.temporal_position;
    automatic int current_channel = ctx.current_channel;
    automatic int current_group = ctx.current_group;
    automatic int tap_offset = -ctx.padding;

    for (int i = 0; i < 256; i++) begin
        // Boundary handling
        if (kernel_position >= ctx.kernel_size) begin
            vector_a[i] = 8'sd0;
            vector_b[i] = 8'sd0;
        end else begin
            if (ctx.data_context == 1'b0) begin
                // === Tensor is input/intermediate ===
                tensor_index = (kernel_index + kernel_position + tap_offset) * ctx.in_chan + ctx.current_channel;
                weight_index = ctx.weight_pointer + kernel_position * ctx.group_size + current_group;

                // Input tensor or intermediate results
                if (tensor_index < 0 || tensor_index >= ctx.tensor_size)
                    vector_a[i] = 8'sd0;
                else if (ctx.first_block)
                    vector_a[i] = input_tensor[tensor_index];
                else
                    vector_a[i] = intermediate_results[tensor_index][ctx.pong];

                // Weight from param_mem
                if (kernel_position >= ctx.kernel_size)
                    vector_b[i] = 8'sd0;
                else
                    vector_b[i] = param_mem[weight_index];

            end else begin
                // === Tensor is param_mem; weights are input/intermediate ===
                tensor_index = (kernel_index + tap_offset) * ctx.in_chan + ctx.current_group;
                weight_index = ctx.weight_pointer + ctx.current_channel * ctx.kernel_size + kernel_position;

                // Tensor from param_mem
                if (kernel_position >= ctx.kernel_size)
                    vector_a[i] = 8'sd0;
                else
                    vector_a[i] = param_mem[weight_index];

                // Weight from input/intermediate
                if (tensor_index < 0 || tensor_index >= ctx.tensor_size)
                    vector_b[i] = 8'sd0;
                else if (ctx.first_block)
                    vector_b[i] = input_tensor[tensor_index];
                else
                    vector_b[i] = intermediate_results[tensor_index][ctx.pong];
            end

        end
            // Update counters
            pack_position++;
            kernel_position++;
            if (pack_position == ctx.pack_size) begin
                pack_position = 0;
                kernel_position = 0;
                kernel_index = kernel_index + ctx.stride;
            end
            if (kernel_index > ctx.temporal_length) begin
                kernel_index = 0;
                current_group++;
                if (in_chan > 1)
                    current_channel++;
            end
        end
    end

    // === The actual convolution operation ===
    always_ff @(posedge clk) begin
        if (reset) begin
            current_batch <= 0;
            layer_complete <= 1'b0;
            instruction_pointer <= 0;
            ctx.weight_pointer <= 0; // Reset weight pointer
            busy <= 1'b0;
            done <= 1'b0;
            ctx.pong <= 0;
            batches_processed <= 0;

        end else if (start) begin
            current_instr <= instr_mem[instruction_pointer];
            busy <= 1'b1;
            done <= 1'b0;

        end else if (busy) begin
            if (valid) begin
                // Process the current batch of convolution
                if (batches_processed < ctx.pack_batch) begin
                    // Perform the dot product operation
                    for (int i = 0; i < 64; i++) begin
                        if (i < ctx.pack_count) begin
                            intermediate_results[batches_processed * ctx.pack_count + i][~ctx.pong] <= dot_product_return[i];
                        end
                    end
                    batches_processed <= batches_processed + 1;
                end else begin
                    // End of the layer, prepare for the next one
                    current_batch <= 0;
                    layer_complete <= 1'b1; // Indicate that the layer processing is done
                    busy <= 1'b0; // Reset busy signal
                end
            end

            if (toggle) begin
                if (current_batch < ctx.pack_batch) begin
                    // Advance to the next batch
                    current_batch <= current_batch + 1;
                end
            end
        end else if (layer_complete) begin
            // Layer processing is complete, reset for the next layer
            ctx.pong <= ~ctx.pong; // Toggle pong for the next layer
            batches_processed <= 0;
            ctx.weight_pointer <= ctx.weight_pointer + pointer_advance; // Advance the weight pointer
            $display("Layer complete. Preparing for next layer...");
            if (instr_mem[instruction_pointer + 1] != 0) begin
                // Load the next instruction
                current_instr <= instr_mem[instruction_pointer + 1];
                instruction_pointer <= instruction_pointer + 1;
                busy <= 1'b1; // Set busy signal for the next layer
            end else begin
                // No more instructions, signal done
                busy <= 1'b0;
                done <= 1'b1;
                inferred_class <= intermediate_results[0][~ctx.pong];
            end
            layer_complete <= 1'b0; // Reset layer complete flag
        end
    end
endmodule