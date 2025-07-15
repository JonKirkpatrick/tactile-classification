module convolution (
    input logic clk,
    input logic reset
);

    // These figures are based on the intended target hardware: DE10-Lite FPGA
    // We intend to use BRAM for parameters, instructions and intermediate results.
    localparam int PARAM_MEM_DEPTH = 131072;
    localparam int INSTR_MEM_DEPTH = 256;

    initial $readmemh("parameters.hex", param_mem);
    initial $readmemh("instructions.hex", instr_mem);

    // Instruction Format:
    // 1-bit  [31]     - First block (1) or not (0)
    // 1-bit  [30]     - Apply ReLU (1) or not (0)
    // 3-bits [29:27]  - MaxPool log2 size (0 for no pooling, 1 for GAP)
    // 4-bits [26:23]  - Padding
    // 4-bits [22:19]  - Temporal length (log2 of time steps)
    // 6-bits [18:13]  - Channels in (log2 of input channels or literal if first block)
    // 4-bits [12:9]   - Channels out (log2 of output channels)
    // 4-bits [8:5]    - Shift amount for quantization to int8
    // 5-bits [4:0]    - Reserved for future use

    logic data_context;
    logic temporal_axis;
    logic signed [7:0] param_mem [0:PARAM_MEM_DEPTH-1]; // Parameter memory for weights
    logic [31:0] instr_mem [0:INSTR_MEM_DEPTH-1]; // Instruction memory for CNN blocks
    logic [31:0] current_instr; // Current instruction being executed
    logic [31:0] address_pointer [0:1]; // Address pointers for tensors and parameters

    // Decoded instruction fields
    wire first_block;
    wire apply_relu;
    wire [2:0] maxpool;
    wire [3:0] padding;
    wire [3:0] time_steps;
    wire [5:0] in_channels;
    wire [3:0] out_channels;
    wire [3:0] shift_amt;

    logic [5:0] in_chan;
    logic [5:0] out_chan;
    logic [5:0] kernel_size;
    logic [7:0] group_size;
    logic [9:0] stride;
    logic [4:0] kernel_padding;
    logic [31:0] actual_tensor_size;
    logic [15:0] temporal_length;
    logic [4:0] pool_amt [2];

    always_comb begin
        if (first_block) begin
            // First Block: Use literal for in_channels
            in_chan = in_channels;
        end else begin
            // Subsequent Blocks: Use log2 of in_channels
            in_chan = 1 << in_channels;
        end
        temporal_length = 1 << time_steps;
        out_chan = 1 << out_channels;
        actual_tensor_size = temporal_length * in_chan;
        pool_amt[0] = 1 << maxpool;
        pool_amt[1] = 1;
        if (in_chan == out_chan and padding != 0) begin

            // This is a depthwise convolution
            kernel_size = 2 * padding + 1;
            group_size = in_chan;
            stride = 1;
            kernel_padding = padding;
            data_context = 0;
            temporal_axis = 0;

        end else if (in_chan != out_chan and temporal_length > 1) begin

            // This is a pointwise convolution
            kernel_size = in_chan;
            group_size = temporal_length;
            stride = in_chan;
            kernel_padding = 0;
            data_context = 1;
            temporal_axis = 1;
        end else begin

            // This is a fully connected layer
            kernel_size = in_chan;
            group_size = out_chan;
            stride = in_chan;
            kernel_padding = 0;
            data_context = 0;
            temporal_axis = 1;
        end


    end
endmodule