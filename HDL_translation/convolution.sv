module convolution (
    input logic clk,
    input logic reset,
    input logic signed [7:0] input_tensor [0:8191],
    input logic [31:0] current_instr
);

    // These figures are based on the intended target hardware: DE10-Lite FPGA
    // We intend to use BRAM for parameters, instructions and intermediate results.
    localparam int PARAM_MEM_DEPTH = 131072;
    localparam int INSTR_MEM_DEPTH = 256;

    //initial $readmemh("parameters.hex", param_mem);
    //initial $readmemh("instructions.hex", instr_mem);

    // Instruction Format:
    // 1-bit  [31]     - First block (1) or not (0)
    // 1-bit  [30]     - Apply ReLU (1) or not (0)
    // 1-bit  [29]     - Global Average Pooling (1) or not (0)
    // 3-bits [28:26]  - MaxPool log2 size (0 for no pooling)
    // 4-bits [25:22]  - Padding (Implies a max kernel size of 31)
    // 4-bits [21:18]  - Temporal length (log2 of time steps) (Max literal of 32768)
    // 6-bits [17:12]  - Channels in (log2 of input channels or literal if first block)
    // 4-bits [11:8]   - Channels out (log2 of output channels)
    // 4-bits [7:4]    - Shift amount for quantization to int8
    // 4-bits [3:0]    - Packet size (ceiling of the (log2 of the kernel size) - 1)

    logic data_context;
    logic temporal_axis;
    logic signed [7:0] param_mem [0:PARAM_MEM_DEPTH-1]; // Parameter memory for weights
    logic [31:0] instr_mem [0:INSTR_MEM_DEPTH-1]; // Instruction memory for CNN blocks
    //logic [31:0] current_instr; // Current instruction being executed
    logic [31:0] address_pointer [0:1]; // Address pointers for tensors and parameters
    logic [7:0] intermediate_results [0:16383][2]; // Intermediate results buffer

    // Interpretted Instruction Fields
    logic [5:0] in_chan;
    logic [5:0] out_chan;
    logic [5:0] kernel_size;
    logic [7:0] group_size;
    logic [9:0] stride;
    logic [4:0] kernel_padding;
    logic [31:0] actual_tensor_size;
    logic [15:0] temporal_length;
    logic [4:0] pool_amt [2];
    logic [8:0] pack_size;
    logic [7:0] pack_count;
    logic [8:0] pack_padding;
    logic [8:0] pack_slack;
    logic [31:0] pack_batch;

    // Raw Instruction Fields
    logic first_block = current_instr[31];
    logic relu_active = current_instr[30];
    logic global_avg_pool = current_instr[29];
    logic [2:0] maxpool = current_instr[28:26];
    logic [3:0] padding = current_instr[25:22];
    logic [3:0] time_steps = current_instr[21:18];
    logic [5:0] in_channels = current_instr[17:12];
    logic [3:0] out_channels = current_instr[11:8];
    logic [3:0] shift_amt = current_instr[7:4];
    logic [3:0] packet = current_instr[3:0];


    always_comb begin
        first_block = current_instr[31];
        relu_active = current_instr[30];
        global_avg_pool = current_instr[29];
        maxpool = current_instr[28:26];
        padding = current_instr[25:22];
        time_steps = current_instr[21:18];
        in_channels = current_instr[17:12];
        out_channels = current_instr[11:8];
        shift_amt = current_instr[7:4];
        packet = current_instr[3:0];

        if (first_block) begin
            // First Block: Use literal for in_channels
            in_chan = in_channels;
            out_chan = in_channels;
        end else begin
            // Subsequent Blocks: Use log2 of in_channels
            in_chan = 1 << in_channels;
            out_chan = 1 << out_channels;
        end

        temporal_length = 1 << time_steps;
        actual_tensor_size = temporal_length * in_chan;
        pool_amt[0] = 1 << maxpool;
        pool_amt[1] = 1;
        if (in_chan == out_chan && padding != 0) begin

            // This is a depthwise convolution
            kernel_size = 2 * padding + 1;
            group_size = in_chan;
            stride = 1;
            kernel_padding = padding;
            data_context = 0;
            temporal_axis = 0;
            actual_tensor_size = temporal_length * in_chan;

        end else if (padding == 0 && temporal_length > 1) begin

            // This is a pointwise convolution
            kernel_size = in_chan;
            group_size = temporal_length;
            stride = in_chan;
            kernel_padding = 0;
            actual_tensor_size = in_chan * out_chan;
            data_context = 1;
            temporal_axis = 1;
        end else if (temporal_length == 1) begin

            // This is a fully connected layer
            kernel_size = in_chan;
            group_size = out_chan;
            stride = in_chan;
            kernel_padding = 0;
            data_context = 0;
            temporal_axis = 1;
            actual_tensor_size = in_chan;
        end
        begin
            // Determine Optimal Packing Strategy
            pack_size = 2 << packet;
            pack_count = 1 << 7 - packet;
            pack_padding = pack_size - kernel_size;
            pack_batch = out_chan * temporal_length >> (7 - packet);
            pack_slack = (pack_batch * pack_count - out_chan * temporal_length * pack_size) * pack_size;
        end
    end
endmodule