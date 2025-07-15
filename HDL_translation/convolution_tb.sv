`timescale 1ns/1ps

module convolution_tb;

    logic clk = 0;
    logic reset = 0;
    logic signed [7:0] input_tensor [0:8191];
    logic [31:0] current_instr;

    // Instantiate your module
    convolution uut (
        .clk(clk),
        .reset(reset),
        .input_tensor(input_tensor),
        .current_instr(current_instr)
    );

    // Clock generator
    always #5 clk = ~clk;

    initial begin
        // Test setup
        reset = 1;
        #10;
        reset = 0;

        // Instruction format recap:
        // 31 = first_block
        // 30 = relu_active
        // 29 = global_avg_pool
        // 28:26 = maxpool (log2)
        // 25:22 = padding
        // 21:18 = temporal length (log2)
        // 17:12 = in_channels (log2 or literal)
        // 11:8  = out_channels (log2)
        // 7:4   = shift
        // 3:0   = packet size (log2 of pack_size - 1)

        // Example: First block, no relu, no GAP, single pooling(kernel = 2),
        // padding=6 (13x1 kernel), time_steps=7 (128),
        // in_channels=13 (literal=13), out_channels=ignored,
        // shift=0, packet=3 (pack_size=16)
        current_instr = 32'b1_0_0_001_0011_0111_001101_0000_1010_0010;

        #10; // Wait for combinational logic to propagate

        $display("Decoded Instruction:");
        $display("  first_block     = %b", uut.first_block);
        $display("  relu_active     = %b", uut.relu_active);
        $display("  global_avg_pool = %b", uut.global_avg_pool);
        $display("  maxpool         = %d", uut.maxpool);
        $display("  padding         = %d", uut.padding);
        $display("  time_steps      = %d (temporal_length = %d)", uut.time_steps, uut.temporal_length);
        $display("  in_channels     = %d (in_chan = %d)", uut.in_channels, uut.in_chan);
        $display("  out_channels    = %d (out_chan = %d)", uut.out_channels, uut.out_chan);
        $display("  kernel_size     = %d", uut.kernel_size);
        $display("  stride          = %d", uut.stride);
        $display("  group_size      = %d", uut.group_size);
        $display("  pack_size       = %d", uut.pack_size);
        $display("  pack_count      = %d", uut.pack_count);
        $display("  pack_padding    = %d", uut.pack_padding);
        $display("  pack_batch      = %d", uut.pack_batch);
        $display("  pack_slack      = %d", uut.pack_slack);

        $finish;
    end
endmodule
