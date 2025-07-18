`timescale 1ns/1ps

module convolution_tb;

    logic clk = 0;
    logic reset = 0;
    logic signed [7:0] input_tensor [0:8191];
    logic [63:0] current_instr;
    logic [7:0] inferred_class;
    logic busy;
    logic done;

    // Instantiate your module
    convolution uut (
        .clk(clk),
        .reset(reset),
        .input_tensor(input_tensor),
        .current_instr(current_instr),
        .inferred_class(inferred_class),
        .busy(busy),
        .done(done)
    );

    // Clock generator
    always #5 clk = ~clk;

    initial begin
        // Test setup
        reset = 1;
        #10;
        reset = 0;

        for (int i = 0; i < 1664; i++) begin
            input_tensor[i] = 8'sd1;
        end        

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
        current_instr = 64'b_0011_0000_0000_0000_0000_0000_0000_0000_1_0_0_000_0110_0111_001101_0000_1010_0011;
        // current_instr =    64'b_0011_1000_0000_0000_0000_0000_0000_0000_0_1_0_010_0000_0111_001101_0110_0000_0011;
        #10; // Wait for combinational logic to propagate

        $display("Decoded Instruction:");
        $display("  first_block     = %b", uut.ctx.first_block);
        $display("  relu_active     = %b", uut.relu_active);
        $display("  global_avg_pool = %b", uut.global_avg_pool);
        $display("  maxpool         = %d", uut.maxpool);
        $display("  padding         = %d", uut.ctx.padding);
        $display("  time_steps      = %d (temporal_length = %d)", uut.time_steps, uut.temporal_length);
        $display("  in_channels     = %d (in_chan = %d)", uut.in_chan, uut.ctx.in_chan);
        $display("  out_channels    = %d (out_chan = %d)", uut.out_chan, uut.out_chan);
        $display("  kernel_size     = %d", uut.ctx.kernel_size);
        $display("  stride          = %d", uut.ctx.stride);
        $display("  group_size      = %d", uut.ctx.group_size);
        $display("  pack_size       = %d", uut.ctx.pack_size);
        $display("  pack_count      = %d", uut.ctx.pack_count);
        $display("  pack_batch      = %d", uut.ctx.pack_batch);
        $display("  tap_level       = %d", uut.tap_level);
        $display("  shift_amt       = %d", uut.shift_amt);
        $display("  batch_chan_ref  = %d", uut.batch_chan_ref);
        $display("  current_batch   = %d", uut.current_batch);
        $display("  current_channel = %d", uut.ctx.current_channel);
        $display("  temporal_position = %d", uut.ctx.temporal_position);
        $display("  current_group = %d", uut.ctx.current_group);
        $display("  dots_per_channel = %d", uut.ctx.dots_per_channel);
        $display("  first_pw = %b", uut.first_pw);
        $display("  vector_a = %p", uut.vector_a);
        $display("  vector_b = %p", uut.vector_b);
        $finish;
    end
endmodule
