`timescale 1ns/1ps

module convolution_tb;

    logic clk = 0;
    logic reset = 0;
    logic signed [7:0] input_tensor [0:8191];
    logic signed [7:0] inferred_class;
    logic busy;
    logic done;
    logic start;

    // Instantiate your module
    convolution uut (
        .clk(clk),
        .reset(reset),
        .start(start),
        .input_tensor(input_tensor),
        .inferred_class(inferred_class),
        .busy(busy),
        .done(done)
    );

    // Clock generator
    always #5 clk = ~clk;

    initial begin
        // Test setup
        reset = 1;
        @(posedge clk);
        reset = 0;
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;

        for (int i = 0; i < 1664; i++) begin
            input_tensor[i] = i % 256 - 128; // Example input data
        end        
        @(posedge clk);
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
        wait(done);
        $display("Done signal received. Test completed successfully.");
        for (int i = 1600; i < 1665; i++) begin
            $display("intermediate[%0d] = %0d", i, uut.intermediate_results[i][uut.ctx.pong]);
        end
        $display("inferred_class = %d", inferred_class);
        $finish;
    end
endmodule
