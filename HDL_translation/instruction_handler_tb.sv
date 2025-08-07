// instruction_handler_tb.sv

`timescale 1ns / 1ps

module instruction_handler_tb;

    // Clock and reset
    logic clk = 0;
    logic rst = 1;
    always #5 clk = ~clk; // 100 MHz clock

    // DUT interface
    logic start;
    logic [31:0] current_instr;
    logic valid_rise;
    logic global_avg_pool;
    logic [3:0] tap_level;
    logic packing_mode;
    logic relu_active;
    logic has_pool;
    logic map_pulse;
    logic [31:0] total_conv_batches;
    logic [31:0] total_pool_batches;
    logic [31:0] tensor_size_mid;
    logic [31:0] tensor_size_end;
    logic [31:0] vector_a_map;
    logic [31:0] vector_b_map;
    logic [31:0] vector_c_map;
    logic instr_done;
    
    // Internal Signals
    logic [7:0] current_index;


    // Instantiate the DUT
    instruction_handler uut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .current_instr(current_instr),
        .valid_rise(valid_rise),
        .global_avg_pool(global_avg_pool),
        .tap_level(tap_level),
        .packing_mode(packing_mode),
        .relu_active(relu_active),
        .has_pool(has_pool),
        .map_pulse(map_pulse),
        .total_conv_batches(total_conv_batches),
        .total_pool_batches(total_pool_batches),
        .tensor_size_mid(tensor_size_mid),
        .tensor_size_end(tensor_size_end),
        .vector_a_map(vector_a_map),
        .vector_b_map(vector_b_map),
        .vector_c_map(vector_c_map),
        .instr_done(instr_done)
    );

    // Test procedure
    initial begin
        $display("Starting instruction_handler testbench");

        // Reset
        rst = 1;
        start = 0;

        
        @(posedge clk);
        rst = 0;

        // === TEST CASE 1 ===
        // Based on current_instr = 32'h00120301;
        current_instr = 32'h01111502;
        
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;

        // Wait for a few cycles to allow values to propagate through the FSM
        repeat (5) @(posedge clk);
        $display("Test Case 1: Depthwise convolution with in_channels = 4, padding = 1, time_steps = 5");
        $display("\nData context: %0b", uut.data_context);
        $display("Total batches: %0d", uut.total_conv_batches);
        $display("Total pool batches: %0d", uut.total_pool_batches);
        $display("Group size: %0d", 1 << uut.ctx.group_size);
        $display("Kernel size: %0d", uut.ctx.kernel_size);
        $display("Stride: %0d", 1 << uut.ctx.stride_log2);
        $display("Temporal length: %0d", uut.ctx.temporal_length);
        $display("Temporal log2: %0d", uut.ctx.temporal_log2);
        $display("Dots per channel: %0d", 1 << uut.ctx.dots_per_channel);
        $display("The temporal split: %0d", uut.temporal_split_log2);
        $display("Max_Pooling: %0b", uut.maxpool);
        wait(instr_done);
        $display("ctx.weight_pointer = %0d", uut.ctx.weight_pointer);
        $display("ctx.tensor_size = %0d", uut.ctx.tensor_size);
        $finish;
    end

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            $display("Resetting testbench...");
            current_index <= 0;
        end else begin
            if (map_pulse && !packing_mode) begin
                $display("Current index: %0d, Vector_a_map: %0d and Vector_b_map: %0d", current_index, uut.vector_a_map, uut.vector_b_map);
                //$display("Kernel Index: %0d, Kernel Position: %0d, Pack Position: %0d, Current Channel: %0d, Current Group: %0d", uut.kernel_index, uut.kernel_position, uut.pack_position, uut.current_channel, uut.current_group);
            end else if (map_pulse && packing_mode) begin
                $display("Vector_c_map: %0d", uut.vector_c_map);
            end
        end
    end

endmodule
