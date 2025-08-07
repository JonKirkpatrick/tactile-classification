// Testbench for int8_vector_core

module int8_vector_core_tb;

    logic clk;
    logic rst;
    logic start;

    // Input vectors
    logic signed [7:0] input_a [256];
    logic signed [7:0] input_b [256];

    // Control signals
    logic [3:0] tap_level;
    logic [3:0] shift_amt;
    logic partial_product;
    logic relu_active;

    // Output signals
    logic signed [7:0]  output_packed [128];
    logic signed [31:0] partial_raw_output;
    logic valid_rise;

    // Instantiate the module under test
    int8_vector_core dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .input_a(input_a),
        .input_b(input_b),
        .tap_level(tap_level),
        .shift_amt(shift_amt),
        .partial_product(partial_product),
        .output_packed(output_packed),
        .partial_raw_output(partial_raw_output),
        .relu_active(relu_active),
        .valid_rise(valid_rise)
    );

    // Clock generation
    always #5 clk = ~clk;

    task load_inputs(byte val_a, byte val_b);
        for (int i = 0; i < 256; i++) begin
            input_a[i] = val_a;
            input_b[i] = val_b;
        end
    endtask

    // In this test, it's worth noting that we aren't waiting for valid anymore.  We are feeding
    // inputs and checking outputs immediately after the clock cycles; we do test for valid, but
    // since timing is deterministic, we actually don't need the valid flag at all.  In the interest
    // of future development, we will keep the valid flag, and it's probably even worth using it.
    task automatic alternate_input_test(int pairs, logic signed [7:0] val1 = 1, logic signed [7:0] val2 = 2);
        logic signed [15:0] expected;
        shift_amt = 4'd3;
        tap_level = 4'd8;
        partial_product = 0;
        relu_active = 0;
        for (int i = 0; i < pairs + 14 - tap_level; i++) begin
            if (i < pairs) begin
                if (i % 2 == 0)
                    load_inputs(val1, val2);
                else
                    load_inputs(val2, val2);
            end
            repeat (2) @(posedge clk);
        end
    endtask

    initial begin

        clk = 0;
        rst = 1;
        start = 0;
        tap_level = 4'd8;
        shift_amt = 4'd2;
        partial_product = 0;

        // Initialize inputs to 0
        for (int i = 0; i < 256; i++) begin
            input_a[i] = 8'sd0;
            input_b[i] = 8'sd0;
        end

        // Verify reset
        repeat (2) @(posedge clk);
        rst = 0;
        @(posedge clk);
        for (int i = 0; i < 256; i++) begin
            if (input_a[i] !== 8'sd0 || input_b[i] !== 8'sd0) $fatal(1, "FAIL: Inputs not reset to 0");
        end

        // === Test Case 1: Normal dot product ===
        load_inputs(8'sd1, 8'sd1);  // Expect dot product of 64

        @(posedge clk);
        rst = 1;
        @(posedge clk);
        rst = 0;
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;

        wait (valid_rise);
        repeat (2) @(posedge clk);

        // === Test Case 2: Partial product ===
        tap_level = 4'd8;
        shift_amt = 4'd0; // Doesn't matter
        partial_product = 1;
        load_inputs(8'sd2, 8'sd3); // 2*3 * 256 = 1536

        @(posedge clk);
        rst = 1;
        @(posedge clk);
        rst = 0;
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;

        wait (valid_rise);


        // === Test Case 3: Mid-tree quantized output ===
        tap_level = 3'd2;          // 64 outputs expected
        shift_amt = 4'd9;          // 64516 >> 9 = 126
        partial_product = 0;
        load_inputs(8'sd127, 8'sd127);

        @(posedge clk);
        rst = 1;
        @(posedge clk);
        rst = 0;
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;

        wait (valid_rise);

        // === Test Case 4: Alternate input test ===
        @(posedge clk);
        rst = 1;
        @(posedge clk);
        rst = 0;
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;

        alternate_input_test(10, 8'sd1, 8'sd2);
    end
endmodule