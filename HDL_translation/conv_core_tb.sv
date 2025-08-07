module conv_core_tb;

    // Clock and reset
    logic clk = 0;
    logic reset = 1;
    always #5 clk = ~clk; // 100 MHz clock

    // DUT interface
    logic start;
    logic signed [7:0] input_tensor [0:8191];
    logic [7:0] inferred_class;
    logic busy;
    logic done;
    logic read_delay;


    // Instantiate your module
    conv_core uut (
        .clk(clk),
        .reset(reset),
        .start(start),
        .input_tensor(input_tensor),
        .inferred_class(inferred_class),
        .busy(busy),
        .done(done)
    );
	 
    initial begin

        // Test setup
        reset = 1;
        @(posedge clk);
        reset = 0;
        @(posedge clk);
        $display("Starting conv_core testbench");
        start = 1;
        @(posedge clk);
        start = 0;

        input_tensor[ 0] = 8'd1;
        input_tensor[ 1] = 8'd33;
        input_tensor[ 2] = 8'd2;
        input_tensor[ 3] = 8'd34;
        input_tensor[ 4] = 8'd3;
        input_tensor[ 5] = 8'd35;
        input_tensor[ 6] = 8'd4;
        input_tensor[ 7] = 8'd36;
        input_tensor[ 8] = 8'd5;
        input_tensor[ 9] = 8'd37;
        input_tensor[10] = 8'd6;
        input_tensor[11] = 8'd38;
        input_tensor[12] = 8'd7;
        input_tensor[13] = 8'd39;
        input_tensor[14] = 8'd8;
        input_tensor[15] = 8'd40;
        input_tensor[16] = 8'd9;
        input_tensor[17] = 8'd41;
        input_tensor[18] = 8'd10;
        input_tensor[19] = 8'd42;
        input_tensor[20] = 8'd11;
        input_tensor[21] = 8'd43;
        input_tensor[22] = 8'd12;
        input_tensor[23] = 8'd44;
        input_tensor[24] = 8'd13;
        input_tensor[25] = 8'd45;
        input_tensor[26] = 8'd14;
        input_tensor[27] = 8'd46;
        input_tensor[28] = 8'd15;
        input_tensor[29] = 8'd47;
        input_tensor[30] = 8'd16;
        input_tensor[31] = 8'd48;
        input_tensor[32] = 8'd17;
        input_tensor[33] = 8'd49;
        input_tensor[34] = 8'd18;
        input_tensor[35] = 8'd50;
        input_tensor[36] = 8'd19;
        input_tensor[37] = 8'd51;
        input_tensor[38] = 8'd20;
        input_tensor[39] = 8'd52;
        input_tensor[40] = 8'd21;
        input_tensor[41] = 8'd53;
        input_tensor[42] = 8'd22;
        input_tensor[43] = 8'd54;
        input_tensor[44] = 8'd23;
        input_tensor[45] = 8'd55;
        input_tensor[46] = 8'd24;
        input_tensor[47] = 8'd56;
        input_tensor[48] = 8'd25;
        input_tensor[49] = 8'd57;
        input_tensor[50] = 8'd26;
        input_tensor[51] = 8'd58;
        input_tensor[52] = 8'd27;
        input_tensor[53] = 8'd59;
        input_tensor[54] = 8'd28;
        input_tensor[55] = 8'd60;
        input_tensor[56] = 8'd29;
        input_tensor[57] = 8'd61;
        input_tensor[58] = 8'd30;
        input_tensor[59] = 8'd62;
        input_tensor[60] = 8'd31;
        input_tensor[61] = 8'd63;
        input_tensor[62] = 8'd32;
        input_tensor[63] = 8'd64;

        // Wait for the module to finish processing
        wait(done);
        $display("Inferred class: %d", inferred_class);
        for (int i = 0; i < 64; i++) begin
            $display("Intermediate Results[%0d][%0b]: %0d", i, uut.pong, uut.intermediate_results[i][uut.pong]);
        end
        $finish;
    end

    always @(posedge clk) begin
        if (uut.new_instr_rise) begin
            $display("Test Bench notes: New Instruction Received: %0b", uut.new_instr_rise);
        end
        if(uut.new_vector_rise) begin
            $display("Test Bench notes: New Vector Written: %0b", uut.new_vector_rise);
            $display("has_pool: %b, packing_mode: %b, tap_level: %0d", uut.has_pool, uut.packing_mode, uut.tap_level);
        end
        if (uut.valid_rise) begin
            $display("Test Bench notes: New Results Available: %0b", uut.valid_rise);
            $display("has_pool: %b, packing_mode: %b", uut.has_pool, uut.packing_mode);
        end
    end
endmodule