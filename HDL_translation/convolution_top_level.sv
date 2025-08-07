// === Top Level Convolution Module ===
module convolution_top_level (
    input MAX10_CLK1_50,
    input logic [9:0] SW,
    input logic [1:0] KEY,

    output logic [6:0] HEX0,
    output logic [6:0] HEX1,
    output logic [6:0] HEX2,
    output logic [6:0] HEX3,
    output logic [6:0] HEX4,
    output logic [6:0] HEX5
);

	logic KEY0_inverted;
	logic KEY1_inverted;
	assign KEY0_inverted = ~KEY[0];
	assign KEY1_inverted = ~KEY[1];

    // Interface signals for the tensor input module
    logic start_building_tensor;

    // Interface signals for the convolution core
    logic signed [7:0] inferred_class;
    logic busy;
    logic done;

    // Instantiate the convolution core
    conv_core conv_core_inst (
        .clk(MAX10_CLK1_50),
        .reset(KEY1_inverted),
        .start(start_building_tensor),
        .switches(SW),
        .inferred_class(inferred_class),
        .busy(busy),
        .done(done)
    );

    // Instantiate the six-segment display module
    six_seven_seg display_inst (    
        .number(inferred_class),
        .dis_0(HEX0),
        .dis_1(HEX1),
        .dis_2(HEX2),
        .dis_3(HEX3),
        .dis_4(HEX4),
        .dis_5(HEX5)
    );

    always_ff @(posedge MAX10_CLK1_50 or posedge KEY1_inverted) begin
        if (KEY1_inverted) begin
            start_building_tensor <= 0;
        end else begin
            if (KEY0_inverted && !busy) begin
                start_building_tensor <= 1;
            end else begin
                start_building_tensor <= 0;
            end
        end
    end

endmodule