module six_seven_seg(
    input [15:0] number,
    output [6:0] dis_0,
    output [6:0] dis_1,
    output [6:0] dis_2,
    output [6:0] dis_3,
    output [6:0] dis_4,
    output [6:0] dis_5
);

// Define inverted patterns for each decimal digit from 0 to 9
// Each pattern represents which segments are active for that digit,
// with the bits inverted to match the polarity of the LEDs
// Each pattern is a 7-bit value where each bit represents a segment (A to F)
// For example, 'b1000000 represents the number 0 with the bits inverted
// Pattern for 0 to 9 digits (inverted): ~abcdefg
// 'b1000000, 'b1111001, 'b0100100, 'b0110000, 'b0011001, 'b0010010, 'b0000010, 'b1111000, 'b0000000, 'b0010000
reg [6:0] inverted_patterns [0:9] = '{7'b1000000, 7'b1111001, 7'b0100100, 7'b0110000, 7'b0011001, 7'b0010010, 7'b0000010, 7'b1111000, 7'b0000000, 7'b0010000};

// Convert the 16-bit binary input number to decimal digits
reg [15:0] temp_num;
reg [3:0] digits [5:0];
reg [6:0] decimal_sign = 7'b1111111;
integer i;

always @* begin
	if (number[15]) begin
		temp_num = ~number+1;
		decimal_sign = 7'b0111111;
	end
	else begin
		decimal_sign = 7'b1111111;
		 temp_num = number;
	 end
	 for (i = 0; i < 6; i++) begin
			digits[i] = temp_num % 10;
			temp_num = temp_num / 10;
		 end
end

// Assign the inverted display patterns to the outputs based on the decimal digits
assign {dis_0, dis_1, dis_2, dis_3, dis_4, dis_5} = {decimal_sign, inverted_patterns[digits[4]], inverted_patterns[digits[3]], inverted_patterns[digits[2]], inverted_patterns[digits[1]], inverted_patterns[digits[0]]};

endmodule