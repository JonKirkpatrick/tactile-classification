// === Tensor Input Module ===
module tensor_input (
    input logic clk,
    input logic reset,
    input logic start,
    input logic [9:0] switches,
    output logic new_tensor_ready,
    output logic ping,
    output logic signed [7:0] input_tensor [0:8191][2]
);

    localparam int VECTOR_SIZE = 16; // Size of the input vector
    localparam int NUM_VECTORS = 128; // Number of vectors in the input tensor

    // Counter to generate input tensors
    logic [3:0] vector_element_count;
    logic [6:0] vector_count;
    logic signed [7:0] input_vector [0:15];

    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            new_tensor_ready <= 0;
            vector_element_count <= 0;
            vector_count <= 0;
        end else if (start) begin
        end else begin
        end
    end

endmodule