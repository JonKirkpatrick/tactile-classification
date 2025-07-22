module top_level (
    input logic clk,                // Clock signal
    input logic reset_button,       // Reset button on the FPGA board
    input logic uart_rx,            // UART receive signal
    output logic uart_tx,           // UART transmit signal
    output logic [7:0] inferred_class // Output from convolution module
);

    // Internal signals
    logic reset;                    // Reset signal for modules
    logic start;                    // Start signal for convolution module
    logic new_tensor;               // Signal from UART loader
    logic busy, done;               // Status signals from convolution module
    logic signed [7:0] input_tensor [0:8191]; // Input tensor for convolution module

    // Instantiate UART loader
    uart_loader uart_inst (
        .clk(clk),
        .reset(reset),
        .rx(uart_rx),
        .tx(uart_tx),
        .new_tensor(new_tensor),
        .tensor_data(input_tensor)
    );

    // Instantiate convolution module
    convolution conv_inst (
        .clk(clk),
        .reset(reset),
        .start(start),
        .input_tensor(input_tensor),
        .inferred_class(inferred_class),
        .busy(busy),
        .done(done)
    );

    // Reset logic
    always_ff @(posedge clk or posedge reset_button) begin
        if (reset_button)
            reset <= 1'b1;
        else
            reset <= 1'b0;
    end

    // Start signal logic
    always_ff @(posedge clk) begin
        if (new_tensor)
            start <= 1'b1;
        else
            start <= 1'b0;
    end

endmodule