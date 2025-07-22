module uart_loader #(
    parameter ADDR_WIDTH = 8
)(
    input  logic         clk,
    input  logic         rst,

    // UART RX
    input  logic         uart_rx,

    // BRAM interface
    output logic [ADDR_WIDTH-1:0] bram_addr,
    output logic [31:0]           bram_din,
    output logic                  bram_we
);
    // UART receiver
    logic [7:0] uart_data;
    logic       uart_valid;

    uart_rx_simple uart (
        .clk(clk),
        .rst(rst),
        .rx(uart_rx),
        .data(uart_data),
        .valid(uart_valid)
    );

    // FSM to collect 4 bytes and write to BRAM
    logic [1:0] byte_count;
    logic [31:0] word_buffer;
    logic [ADDR_WIDTH-1:0] address;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            byte_count <= 0;
            word_buffer <= 0;
            address <= 0;
            bram_we <= 0;
        end else begin
            bram_we <= 0;
            if (uart_valid) begin
                word_buffer <= {word_buffer[23:0], uart_data};
                byte_count <= byte_count + 1;

                if (byte_count == 2'd3) begin
                    bram_din <= {word_buffer[23:0], uart_data};
                    bram_addr <= address;
                    bram_we <= 1;
                    address <= address + 1;
                end
            end
        end
    end
endmodule