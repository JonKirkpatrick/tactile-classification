module conv_core (
    input logic clk,
    input logic reset,
    input logic start,
    input logic [9:0] switches,
    output logic [7:0] inferred_class,
    output logic busy,
    output logic done
);

    // These figures are based on the intended target hardware: DE10-Lite FPGA
    localparam int PARAM_MEM_DEPTH = 131072;
    localparam int INSTR_MEM_DEPTH = 256;

    // These are the memory blocks for parameters and instructions.
    // In their current form, they are only used for simulation purposes.
    // We will need to replace them with actual BRAM in production.

    initial $readmemh("parameters.hex", param_mem);
    initial $readmemh("instructions.hex", instr_mem);
    logic [31:0] instr_mem [0:INSTR_MEM_DEPTH-1];           // Instruction memory for CNN layers
    logic signed [7:0] param_mem [0:PARAM_MEM_DEPTH-1];     // Parameter memory for weights
    
    // === FSM State and Signals ===
    typedef enum logic [2:0] {
        IDLE,
        LOAD_INSTR,
        CONVOLVE,
        ARGMAX,
        COMPLETE
    } state_t;

    state_t state, next_state;
    
    
    // === Dot Product Engine Signals ===
    logic signed [7:0] vector_a [0:255];                    // Vector A for dot product
    logic signed [7:0] vector_b [0:255];                    // Vector B for dot product
    logic signed [7:0] vector_c [0:255];                    // Vector C for dot product (used for pooling)
    logic signed [7:0] dot_product_return [0:127];          // Dot product results
    logic signed [31:0] raw_output;                         // Raw output for partial products
    logic partial_product;                                  // Flag for a dot product > 256 elements
    logic [3:0] tap_level;                                  // Tap level for the adder tree
    logic valid_rise;                                       // Rising edge of valid signal from the dot product engine
    logic relu_active;                                      // ReLU activation is active
    logic [3:0] shift_amt;                                  // Amount to shift for the adder tree
    logic pool_active;                                      // Pooling is active for the current instruction
    logic idle;                                             // Dot product engine is idle
    
    // === Instruction Handler Signals ===
    logic [31:0] current_instr;                             // Current instruction being processed
    logic [31:0] tensor_size_mid;                           // Size of the intermediate tensor
    logic [31:0] tensor_size_end;                           // Size of the output tensor
    logic [31:0] total_conv_batches;                        // Total number of convolution batches
    logic [31:0] total_pool_batches;                        // Total number of pooling batches
    logic [31:0] vector_a_map;                              // Mapping for vector A
    logic [31:0] vector_b_map;                              // Mapping for vector B
    logic [31:0] vector_c_map;                              // Mapping for vector C
    logic global_avg_pool;                                  // Global average pooling is active
    logic map_pulse;                                        // Pulse for a new conv map index
    logic write_done;                                       // Flag indicating all writes for convolution are done
    logic instr_done;                                       // Current layer processing is complete
    logic packing_mode;                                     // Pooling is active
    logic has_pool;                                         // There is pooling in the current instruction

    // === Input Tensor Signals ===
    logic new_tensor_ready;                                 // Flag indicating a new tensor is ready
    logic signed [7:0] input_tensor [0:8191];               // Input tensor
    
    // === Core Signals and Registers ===
    logic signed [7:0] intermediate_results [0:16383][3];   // Intermediate results buffer
    logic [7:0] instruction_pointer;                        // Pointer to the current instruction being processed
    logic [31:0] current_write_batch;
    logic [31:0] current_read_batch;
    logic [7:0] current_index;
    logic pong;                                             // Toggle target for intermediate results read/write
    logic convolve_done;
    logic [8:0] num_classes;
    logic argmax_done;
    logic [8:0] argmax_index;
    logic [8:0] argmax_max_index;
    logic signed [7:0] argmax_max_value;
    logic [7:0] loop_counter;
    logic       looping;
    logic [7:0] loop_max;
    logic [2:0] batch_pending;


    
    // === Vector Core and Instruction Handler Control Signals ===
    logic new_vector_rise;                                  // Rising edge of new vector signal.
    logic new_vector;
    logic new_vector_d;

    logic new_instr_rise;                                   // Rising edge of new instruction signal.
    logic new_instr;
    logic new_instr_d;
    
    assign new_vector_rise = new_vector & ~new_vector_d;
    assign new_instr_rise = new_instr & ~new_instr_d;

    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            new_vector_d <= 0;
            new_instr_d <= 0;
        end else begin
            new_vector_d <= new_vector;
            new_instr_d <= new_instr;
        end
    end
    
    // === Instantiate Submodules ===
    int8_vector_core int8_core (
        // Inputs
        .clk(clk),
        .rst(reset),
        .input_a(vector_a),
        .input_b(vector_b),
        .input_c(vector_c),
        .tap_level(tap_level),
        .shift_amt(shift_amt),
        .partial_product(partial_product),
        .relu_active(relu_active),
        .pool_active(pool_active),
        .gap_active(global_avg_pool),
        .new_vector_rise(new_vector_rise),
        // Outputs
        .output_packed(dot_product_return),
        .partial_raw_output(raw_output),
        .valid_rise(valid_rise),
        .idle(idle)
    );

    instruction_handler index_map (
        // Inputs
        .clk(clk),
        .rst(reset),
        .start(new_instr_rise),
        .current_instr(current_instr),
        .write_done(write_done),
        // Outputs
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

    tensor_input tensor_input (
        // Inputs
        .clk(clk),
        .reset(reset),
        .start(start),
        .switches(switches),
        // Outputs
        .new_tensor_ready(new_tensor_ready),
        .input_tensor(input_tensor)
    );


    // === Loop Logic to Collect Results from the Dot Product Engine ===

    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            loop_counter        <= 0;
            looping             <= 0;
            write_done          <= 0;
            current_write_batch <= 0;
            batch_pending       <= 0;
        end else if (new_instr_rise || new_instr) begin
            loop_counter        <= 0;
            looping             <= 0;
            write_done          <= 0;
            current_write_batch <= 0;
            batch_pending       <= 0;
        end else begin
            if (batch_pending > 0 && !looping) begin
                batch_pending    <= batch_pending - 1;
                loop_max         <= (256 >> tap_level);
                loop_counter     <= 0;
                looping          <= 1;
            end else if (looping) begin
                if (loop_counter == loop_max - 1) begin
                    looping             <= 0;
                    current_write_batch <= current_write_batch + 1;
                end
                if (!packing_mode && has_pool) begin
                    intermediate_results[loop_counter][2] <= dot_product_return[loop_counter];
                end else begin
                    intermediate_results[loop_counter][!pong] <= dot_product_return[loop_counter];
                end
                loop_counter <= loop_counter + 1;
            end else if (current_write_batch == total_conv_batches) begin
                write_done <= 1; // All convolution batches processed, so we can move to pooling
            end
            if (valid_rise) begin
                batch_pending <= batch_pending + 1;
            end
        end
    end

    // === Loop Logic to Build Input for the Dot Product Engine ===

    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            current_index <= 0;
            current_read_batch <= 0;
            new_vector <= 0;
            pool_active <= 0;
        end else if (new_instr_rise) begin
            current_read_batch <= 0;
            current_index <= 0;
            new_vector <= 0;
            pool_active <= 0;
        end else if (map_pulse) begin
            new_vector <= 0;
            shift_amt <= (global_avg_pool) ? instr_mem[instruction_pointer - 1][11:8] : instr_mem[instruction_pointer - 1][7:4];
            if (packing_mode) begin
                pool_active <= 1;
                if (vector_c_map != 32'hFFFFFFFF) begin
                    if (global_avg_pool) begin
                        vector_c[current_index] <= intermediate_results[vector_c_map][pong];
                    end else begin
                        vector_c[current_index] <= intermediate_results[vector_c_map][2];
                    end
                end else begin
                    vector_c[current_index] <= 0;
                end
            end else begin
                if (vector_a_map != 32'hFFFFFFFF) begin
                    if (instruction_pointer == 1) begin
                        vector_a[current_index] <= input_tensor[vector_a_map];
                    end else begin
                        vector_a[current_index] <= intermediate_results[vector_a_map][pong];
                    end
                    vector_b[current_index] <= param_mem[vector_b_map];
                end else begin
                    vector_a[current_index] <= 0;
                    vector_b[current_index] <= 0;
                end
            end
            if (current_index == 255) begin
                new_vector <= 1;
                current_read_batch <= current_read_batch + 1;
            end
            current_index <= current_index + 1;
        end
    end
    
    // === FSM Logic for Conv Core ===

    // State transition logic
    always_comb begin
        next_state = state;
        case (state)
            IDLE: begin
                if (new_tensor_ready) next_state = LOAD_INSTR;
            end
            LOAD_INSTR: begin
                if (instr_mem[instruction_pointer] == 0) next_state = ARGMAX;
                else next_state = CONVOLVE;
            end
            CONVOLVE: begin
                if (convolve_done) next_state = LOAD_INSTR;
            end
            ARGMAX: begin
                if (argmax_done) next_state = COMPLETE;
            end
            COMPLETE: begin
                next_state = IDLE;
            end
            default: next_state = IDLE;
        endcase
    end

    // State register logic
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            new_instr <= 0;
            instruction_pointer <= 0;
            current_instr <= 0;
            num_classes <= 0;
            argmax_index <= 0;
            argmax_max_index <= 0;
            argmax_max_value <= 0;
            done <= 0;
            busy <= 0;
            pong <= 0;
            partial_product <= 0;
            state <= IDLE;
        end else begin

            state <= next_state;
            partial_product <= 0;  // This just hasn't really been implemented yet
            
            unique case (state)
                IDLE: begin
                    done <= 0;
                    busy <= 0;
                end

                LOAD_INSTR: begin
                    new_instr <= 1;
                    if (instr_mem[instruction_pointer] == 0) begin
                        new_instr <= 0;
                        num_classes <= instr_mem[instruction_pointer + 1];
                    end
                    current_instr <= instr_mem[instruction_pointer];
                    instruction_pointer <= instruction_pointer + 1;
                    busy <= 1;
                    done <= 0;
                    convolve_done <= 0;
                    pong <= ~pong;
                end

                // I had to delay the start of convolution until the next clock cycle due to stale instruction data.
                // Once new_instr_rise is no longer high the total_conv_batches and total_pool_batches will be valid.
                CONVOLVE: begin
                    new_instr <= 0;
                    if (!new_instr_rise && current_write_batch == total_conv_batches + total_pool_batches) begin
                        convolve_done <= 1;
                    end
                end

                ARGMAX: begin
                    if (argmax_index == 0) begin
                        argmax_max_value <= intermediate_results[0][pong];
                        argmax_max_index <= 0;
                        argmax_index <= 1;
                    end else begin
                        if (argmax_index < num_classes) begin
                            if (intermediate_results[argmax_index][pong] > argmax_max_value) begin
                                argmax_max_value <= intermediate_results[argmax_index][pong];
                                argmax_max_index <= argmax_index;
                            end
                            argmax_index <= argmax_index + 1;
                        end else begin
                            inferred_class <= argmax_max_index[8:0];
                            argmax_index <= 0;
                            argmax_done <= 1;
                        end
                    end
                end

                COMPLETE: begin
                    busy <= 0;
                    done <= 1;
                end
            endcase
        end
    end
endmodule