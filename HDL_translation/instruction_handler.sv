typedef struct packed {
    logic [31:0] weight_pointer;                // Pointer to the first weight in layer's parameters
    logic [13:0] pointer_advance;               // The amount to advance the parameter pointer after current layer

    logic [31:0]  kernel_size;                  // Size of the kernel for the convolution
    logic [31:0]  padding;                      // Padding size
    logic [31:0]  stride_log2;                  // Log2 of the stride for the convolution

    logic [31:0] tensor_size;                   // Number of elements in the input tensor
    logic [31:0] in_chan_log2;                  // Log2 of the number of input channels
    logic [31:0] group_size;                    // Size of the group for the convolution
    logic [31:0] temporal_length;               // Length of the temporal dimension
    logic [31:0] temporal_log2;                 // Log2 of the temporal length
    logic [31:0] dots_per_channel;              // The number of dot products per channel

    logic [31:0]  pack_size;                     // Size of each vector to be packed
    logic [31:0]  pack_count;                    // Number of packed vectors per batch

    logic [31:0]  current_channel;               // Current channel being processed
    logic [31:0]  temporal_position;             // Current position in the temporal dimension
    logic [31:0]  current_group;                 // Current group being processed

    logic [31:0]  pool_batches_per_sweep_log2;   // Number of pooling batches required to cover the output channels once
    logic [31:0]  sweeps_per_pool_batch_log2;    // Number of times a single pooling batch covers all output channels
    logic [31:0]  sweeps_per_pool_batch;         // Number of times a single pooling batch covers all output channels
    logic [31:0]  channels_covered_per_sweep;    // Number of output channels covered in a single sweep
    logic [31:0]  pool_channel_offset;           // The base channel of the current pooling operation
    logic [31:0]  pool_temporal_offset;          // The base index of the current pooling operation
} conv_context_t;

module instruction_handler (
    input logic         clk,
    input logic         rst,
    input logic         start,
    input logic [31:0]  current_instr,
    input logic         write_done,
    output logic        global_avg_pool,        // 1 = global average pooling, 0 = convolution
    output logic [3:0]  tap_level,              // 2 to 8
    output logic        packing_mode,           // 0 convolution, 1 pooling
    output logic        relu_active,            // 1 = ReLU activation is active
    output logic        has_pool,               // 1 = pooling exists, 0 = no pooling
    output logic        map_pulse,              // Pulse to indicate a new conv map index is available
    output logic [31:0] total_conv_batches,     // A maximum of 32 bits to accommodate 1024 batches
    output logic [31:0] total_pool_batches,     // A maximum of 32 bits to accommodate 1024 batches
    output logic [31:0] tensor_size_mid,        // Number of elements in the intermediate tensor
    output logic [31:0] tensor_size_end,        // Number of elements in the output tensor
    output logic [31:0] vector_a_map,           // 32 bits to address input/intermediate results buffer
    output logic [31:0] vector_b_map,           // 32 bits to address weights buffer
    output logic [31:0] vector_c_map,           // 32 bits to address intermediate results buffer
    output logic        instr_done
);

    // Internal Fields
    conv_context_t  ctx;                         // Transformed shape and context for convolution
    logic            data_context;               // 0 = input/intermediate tensor, 1 = param_mem
    logic [31:0]     temporal_split_log2;
    logic [31:0]     packs_per_batch_log2;
    logic [31:0]     pooling_mod;                // Modulo for pooling operations
    logic [31:0]     pooling_split_log2;         // Log2 of the pooling split
    logic [31:0]     current_conv_batch;
    logic [31:0]     current_pool_batch;

    // Raw Instruction Fields
    logic [3:0]     maxpool;
    logic [4:0]     in_channels;
    logic [4:0]     out_channels;
    logic [3:0]     time_steps;
    logic [3:0]     packet_log2;

    // === Loop Counter Registers for Convolution ===
    // We use signed counters to account for a negative tap offset
    logic [31:0]         conv_map_index;
    logic [31:0]         previous_conv_map_index;
    logic signed [31:0]  kernel_position;
    logic signed [31:0]  pack_position;
    logic signed [31:0]  kernel_index;
    logic signed [31:0]  current_channel;
    logic signed [31:0]  current_group;
    logic signed [31:0]  tap_offset;
    logic signed [31:0]  tensor_index;
    logic signed [31:0]  weight_index;

    logic [31:0] pool_map_index;
    logic [31:0] previous_pool_map_index;
    logic signed [31:0] pool_base_index;
    logic signed [31:0] pool_pack_index;
    logic signed [31:0] j; // Loop counter for pooling
    logic signed [31:0] k; // Loop counter for pooling


// === Done signals (one per stage) ===
logic read_done, update_done, update_derived_done, derive_conv_done, 
    conv_done, conv_stream_done, derive_pool_done, pool_done, pool_stream_done;

// === Registered (delayed) versions of done signals ===
logic read_done_d, update_done_d, update_derived_done_d, derive_conv_done_d, 
    conv_done_d, conv_stream_done_d, derive_pool_done_d, pool_done_d, pool_stream_done_d;

// === Rising Edges ===
logic read_done_rise, update_done_rise, update_derived_done_rise, derive_conv_done_rise, 
    conv_done_rise, conv_stream_done_rise, derive_pool_done_rise, pool_done_rise, pool_stream_done_rise;

// === Rising edge detection ===
assign read_done_rise           =  read_done            & ~read_done_d;
assign update_done_rise         =  update_done          & ~update_done_d;
assign update_derived_done_rise =  update_derived_done  & ~update_derived_done_d;
assign derive_conv_done_rise    =  derive_conv_done     & ~derive_conv_done_d;
assign conv_done_rise           =  conv_done            & ~conv_done_d;
assign conv_stream_done_rise    =  conv_stream_done     & ~conv_stream_done_d;
assign derive_pool_done_rise    =  derive_pool_done     & ~derive_pool_done_d;
assign pool_done_rise           =  pool_done            & ~pool_done_d;
assign pool_stream_done_rise    =  pool_stream_done     & ~pool_stream_done_d;

// === Delay register ===
always_ff @(posedge clk or posedge rst) begin
    if (rst) begin
        read_done_d             <= 1'b0;
        update_done_d           <= 1'b0;
        update_derived_done_d   <= 1'b0;
        derive_conv_done_d      <= 1'b0;
        conv_done_d             <= 1'b0;
        conv_stream_done_d      <= 1'b0;
        derive_pool_done_d      <= 1'b0;
        pool_done_d             <= 1'b0;
        pool_stream_done_d      <= 1'b0;
    end else begin
        read_done_d             <= read_done;
        update_done_d           <= update_done;
        update_derived_done_d   <= update_derived_done;
        derive_conv_done_d      <= derive_conv_done;
        conv_done_d             <= conv_done;
        conv_stream_done_d      <= conv_stream_done;
        derive_pool_done_d      <= derive_pool_done;
        pool_done_d             <= pool_done;
        pool_stream_done_d      <= pool_stream_done;
    end
end

    typedef enum logic [3:0] {
        IDLE,
        READ_INSTR,
        UPDATE_CTX,
        UPDATE_DERIVED,
        DERIVE_CONV_PACK,
        CONV_MAP,
        CONV_STREAM,
        DERIVE_POOL_PACK,
        POOL_MAP,
        POOL_STREAM,
        DONE
    } state_t;

    state_t state, next_state;

    always_ff @(posedge clk or posedge rst) begin
        if (rst)
            state <= IDLE;
        else
            state <= next_state;
    end
    
    always_comb begin
		  
		next_state = state;

        case (state)
		  
            IDLE: begin
                if (start) next_state = READ_INSTR;
            end

            // Read the instruction and populate the raw instruction fields
            // Occurs once per instruction
            READ_INSTR: begin
                if (read_done_rise) next_state = UPDATE_CTX;
            end

            // Update the context based on the instruction
            // This includes deducing the convolution type and assigning contextual parameters
            // Occurs once per instruction
            UPDATE_CTX: begin
                if (update_done_rise) next_state = UPDATE_DERIVED;
            end

            // Update derived parameters based on the context
            // This includes determining if a batch will span multiple channels or just a portion of a channel
            // Occurs once per instruction
            UPDATE_DERIVED: begin
                if (update_derived_done_rise) begin
                    if (global_avg_pool) begin
                        next_state = DERIVE_POOL_PACK;
                    end else begin
                        next_state = DERIVE_CONV_PACK;
                    end
                end else begin
                    next_state = UPDATE_DERIVED;
                end
            end

            // Derive the packing parameters for convolution
            // This is where we determine current offsets for the next batch of a convolution
            // Occurs once per batch
            DERIVE_CONV_PACK: begin
                if (derive_conv_done_rise) next_state = CONV_MAP;
            end

            // Reset counter values and prepare for convolution mapping
            CONV_MAP: begin
                if (conv_done_rise) next_state = CONV_STREAM;
            end

            // Stream vector index data one at a time to fill vectors for the dot product unit
            // Occurs until all 256 vector elements are indexed
            CONV_STREAM: begin
                if (conv_stream_done_rise) begin
                    if (current_conv_batch < total_conv_batches) begin
                        next_state = DERIVE_CONV_PACK;
                    end else if (maxpool > 0) begin
                        next_state = DERIVE_POOL_PACK;
                    end else begin
                        next_state = DONE;
                    end
                end else begin
                    next_state = CONV_STREAM;
                end
            end

            DERIVE_POOL_PACK: begin
                if (derive_pool_done_rise) begin
                    next_state = POOL_MAP;
                end else begin
                    next_state = DERIVE_POOL_PACK;
                end
            end

            POOL_MAP: begin
                if (pool_done_rise) next_state = POOL_STREAM;
            end

            POOL_STREAM: begin
                if (pool_stream_done_rise) begin
                    if (current_pool_batch < total_pool_batches) begin
                        next_state = DERIVE_POOL_PACK;
                    end else begin
                        next_state = DONE;
                    end
                end else begin
                    next_state = POOL_STREAM;
                end
            end

            DONE: begin
                if (instr_done) begin
                    next_state = IDLE;
                end else begin
                    next_state = DONE;
                end
            end

            default: next_state = IDLE;
        endcase
    end

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            instr_done <= 0;
            ctx.weight_pointer <= 0;
            current_conv_batch <= 0;
            current_pool_batch <= 0;
            packing_mode <= 0;
            read_done <= 0;
            update_done <= 0;
            update_derived_done <= 0;
            derive_conv_done <= 0;
            conv_done <= 0;
            derive_pool_done <= 0;
            pool_done <= 0;
            conv_stream_done <= 0;
            pool_stream_done <= 0;
        end else begin
            unique case (next_state)
                IDLE: begin
                    current_conv_batch <= 0;
                    current_pool_batch <= 0;
                    read_done <= 0;
                    update_done <= 0;
                    update_derived_done <= 0;
                    derive_conv_done <= 0;
                    conv_done <= 0;
                    conv_stream_done <= 0;
                    derive_pool_done <= 0;
                    pool_done <= 0;
                    pool_stream_done <= 0;

                end

                READ_INSTR: begin
                    // Populate Raw Instruction Fields
                    global_avg_pool <= current_instr[28];
                    maxpool <= current_instr[27:24];
                    in_channels <= current_instr[23:20];
                    out_channels <= current_instr[19:16];
                    ctx.padding <= current_instr[15:12];
                    time_steps <= current_instr[11:8];
                    packet_log2 <= current_instr[3:0];
                    packing_mode <= 0;
                    read_done <= 1;
                end

                UPDATE_CTX: begin
                    // Deduce Convolution Type and Assign Contextual Parameters
                    instr_done <= 0;
                    has_pool <= (maxpool > 0);
                    if (ctx.padding != 0) begin

                        // This is a depthwise convolution
                        data_context <= 0;
                        relu_active <= 1'b0;
                        ctx.kernel_size <= (ctx.padding << 1) + 1;
                        ctx.group_size <= in_channels;
                        ctx.stride_log2 <= 0;
                        ctx.tensor_size <= 1 << (time_steps + in_channels);
                        tensor_size_mid <= 1 << (time_steps + in_channels);
                        tensor_size_end <= 1 << (time_steps + in_channels);
                        ctx.in_chan_log2 <= in_channels;
                        ctx.dots_per_channel <= time_steps;
                        ctx.temporal_length <= 1 << time_steps;
                        ctx.temporal_log2 <= time_steps;
                        ctx.pointer_advance <= ((ctx.padding << 1) + 1) << in_channels;

                    end else if (ctx.padding == 0 && time_steps > 0 && ~global_avg_pool) begin

                        // This is a pointwise convolution
                        data_context <= 1;
                        relu_active <= 1'b1;
                        ctx.kernel_size <= 1 << in_channels;
                        ctx.group_size <= time_steps;
                        ctx.stride_log2 <= in_channels;
                        ctx.tensor_size <= 1 << (in_channels + out_channels);
                        tensor_size_mid <= 1 << (time_steps + out_channels);
                        tensor_size_end <= 1 << (time_steps + out_channels - maxpool);
                        ctx.in_chan_log2 <= 0;
                        ctx.dots_per_channel <= out_channels;
                        ctx.temporal_length <= 1 << (in_channels + out_channels);
                        ctx.temporal_log2 <= packet_log2 + out_channels;
                        ctx.pointer_advance <= 1 << (in_channels + out_channels);

                    end else if (~global_avg_pool) begin

                        // This is a fully connected layer
                        data_context <= 0;
                        relu_active <= 1'b0;
                        ctx.kernel_size <= 1 << in_channels;
                        ctx.group_size <= out_channels;
                        ctx.stride_log2 <= in_channels;
                        ctx.tensor_size <= 1 << (in_channels);
                        tensor_size_mid <= 1 << (out_channels);
                        tensor_size_end <= 1 << (out_channels);
                        ctx.in_chan_log2 <= 0;
                        ctx.dots_per_channel <= 1;
                        ctx.temporal_length <= 1 << (in_channels);
                        ctx.temporal_log2 <= in_channels;
                        ctx.pointer_advance <= 1 << (in_channels + out_channels);

                    end else begin

                        // This is a global average pooling layer
                        data_context <= 0;
                        relu_active <= 1'b0;
                        ctx.kernel_size <= 1 << time_steps;
                        ctx.group_size <= out_channels;
                        ctx.stride_log2 <= 0;
                        ctx.tensor_size <= 1 << (time_steps + in_channels);
                        tensor_size_mid <= 1 << (in_channels);
                        tensor_size_end <= 1 << (in_channels);
                        ctx.in_chan_log2 <= in_channels;
                        ctx.dots_per_channel <= 1;
                        ctx.temporal_length <= 1 << time_steps;
                        ctx.temporal_log2 <= time_steps;
                        ctx.pointer_advance <= 0;

                    end
                    ctx.sweeps_per_pool_batch_log2 <= (8 - maxpool - out_channels) < 0 ? 0 : (8 - maxpool - out_channels);
                    ctx.pool_batches_per_sweep_log2 <= (out_channels + maxpool - 8) < 0 ? 0 : (out_channels + maxpool - 8);
                    ctx.channels_covered_per_sweep <= (8 - maxpool);
                    pooling_split_log2 <= out_channels + maxpool;
                    update_done <= 1;
                end

                UPDATE_DERIVED: begin
                    update_done <= 0;
                    // Determine Optimal Vector Packing Strategy
                    // This is static per instruction.
                    begin
                        ctx.pack_size <= 1 << packet_log2; // The size of each vector to be packed
                        ctx.pack_count <= 1 << (8 - packet_log2); // The number of packed vectors per batch
                        if ((ctx.group_size + ctx.temporal_log2 + packet_log2 - ctx.stride_log2) > 8) begin
                            total_conv_batches <= 1 << (ctx.group_size + ctx.temporal_log2 + packet_log2 - ctx.stride_log2 - 8);
                        end else if (global_avg_pool) begin
                            total_conv_batches <= 0;
                        end else begin
                            total_conv_batches <= 1;
                        end
                        if (maxpool > 0) begin
                            if ((out_channels + time_steps) < (maxpool + 8)) begin
                                total_pool_batches <= 1;
                            end else begin
                                total_pool_batches <= 1 << (out_channels + time_steps - maxpool - 8);
                            end
                        end else begin
                            total_pool_batches <= 0;
                        end
                    end
                    packs_per_batch_log2 = 8 - packet_log2;
                    if ((ctx.temporal_log2 + packet_log2 - ctx.stride_log2) >= 8) begin
                        temporal_split_log2 = ctx.temporal_log2 + packet_log2 - ctx.stride_log2 - 8;
                    end else begin
                        temporal_split_log2 = 8 - (ctx.temporal_log2 + packet_log2 - ctx.stride_log2);
                    end
                    pooling_mod <= (1 << ctx.pool_batches_per_sweep_log2) - 1;
                    update_derived_done <= 1;
                end

                DERIVE_CONV_PACK: begin
                    conv_done <= 0;
                    conv_stream_done <= 0;
                    tap_level <= packet_log2;
                    // We use this to determine the correct offsets to apply when packing conv vectors for the dot product unit.
                    if (ctx.temporal_log2 + packet_log2 - ctx.stride_log2 > 8) begin
                        if (ctx.in_chan_log2 > 0) begin
                            ctx.current_channel <= current_conv_batch >> temporal_split_log2;
                            if (current_conv_batch == 0 || 
                                (((current_conv_batch << packs_per_batch_log2) - ((current_conv_batch >> temporal_split_log2) << ctx.dots_per_channel)) << ctx.stride_log2) == ctx.temporal_length) begin
                                ctx.temporal_position <= 0;
                            end else begin
                                ctx.temporal_position <= ((current_conv_batch << packs_per_batch_log2) - ((current_conv_batch >> temporal_split_log2) << ctx.dots_per_channel)) << ctx.stride_log2;
                            end
                        end else begin
                            ctx.current_channel <= 0;
                            if (current_conv_batch == 0 || 
                                (((current_conv_batch << packs_per_batch_log2) - ((current_conv_batch >> temporal_split_log2) << ctx.dots_per_channel)) << ctx.stride_log2) == ctx.temporal_length) begin
                                ctx.temporal_position <= 0;
                            end else begin
                                ctx.temporal_position <= ((current_conv_batch << packs_per_batch_log2) - ((current_conv_batch >> temporal_split_log2) << ctx.dots_per_channel)) << ctx.stride_log2;
                            end
                        end
                        ctx.current_group <= current_conv_batch >> temporal_split_log2;
                    end else begin
                        if (ctx.in_chan_log2 > 0) begin
                            ctx.current_channel <= current_conv_batch << temporal_split_log2;
                        end else begin
                            ctx.current_channel <= 0;
                        end
                        ctx.current_group <= current_conv_batch << temporal_split_log2;
                        ctx.temporal_position <= 0;
                    end
                    packing_mode <= 0; // Set packing mode to convolution
                    derive_conv_done <= 1;
                end

                CONV_MAP: begin
                    derive_conv_done <= 0;
                    conv_map_index <= 0;
                    previous_conv_map_index <= 0;
                    map_pulse <= 0;
                    kernel_position <= 0;
                    pack_position <= 0;
                    vector_a_map <= 32'hFFFFFFFF; // Default to invalid
                    vector_b_map <= 32'hFFFFFFFF; // Default to invalid
                    kernel_index <= ctx.temporal_position;
                    current_channel <= ctx.current_channel;
                    current_group <= ctx.current_group;
                    tap_offset <= -ctx.padding;
                    tensor_index <= 0;
                    weight_index <= 0;
                    conv_done <= 1;
                end

                CONV_STREAM: begin
                    conv_stream_done <= 0;
                    if (conv_map_index < 257) begin
                        map_pulse <= (conv_map_index != previous_conv_map_index);
                    end else begin
                        map_pulse <= 0;
                    end

                    previous_conv_map_index <= conv_map_index; // Ostensibly drives a pulse, but ends up remaining high for the whole stream.

                    // Boundary handling

                    if ((kernel_position) > ctx.kernel_size - 1) begin
                        tensor_index <= 32'hFFFFFFFF; // Map to zero
                        weight_index <= 32'hFFFFFFFF; // Map to zero
                    end else if ((((kernel_index + kernel_position + tap_offset) << ctx.in_chan_log2) + current_channel) < 0 || 
                        (((kernel_index + kernel_position + tap_offset) << ctx.in_chan_log2) + current_channel) > ctx.tensor_size - 1 || 
                        current_channel >= (1 << ctx.in_chan_log2) ||
                        current_group >= (1 << ctx.group_size)) begin
                        tensor_index <= 32'hFFFFFFFF;
                        weight_index <= 32'hFFFFFFFF;
                    end else if (data_context == 0) begin
                            // === Tensor is input/intermediate ===
                            tensor_index <= (((kernel_index + kernel_position + tap_offset) << ctx.in_chan_log2) + current_channel);
                            weight_index <= ctx.weight_pointer + (kernel_position << ctx.group_size) + current_group;
                    end else begin
                            // === Tensor is param_mem; weights are intermediate ===
                            weight_index <= ctx.weight_pointer + kernel_index + kernel_position;
                            tensor_index <= (kernel_position << ctx.group_size) + current_group;
                    end
                    vector_a_map <= tensor_index;
                    vector_b_map <= weight_index;

                    // Update counters
                    if (pack_position == ctx.pack_size - 1) begin
                        pack_position <= 0;
                        kernel_position <= 0;
                        kernel_index <= kernel_index + (1 << ctx.stride_log2);
                        if (kernel_index == ctx.temporal_length - (1 << ctx.stride_log2)) begin
                            kernel_index <= 0;
                            if (ctx.group_size > 0) begin
                                current_group <= current_group + 1;
                            end else begin
                                current_group <= 0;
                            end
                            if (ctx.in_chan_log2 > 0) begin
                                current_channel <= current_channel + 1;
                            end else begin
                                current_channel <= 0;
                            end
                        end
                    end else begin
                        pack_position <= pack_position + 1;
                        kernel_position <= kernel_position + 1;
                    end

                    conv_map_index <= conv_map_index + 1;

                    if (conv_map_index == 257) begin                                              
                        current_conv_batch <= current_conv_batch + 1;
                        conv_stream_done <= 1;
                        map_pulse <= 0;
                    end
                end

                DERIVE_POOL_PACK: begin
                    pool_done <= 0;
                    // We use this to determine the correct offsets to apply when packing pooling vectors for the dot product unit.
                    if (pooling_split_log2 > 8) begin
                        // Channels split across batches
                        ctx.pool_temporal_offset = current_pool_batch >> ctx.pool_batches_per_sweep_log2;
                        ctx.pool_channel_offset  = (current_pool_batch & pooling_mod) << ctx.channels_covered_per_sweep;
                    end else begin
                        // All channels covered in each batch; batches are temporal
                        ctx.pool_temporal_offset = current_pool_batch << ctx.sweeps_per_pool_batch_log2;
                        ctx.pool_channel_offset  = 0;
                    end
                    if (global_avg_pool || write_done)
                        derive_pool_done <= 1;
                end

                POOL_MAP: begin
                    derive_pool_done <= 0;
                    tap_level <= (global_avg_pool) ? time_steps : maxpool;
                    packing_mode <= 1;
                    pool_map_index <= 0;
                    previous_pool_map_index <= 0;
                    map_pulse <= 0;
                    pool_base_index <= (ctx.pool_temporal_offset << (maxpool + out_channels)) + ctx.pool_channel_offset;
                    j <= 0;
                    k <= 0;
                    pool_pack_index <= 0;
                    pool_done <= 1;
                end

                POOL_STREAM: begin
                    pool_stream_done <= 0;
                    if (pool_map_index < 257) begin
                        map_pulse <= (pool_map_index != previous_pool_map_index);
                    end else begin
                        map_pulse <= 0;
                    end
                    previous_pool_map_index <= pool_map_index;
                    pool_pack_index <= pool_base_index + (j & ((1 << out_channels) - 1)) + (k << out_channels) + (j >> out_channels);
                    if (pool_pack_index < (1 << (out_channels + time_steps)))
                        vector_c_map <= pool_pack_index;
                    else
                        vector_c_map <= 32'hFFFFFFFF;

                    if ((k + 1) == (1 << maxpool)) begin
                        k <= 0;
                        j <= j + 1;
                        if (j == (1 << out_channels) - 1)begin
                            j <= 0;
                            pool_base_index <= pool_base_index + (1 << (maxpool + out_channels));
                        end
                    end else k <= k + 1;
                    pool_map_index <= pool_map_index + 1;
                    if (pool_map_index == 257) begin
                        pool_stream_done <= 1;
                        map_pulse <= 0;
                        current_pool_batch <= current_pool_batch + 1;
                    end
                end

                DONE: begin
                    ctx.weight_pointer <= ctx.weight_pointer + ctx.pointer_advance; // Advance the weight pointer for the next instruction
                    instr_done <= 1;
                end
            endcase
        end
    end
endmodule