module clock_domain_cross#(
    parameter DATA_WIDTH = 16
)(
    input   logic                   adc_clk,
    input   logic [DATA_WIDTH-1:0]  adc_data,
    input   logic                   adc_valid,
    input   logic                   fpga_clk,

    output  logic [DATA_WIDTH-1:0]  adc_data_sync,
    output  logic                   adc_valid_sync
);

    // CDC for adc_valid_signal (single bit)

    // Register in ADC Clock Domain
    logic adc_valid_reg;
    always_ff @(posedge adc_clk) begin
        adc_valid_reg <= adc_valid;
    end

    // Double Flop Sycrhonizer to FPGA Clock Domain
    logic [1:0] valid_sync_chain;
    always_ff @(posedge fpga_clk) begin
        valid_sync_chain[0] <= adc_valid_reg;
        valid_sync_chain[1] <= valid_sync_chain[0];
    end

    // Edge detection to create single-cycle pulse in FPGA domain
    logic valid_prev;
    always_ff @(posedge fpga_clk) begin
        valid_prev <= valid_sync_chain[1];
    end

    assign adc_valid_sync = valid_sync_chain[1] & ~valid_prev;

    // CDC for adc_valid_signal (multi-bit)

    logic [DATA_WIDTH-1:0] adc_data_captured;
    always_ff @(posedge adc_clk) begin
        if (adc_valid) begin
            adc_data_captured <= adc_data;
        end
    end

    // Handshake synchronization for multi-bit data
    typedef enum logic [1:0] {
        IDE,
        REQUEST,
        ACKNOWLEDGE,
        COMPLETE
    } sync_state_t;

    sync_state_t state_adc, state_fpga;
    logic request_pulse, ack_pulse;
    logic [DATA_WIDTH-1:0] data_buffer;

    // ADC-side state machine
    always_ff @(posedge adc_clk) begin
        case (state_adc)
            IDLE: begin
                if (adc_valid) begin
                    data_buffer <= adc_data_captured;
                    state_adc <= REQUEST;
                end
            end

            REQUEST: begin
                if (ack_pulse) begin
                    state_adc <= COMPLETE;
                end
            end

            COMPLETE: begin
                state_adc <= IDLE;
            end

            default: state_adc <= IDLE; 
        endcase
    end

    // Generate request pulse to FPGA domain
    logic request_reg;
    always_ff @(posedge adc_clk) begin
        request_reg <= (state_adc == REQUEST);
    end

    // Synchronize request to FPGA domain
    logic [2:0] request_sync;
    always_ff @(posedge fpga_clk) begin
        request_sync[0] <= request_reg;
        request_sync[1] <= request_sync[0];
        request_sync[2] <= request_sync[1];
    end

    assign request_pulse = request_sync[1] & ~request_sync[2];

    // FPGA-side state machine
    always @(posedge fpga_clk) begin
        case (state_fpga)
            IDLE: begin
                if (request_pulse) begin
                    adc_data_sync <= data_buffer;
                    state_fpga <= ACKNOWLEDGE;
                end
            end

            ACKNOWLEDGE: begin
                state_fpga <= COMPLETE;
            end

            COMPLETE: begin
                state_fpga <= IDLE;
            end

            default: state_fpga <= IDLE;
        endcase
    end

    // Generate acknowledge pulse back to ADC domain
    logic ack_reg;
    always_ff @(posedge fpga_clk) begin
        ack_reg <= (state_fpga == ACKNOWLEDGE);
    end

    // Sycnrhonize acknowledge to ADC domain
    logic [2:0] ack_sync;
    always_ff @(posedge adc_clk) begin
        ack_sync[0] <= ack_reg;
        ack_sync[1] <= ack_sync[0];
        ack_sync[2] <= ack_sync[1];
    end

    assign ack_pulse = ack_sync[1] & ~ack_sync[2];
endmodule