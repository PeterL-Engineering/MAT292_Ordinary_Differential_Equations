module uart_transmitter #(
    parameter CLK_FREQ = 100_000_000,
    parameter BAUD_RATE = 115200,
    parameter DATA_WIDTH = 32
)(
    input   logic                   clk,        
    input   logic                   reset_n,
    input   logic [DATA_WIDTH-1:0]  data,
    input   logic                   data_valid, // Data ready to transmit
    output  logic                   uart_tx,    // UART transmit line
    output  logic                   tx_busy,    // Transmitter busy flag
);

    localparam BAUD_COUNT = CLK_FREQ / BAUD_RATE;
    logic [$clog2(BAUD_COUNT)-1:0] baud_counter;
    logic baud_tick;

    always_ff @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            baud_counter <= 0;
            baud_tick <= 0;
        end else begin
            if (baud_counter == BAUD_COUNT - 1) begin
                baud_counter <= 0;
                baud_tick <= 1;
            end else begin
                baud_counter <= baud_counter + 1;
                baud_tick <= 0;
            end
        end
    end

    typedef enum logic [3:0] {
        IDLE,
        START_BIT,
        DATA_BIT_0,
        DATA_BIT_1,
        DATA_BIT_2,
        DATA_BIT_3,
        DATA_BIT_4,
        DATA_BIT_5,
        DATA_BIT_6,
        DATA_BIT_7,
        STOP_BIT,
        BYTE_DONE
    } uart_state_t;

    uart_state_t state, next_state;

    // Data shift register and bit counter
    logic [7:0] tx_shift_reg;
    logic [2:0] bit_counter;
    logic [DATA_WIDTH-1:0] data_buffer;
    logic [5:0] byte_counter;

    always_ff @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            state <= IDLE;
            uart_tx <= 1'b1;
            tx_shift_reg <= 8'h00;
            bit_counter <= 0;
            byte_counter <= 0;
            data_buffer <= 0;
            tx_busy <= 0;
        end else begin
            state <= next state;

            case (state)
                IDLE: begin
                    uart_tx <= 1'b1;
                    tx_busy = 0;
                    if (data_valid) begin
                        data_buffer <= data;
                        byte_counter <= (DATA_WIDTH / 8) - 1;
                        tx_busy <= 1;
                    end
                end

                START_BIT: begin
                    if (baud_tick) begin
                        uart_tx <= 1'b0;
                    end
                end

                DATA_BIT_0, DATA_BIT_1, DATA_BIT_2, DATA_BIT_3, 
                DATA_BIT_4, DATA_BIT_5, DATA_BIT_6, DATA_BIT_7: begin
                    if (baud_tick) begin
                        uart_tx <= tx_shift_reg[bit_counter];
                        bit_counter <= bit_counter + 1;
                    end
                end

                STOP_BIT: begin
                    if (baud_tick) begin
                        uart_tx <= 1'b1;
                    end
                end
                
                BYTE_DONE: begin
                    if (byte_counter == 0) begin
                        // All bytes transmitted
                        tx_busy <= 0; 
                    end else begin
                        // Load next byte
                        byte_counter <= byte_counter - 1;
                        data_buffer <= data_buffer >> 8;
                    end
                end
            endcase
        end
    end

    // Next state logic
    always_comb begin
        next_state = state;

        case(state)
            IDLE: begin
                if (data_valid) begin
                    next_state = START_BIT;
                end
            end

            START_BIT: begin
                if (baud_tick) begin
                    // Load first byte of data
                    tx_shift_reg = data_buffer[7:0];
                    next_state = DATA_BIT_0;
                end
            end

            DATA_BIT_0: if (baud_tick) next_state = DATA_BIT_1;
            DATA_BIT_1: if (baud_tick) next_state = DATA_BIT_2;
            DATA_BIT_2: if (baud_tick) next_state = DATA_BIT_3;
            DATA_BIT_3: if (baud_tick) next_state = DATA_BIT_4;
            DATA_BIT_4: if (baud_tick) next_state = DATA_BIT_5;
            DATA_BIT_5: if (baud_tick) next_state = DATA_BIT_6;
            DATA_BIT_6: if (baud_tick) next_state = DATA_BIT_7;
            DATA_BIT_7: if (baud_tick) next_state = STOP_BIT;
        
            STOP_BIT: begin
                if (baud_tick) begin
                    next_state = BYTE_DONE;
                end
            end

            BYTE_DONE: begin
                if (byte_counter == 0) begin
                    next_state = IDLE;
                end else begin
                    next_state = START_BIT;
                end
            end
        endcase
    end
    
endmodule