module analog_control #(
    parameter DATA_WIDTH = 16,
    parameter HYSTERESIS = 100,
    parameter CALIBRATION_CYCLES = 1000
)(
    input   logic                         clk,       
    input   logic                         reset_n,
    input   logic signed [DATA_WIDTH-1:0] signal_level,
    input   logic                         signal_valid,

    // Control outputs to analog circuitry
    output  logic                         gain_control,  // 0 = low gain, 1 = high gain
    output  logic                         filter_select, // Filter bandwidth select
    output  logic                         calibration_en // Calibration enable
);

    logic signed [DATA_WIDTH-1:0] signal_abs;
    logic signed [DATA_WIDTH+7:0] signal_avg;
    logic [9:0] avg_counter;

    // Track signal envelope (absolute value)
    assign signal_abs = (signal_level[DATA_WIDTH-1]) ? -signal_level : signal_level;

    // Moving average of signal amplitude
    always_ff @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            signal_avg <= 0;
            avg_counter <= 0;
        end else if (signal_valid) begin
            if (avg_counter < 10'd100) begin
                signal_avg <= signal_avg + signal_abs;
                avg_counter <= avg_counter + 1;
            end else begin
                // Update average every 100 samples
                signal_avg <= (signal_avg * 15 + signal_abs) >> 4;
            end
        end
    end

    // Gain control logic 

    localparam THRESH_HIGH = 16'd30000; // 91% of full scale
    localparam THRESH_LOW = 16'd5000;   // 15% of full scale

    logic [1:0] gain_state;

    always_ff @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            gain_control <= 1'b0;
            gain_state <= 2'b00;
        end else if (signal_valid) begin
            // State machine for gain control with hysteresis
            case (gain_state)
                2'b00: begin  // Low gain, check for saturation
                    if (signal_abs > (THRESH_HIGH - HYSTERESIS)) begin
                        // Signal too large, need to reduce gain
                        gain_control <= 1'b1;  // Switch to high gain
                        gain_state <= 2'b10;
                    end else if (signal_avg < (THRESH_LOW + HYSTERESIS)) begin
                        // Signal consistently small, could increase gain
                        gain_state <= 2'b01;  // Consider switching to high gain
                    end
                end
                
                2'b01: begin  // Considering high gain
                    if (signal_avg < THRESH_LOW) begin
                        // Signal confirmed small, switch to high gain
                        gain_control <= 1'b1;
                        gain_state <= 2'b11;
                    end else begin
                        // Signal increased, stay with low gain
                        gain_state <= 2'b00;
                    end
                end
                
                2'b10: begin  // High gain, just switched
                    gain_state <= 2'b11;  // Move to monitoring
                end
                
                2'b11: begin  // High gain, monitor
                    if (signal_abs > THRESH_HIGH) begin
                        // Saturated even at high gain
                        gain_control <= 1'b0;  // Switch back to low gain
                        gain_state <= 2'b00;
                    end
                end
            endcase
        end
    end

    // Filter Selection Logic
    
    logic [23:0] sample_counter;
    logic [1:0] filter_state;
    
    // Simple filter selection based on noise conditions
    always_ff @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            filter_select <= 2'b00;  // Default filter setting
            filter_state <= 2'b00;
            sample_counter <= 0;
        end else if (signal_valid) begin
            sample_counter <= sample_counter + 1;
            
            // Change filter every 10k samples to find optimal setting
            if (sample_counter == 24'd10000) begin
                sample_counter <= 0;
                
                case (filter_state)
                    2'b00: begin  // Wide bandwidth
                        filter_select <= 2'b00;
                        filter_state <= 2'b01;
                    end
                    2'b01: begin  // Medium bandwidth
                        filter_select <= 2'b01;
                        filter_state <= 2'b10;
                    end
                    2'b10: begin  // Narrow bandwidth
                        filter_select <= 2'b10;
                        filter_state <= 2'b11;
                    end
                    2'b11: begin  // Very narrow
                        filter_select <= 2'b11;
                        filter_state <= 2'b00;
                    end
                endcase
            end
        end
    end
    
    // Calibration Control
    
    logic [23:0] cal_counter;
    logic cal_active;
    
    always_ff @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            calibration_en <= 1'b0;
            cal_counter <= 0;
            cal_active <= 1'b0;
        end else begin
            // Enable calibration every 1 million cycles (~10ms at 100MHz)
            if (cal_counter == 24'd1000000) begin
                cal_counter <= 0;
                calibration_en <= 1'b1;
                cal_active <= 1'b1;
            end else begin
                cal_counter <= cal_counter + 1;
                
                // Keep calibration enabled for CALIBRATION_CYCLES
                if (cal_active) begin
                    if (cal_counter == CALIBRATION_CYCLES) begin
                        calibration_en <= 1'b0;
                        cal_active <= 1'b0;
                    end
                end
            end
        end
    end
    
endmodule
