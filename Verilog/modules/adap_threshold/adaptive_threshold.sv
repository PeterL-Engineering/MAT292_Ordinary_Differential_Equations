module adavaptive_threshold #(
    parameter DATA_WIDTH = 16,
    parameter NOISE_WINDOW = 64
)(
    input  logic                            clk,
    input  logic                            reset,
    input  logic signed [DATA_WIDTH-1:0]    filtered_signal,
    input  logic                            data_valid,
    output logic signed [DATA_WIDTH-1:0]    threshold,
    output logic                            threshold_valid
);

    logic signed [DATA_WIDTH-1:0] noise_estimate;
    logic signed [DATA_WIDTH+6:0] noise_accumulator;
    logic [5:0] sample_count;

    always_ff @(posedge clk) begin
        if (reset) begin
            noise_accumulator <= 0;
            sample_count <= 0;
            threshold <= 0;
            threshold_valid <= 0;
        end else if (data_valid) begin
            // Update noise estimate using moving average of signal envelope
            noise_accumulator <= noise_accumulator - noise[NOISE_WINDOW-1:0] +
                                (filtered_signal > 0 ? filtered_signal : -filtered_signal);
            
            sample_count <= sample_count + 1;

            // Set threshold to 4x noise floor
            threshold <= (noise_accumulator[NOISE_WINDOW-1:0] << 2);
            threshold_valid <= 1;
        end else begin
            threshold_valid <= 0;
        end
    end
endmodule