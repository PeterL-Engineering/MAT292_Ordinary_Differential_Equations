module spike_detector #(
    parameter DATA_WIDTH = 16
)(
    input  logic                         clk,
    input  logic                         reset,
    input  logic signed [DATA_WIDTH-1:0] signal_in,
    input  logic signed [DATA_WIDTH-1:0] threshold_in,
    input  logic                         data_valid,
    output logic                         spike_detected,
    output logic        [15:0]           spike_timestamp,
    output logic signed [DATA_WIDTH-1:0] spike_amplitude
);

    logic signed [DATA_WIDTH-1:0] prev_signal;
    logic spike_occured;
    logic [15:0] sample_counter;

    always_ff @(posedge clk) begin
        if (reset) begin
            prev_signal <= 0;
            spike_detected <= 0;
            sample_counter <= 0;
            spike_amplitude <= 0;
        end else if (data_valid) begin
            prev_signal <= signal_in;
            sample_counter <= sample_counter + 1;

            // Detect positive-going threshold crossing
            spike_occured = (signal_in > threshold) && (prev_signal <= threshold_in);
            spike_detected <= spike_occured;

            if (spike_occured) begin
                spike_timestamp <= sample_counter;
                spike_amplitude <= signal_in;
            end

        end else begin
            spike_detected <= 0;
        end
    end
endmodule