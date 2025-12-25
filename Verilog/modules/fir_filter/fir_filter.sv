module fir_filter #(
    parameter COEFF_WIDTH = 16,  // Width of filter coefficients
    parameter DATA_WIDTH = 16,   // Width of input/output data  
    parameter TAPS = 32          // Number of filter taps (filter order + 1)
)(
    input  logic                         clk,
    input  logic                         reset,
    input  logic signed [DATA_WIDTH-1:0] data_in,        // Input sample
    input  logic                         data_valid,     // When high, new input data is valid
    output logic        [DATA_WIDTH-1:0] data_out,       // Filtered output
    output logic                         data_out_valid  // When high, output data is valid
);

    // Coefficient ROM - stores filter coefficients that define frequency response
    logic signed [COEFF_WIDTH-1:0] coeffs [TAPS-1:0];
    initial $readmemh("fir_coeffs.hex", coeffs);  // Load coefficients from file

    // Delay line - circular buffer storing last TAPS input samples
    logic signed [DATA_WIDTH-1:0] delay_line [TAPS-1:0];
    logic [4:0] write_ptr;  // Pointer for circular buffer (0-31)

    // Multiply-accumulate register
    logic signed [COEFF_WIDTH+DATA_WIDTH-1:0] acc;

    always_ff @(posedge clk) begin
        if (reset) begin
            // Initialize everything on reset
            for (int i = 0; i < TAPS; i++) delay_line[i] <= '0;
            write_ptr <= '0;
            data_out_valid <= '0;
        end else if (data_valid) begin
            // Store new sample in delay line
            delay_line[write_ptr] <= data_in;
            write_ptr <= write_ptr + 1;

            // FIR computation: sum of (coefficient * delayed sample)
            acc <= 0;
            for (int i = 0; i < TAPS; i++) begin
                automatic int idx = (write_ptr - i) % TAPS;  // Circular indexing
                acc <= acc + delay_line[idx] * coeffs[i];    // Multiply-accumulate
            end

            // Scale down result to original data width
            data_out <= acc[COEFF_WIDTH+DATA_WIDTH-2:DATA_WIDTH-1];
            data_out_valid <= 1;
        end else begin
            data_out_valid <= 0;
        end
    end
endmodule