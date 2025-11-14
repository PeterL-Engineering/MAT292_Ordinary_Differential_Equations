import numpy as np

def fir_filter(coeffs, data_in):
    x_arr = np.zeros(len(coeffs)) 
    y = []  # initialize output array

    for i in range(len(data_in) - 1):  
        circ_ptr = i % len(coeffs)  # pointer for circular buffer
        x_arr[circ_ptr] = data_in[i]
        y.append(np.sum(np.multiply(x_arr, coeffs)))

    return y


def hex_to_decimal(hex_array):
    dec_array = [int(h, 16) for h in hex_array]
    return dec_array

        


if __name__ == "__main__":
    hex_coeff = ["0xFFCE", "0xFF4A", "0xFE3A", "0xFCA6", "0xFA9C", "0xF82B", "0xF567", "0xF267", "0xEF45", "0xEC1D", "0xE90F", "0xE63D", "0xE3C9", "0xE1D5", "0xE083", "0xE0F8", "0xE355", "0xE6B7", "0xEB1A", "0xF077", "0xF6C3", "0xFDF0", "0x05EB", "0x0E9D", "0x17EC", "0x21BA", "0x2BE6", "0x364D", "0x40C7", "0x4B2D", "0x5557", "0x5F1F"]
    coeffs = hex_to_decimal(hex_coeff)

    filtered_data = fir_filter(coeffs, data_in)
