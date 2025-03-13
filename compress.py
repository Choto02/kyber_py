import numpy as np
import os

# Define constants
Q = 3329
powers_of_2 = [2**i for i in range(1, 41)]  # 2^1 to 2^40
test_values_x = [0, 1, 2, 10, 76, 500, 1664, 2500, 3328, 3329]
d_values = range(1, 13)  # d from 1 to 12
# Define MULT_INV_3329 values as approximations of (1 / 3329) * 2^bitwidth
mult_inv_values = [(2**bitwidth) // Q for bitwidth in range(1, 41)]  # 2^bitwidth / 3329


print(mult_inv_values)


# def digital_operation(d, bitwidth, mult_inv_3329, x):
#     shifted_input = 2**d  # Equivalent to shifting by d bits
#     product = shifted_input * mult_inv_3329  # Scaling by MULT_INV_3329
#     result = product * x  # Final multiplication by x
#     shift_back = bitwidth  # Dynamically shift back based on bitwidth used
#     final = result >> shift_back  # Restore original scale by shifting back
#     return final & ((1 << d) - 1)  # Extract LSBs of size d


def digital_operation(d, bitwidth, mult_inv_3329, x):
    shifted_input = 2**d  # Equivalent to shifting by d bits
    product = shifted_input * mult_inv_3329  # Scaling by MULT_INV_3329
    result = product * x  # Final multiplication by x
    shift_back = bitwidth  # Dynamically shift back based on bitwidth used
    
    # Apply rounding before shifting
    final = (result + (1 << (shift_back - 1))) >> shift_back  

    return final & ((1 << d) - 1)  # Extract LSBs of size d


# Function to perform software multiplication using floating-point
def float_operation(d, x):
    shifted_input = 2**d  # Shift operation in float
    result = (shifted_input / Q) * x  # Regular multiplication without inv(Q)
    rounded_result = round(result)
    modded_result = rounded_result % shifted_input
    return modded_result


# Prepare results storage
results = []


# Iterate through values of d and MULT_INV_3329 with dynamic shift back
for d in d_values:
    for bitwidth in range(1, 41):  # Iterate over bitwidths
        mult_inv_3329 = (2**bitwidth) // Q  # Compute MULT_INV_3329 dynamically
        for x in test_values_x:
            digital_result = digital_operation(d, bitwidth, mult_inv_3329, x)
            float_result = float_operation(d, x)
            error = abs(digital_result - float_result)

            # Store results
            results.append({
                "d": d,
                "Bitwidth": bitwidth,
                "MULT_INV_3329": mult_inv_3329,
                "x": x,
                "Digital Result": digital_result,
                "Float Result": float_result,
                "Error": error
            })


# Define the output file name
output_file = "bitwidth_20_accuracy_analysis.txt"


# Save only results where Bitwidth = 20
with open(output_file, "w") as f:
    f.write(f"{'d':<5} {'Bitwidth':<10} {'MULT_INV_3329':<15} {'x':<5} {'Digital Result':<15} {'Float Result':<15} {'Error':<10}\n")
    f.write("-" * 80 + "\n")

    for result in results:
        if result["Bitwidth"] in (20, 32):
            f.write(f"{result['d']:<5} {result['Bitwidth']:<10} {result['MULT_INV_3329']:<15} {result['x']:<5} {result['Digital Result']:<15} {result['Float Result']:<15} {result['Error']:<10}\n")


print(f"Results for Bitwidth = 20 saved to {output_file}")
sum_error_20 = sum(result["Error"] for result in results if result["Bitwidth"] == 20)
sum_error_32 = sum(result["Error"] for result in results if result["Bitwidth"] == 32)


print(sum_error_20, sum_error_32)

# Define the binary string to extract bit-length variations
binary_str = "00100111010111110111"
bit_length = 5
binary_multiplier = int(binary_str[:bit_length], 2) 
print(binary_str)
print(bin(binary_multiplier))

for bit_length in range(0,5):
    print("Bit [", bit_length, "]", bin(binary_multiplier)[bit_length])
if bin(binary_multiplier)[bit_length-1] == 1:
            print("yes")
