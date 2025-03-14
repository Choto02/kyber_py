# Define constants
Q = 3329

# List of MULT_INV_3329 values with corresponding shift values
# mult_inv_values = [
#     (157, 9),
#     (315, 10),
#     (630, 11),
#     (1260, 12),
#     (2520, 13)
# ]

mult_inv_values = [  #CHANGE THESE to 156, 314 etc without rounding them from their binary value
    (157, 9),
    (315, 10),
    (630, 11),
    (1260, 12),
    (2520, 13)
]

# Test input values for x
#test_values_x = [0, 1, 2, 10, 76, 500, 1664, 2500, 3328, 3329]
# Expand test values to include all integers from 0 to 3329
test_values_x = list(range(3330))

# Function to compute error between bitwise and float outputs
def compute_error(bitwise_out, float_out):
    return abs(bitwise_out - float_out)

def float_compress(x):
    return round((1024 * x / 3329)) % 1024

# Function to implement the Verilog logic using bitwise operations
def compress(x, mult_inv_3329, shift_value):
    # Multiply by MULT_INV_3329 (fixed-point multiplication using bitwise shift)
    product = x * mult_inv_3329

    # Shift right by corresponding shift value to adjust for the fixed-point scale
    result = product >> shift_value  

    # Check bit (shift_value - 1) of the original product (before shift) to round
    rounded = result + 1 if (product & (1 << (shift_value - 1))) != 0 else result

    # Take mod 1024 (Extract the lower 10 bits using bitwise AND)
    out = rounded & 0x3FF  # 0x3FF is 1023 (binary 1111111111) to get last 10 bits

    return out

# Store results
results = []
accuracy_summary = {}
for mult_inv_3329, shift_value in mult_inv_values:
    total_error = 0

    for x in test_values_x:
        bitwise_result = compress(x, mult_inv_3329, shift_value)
        float_result = float_compress(x)
        error = compute_error(bitwise_result, float_result)

        total_error += error

        results.append({
            "MULT_INV_3329": mult_inv_3329,
            "Shift": shift_value,
            "x": x,
            "Bitwise Out": bitwise_result,
            "Float Out": float_result,
            "Error": error
        })

# Store total error for this MULT_INV_3329
    accuracy_summary[mult_inv_3329] = total_error

# Find the most accurate MULT_INV_3329 (one with least total error)
best_mult_inv = min(accuracy_summary, key=accuracy_summary.get)

# Print results
print(f"{'MULT_INV_3329':<15} {'Shift':<6} {'x':<5} {'Bitwise Out':<12} {'Float Out':<10} {'Error':<6}")
print("-" * 70)

for result in results[:100]:  # Print only the first 100 rows for readability
    print(f"{result['MULT_INV_3329']:<15} {result['Shift']:<6} {result['x']:<5} {result['Bitwise Out']:<12} {result['Float Out']:<10} {result['Error']:<6}")

# Sort accuracy summary to find the top 3 most accurate MULT_INV_3329 values
top_accurate = sorted(accuracy_summary.items(), key=lambda x: x[1])[:3]

# Print the top 3 most accurate MULT_INV_3329 values
print("\nTop 3 Most Accurate MULT_INV_3329 Values:")
for rank, (mult_inv, error) in enumerate(top_accurate, start=1):
    print(f"{rank}. MULT_INV_3329: {mult_inv}, Total Error: {error}")
