# Define the binary string to extract bit-length variations
binary_str = "00100111010111110111"

# Define values of x and convert them to binary
x_values = [0, 1, 2, 10, 76, 500, 1664, 2500, 3328, 3329]

# Function to perform binary multiplication and compare with normal operation
results = []

for bit_length in range(5, 16):  # From 5 to 15 bits
    binary_multiplier = int(binary_str[:bit_length], 2)  # Convert extracted bits to integer
    print("Binary Multiplier: ")
    for x in x_values:
        x_binary = int(bin(x)[2:])  # Convert x to binary integer
        print("X: ",bin(x))
        print("Binary Multiplier: ",bin(binary_multiplier))
        binary_result = binary_multiplier * x  # Perform binary multiplication
        print("Binary Result: ",bin(binary_result))
        prerounded_result = binary_result *1024
        print("Prerounded result: ",bin(prerounded_result))
        binary_str = bin(prerounded_result)[2:]

        if bin(binary_multiplier)[bit_length-1] == 1:
            prerounded_result += 1
        shifted_result = prerounded_result >> bit_length  # Left shift the result


        # Normal operation: (1024/3329) * x
        normal_result = round((1024 / 3329) * x) % 1024
        difference = abs(shifted_result - normal_result)  # Compute the error



        # Store results
        results.append({
            "Bit Length": bit_length,
            "Binary Multiplier": binary_multiplier,
            "X": x,
            "Binary Result": binary_result,
            "Shifted Result": shifted_result,
            "Normal Result": normal_result,
            "Difference": difference
        })

# Define the output file name (Change path as needed)
output_file = "binary_multiplication_comparison.txt"

# Save results to a text file
with open(output_file, "w") as f:
    f.write(f"{'Bit Length':<12} {'Binary Multiplier':<18} {'X':<5} {'Binary Result':<15} {'Shifted Result':<20} {'Normal Result':<15} {'Difference':<15}\n")
    f.write("-" * 100 + "\n")

    for result in results:
        f.write(f"{result['Bit Length']:<12} {result['Binary Multiplier']:<18} {result['X']:<5} {result['Binary Result']:<15} {result['Shifted Result']:<20} {result['Normal Result']:<15.6f} {result['Difference']:<15.6f}\n")

print(f"Results saved to {output_file}")
