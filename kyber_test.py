import os
from hashlib import sha3_256, sha3_512, shake_128, shake_256
import random
import sys

# Kyber-512 Parameters
KYBER_N = 256  # Polynomial degree
KYBER_Q = 3329  # Modulus
KYBER_K = 2  # Number of polynomials in the vector (k = 2 for Kyber-512)
ETA = 2  # CBD parameter for Kyber-512
root_of_unity = 17

        

def pke_keygen():
    
    d = os.urandom(32)
    rho, sigma = _G(d + bytes([KYBER_K]))

    A_hat = generate_matrix(rho, True)

    

    # Set counter for PRF
    N = 0

    # Generate the error vector s ∈ R^k
    s, N = _generate_error_vector(sigma, ETA, N)
    #print(s)

    # Generate the error vector e ∈ R^k
    e, N = _generate_error_vector(sigma, ETA, N)

    ntt_zetas = [
            pow(root_of_unity, _br(i, 7), 3329) for i in range(128)
        ]

    s_hat = vector_to_ntt(s, ntt_zetas)
    e_hat = vector_to_ntt(e, ntt_zetas)

    print(len(A_hat[0]))
    print(len(s_hat))

    As_hat = matrix_multiply(A_hat, s_hat)
    t_hat = matrix_addition(As_hat, e_hat)
    
    # Byte encode
    ek_pke = t_hat.encode(12) + rho
    dk_pke = s_hat.encode(12)

    return (ek_pke, dk_pke)


def _G(s):
    """
    Hash function described in 4.5 of FIPS 203 (page 18)
    """
    h = sha3_512(s).digest()
    return h[:32], h[32:]

def generate_matrix(seed, transpose, dim=KYBER_K):
    A = [[None for _ in range(dim)] for _ in range(dim)]
    for i in range(dim):
        for j in range(dim):
            shake_input = seed + bytes([j]) + bytes([i])  # Combine seed with matrix indices
            poly_bytes = shake128(shake_input, 840)  # Generate 2n bytes
            A[i][j] = ntt_sample(poly_bytes)  # Convert to polynomial
    if transpose is True:  
        rows, cols = len(A), len(A[0])
        transposed_A = [[A[j][i] for j in range(rows)] for i in range(cols)]
        A = transposed_A
    return A

def ntt_sample(input_bytes):
    i, j = 0, 0
    coefficients = [0 for _ in range(KYBER_N)]
    while j < KYBER_N:
        d1 = input_bytes[i] + 256 * (input_bytes[i + 1] % 16)
        d2 = (input_bytes[i + 1] // 16) + 16 * input_bytes[i + 2]

        if d1 < 3329:
            coefficients[j] = d1
            j = j + 1

        if d2 < 3329 and j < KYBER_N:
            coefficients[j] = d2
            j = j + 1

        i = i + 3
    return coefficients

def _generate_error_vector(sigma, eta, N):
    """
    Helper function which generates a element in the
    module from the Centered Binomial Distribution.
    """
    elements = [0 for _ in range(KYBER_K)]
    for i in range(KYBER_K):
        prf_output = _prf(eta, sigma, bytes([N]))
        elements[i] = cbd(prf_output, eta)
        N += 1
    #v = self.M.vector(elements)

    return elements, N

def _prf(eta, s, b):
    """
    Pseudorandom function described in 4.3 of FIPS 203 (page 18)
    """
    input_bytes = s + b
    if len(input_bytes) != 33:
        raise ValueError(
            "Input bytes should be one 32 byte array and one single byte."
        )
    return shake_256(input_bytes).digest(eta * 64)

def cbd(input_bytes, eta, is_ntt=False):
    """
    Algorithm 2 (Centered Binomial Distribution)
    https://pq-crystals.org/kyber/data/kyber-specification-round3-20210804.pdf

    Algorithm 6 (Sample Poly CBD)

    Expects a byte array of length (eta * deg / 4)
    For Kyber, this is 64 eta.
    """
    assert 64 * eta == len(input_bytes)
    coefficients = [0 for _ in range(256)]
    b_int = int.from_bytes(input_bytes, "little")
    mask = (1 << eta) - 1
    mask2 = (1 << 2 * eta) - 1
    for i in range(256):
        x = b_int & mask2
        a = bit_count(x & mask)
        b = bit_count((x >> eta) & mask)
        b_int >>= 2 * eta
        coefficients[i] = (a - b) % 3329
    return coefficients


def bit_count(x: int) -> int:
    """
    Count the number of bits in x
    """
    return x.bit_count()

def matrix_to_ntt(matrix, ntt_zeta):
    """
    Convert every element of the matrix into NTT form
    """
    print(matrix)
    data = [[to_ntt(x, ntt_zeta) for x in row] for row in matrix]
    
    return data

def vector_to_ntt(matrix, ntt_zeta):
    """
    Convert every element of the matrix into NTT form
    """
    print(matrix)
    data = [to_ntt(x, ntt_zeta) for x in matrix]
    
    return data

def to_ntt(input_coeffs, input_ntt_zetas):
    """
    Convert a polynomial to number-theoretic transform (NTT) form.
    The input is in standard order, the output is in bit-reversed order.
    """
    k, l = 1, 128
    coeffs = input_coeffs
    zetas = input_ntt_zetas
    
    while l >= 2:
        start = 0
        while start < 256:
            zeta = zetas[k]
            k = k + 1
            for j in range(start, start + l):
                t = zeta * coeffs[j + l]
                coeffs[j + l] = coeffs[j] - t
                coeffs[j] = coeffs[j] + t
            start = l + (j + 1)
        l = l >> 1

    for j in range(256):
        coeffs[j] = coeffs[j] % 3329

    return coeffs

def _br(i, k):
        """
        bit reversal of an unsigned k-bit integer
        """
        bin_i = bin(i & (2**k - 1))[2:].zfill(k)
        return int(bin_i[::-1], 2)

def matrix_multiply(A, B):
    """
    Multiplies two matrices A and B.
    Args:
        A: Matrix A (list of lists), dimensions m x n.
        B: Matrix B (list of lists), dimensions n x p.
    Returns:
        result: Resulting matrix (list of lists), dimensions m x p.
    Raises:
        ValueError: If matrix dimensions are incompatible for multiplication.
    """
    # Validate dimensions
    if len(A[0]) != len(B):
        raise ValueError("Matrix dimensions do not match for multiplication. A's columns must equal B's rows.")
    
    # Initialize the result matrix with zeros
    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
    
    # Perform multiplication
    for i in range(len(A)):  # Iterate over rows of A
        for j in range(len(B[0])):  # Iterate over columns of B
            for k in range(len(B)):  # Iterate over rows of B / columns of A
                result[i][j] += A[i][k] * B[k][j]
    
    return result

def matrix_addition(A, B):
    """
    Adds two matrices A and B element-wise.
    Args:
        A: Matrix A (list of lists), dimensions m x n.
        B: Matrix B (list of lists), dimensions m x n.
    Returns:
        result: Resulting matrix (list of lists), dimensions m x n.
    Raises:
        ValueError: If matrix dimensions do not match.
    """
    # Validate dimensions
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        raise ValueError("Matrix dimensions do not match for addition.")
    
    # Perform element-wise addition
    result = [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]
    return result

def shake128(input_bytes, output_length):
    shake = shake_128()
    shake.update(input_bytes)
    return shake.digest(output_length)



### Example Usage ###

if __name__ == "__main__":
    ek, dk = pke_keygen()
    
