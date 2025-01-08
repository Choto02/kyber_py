import os
from hashlib import sha3_256, sha3_512, shake_128, shake_256
import random
import sys

# Kyber-512 Parameters
KYBER_N = 256  # Polynomial degree
KYBER_Q = 3329  # Modulus
KYBER_K = 2  # Number of polynomials in the vector (k = 2 for Kyber-512)
ETA = 2  # CBD parameter for Kyber-512

def pke_keygen():
    d = os.urandom(32)
    rho, sigma = _G(d + bytes([KYBER_K]))

    A_hat = generate_matrix(rho, True)

    # Set counter for PRF
    N = 0

    # Generate the error vector s ∈ R^k
    s, N = _generate_error_vector(sigma, ETA, N)

    # Generate the error vector e ∈ R^k
    e, N = _generate_error_vector(sigma, ETA, N)

    s_hat = s.to_ntt()
    e_hat = e.to_ntt()

    # # Debugging
    # print(f"Secret vector length: {len(s)} (expected: {KYBER_K})")
    # print(f"First polynomial in s has length: {len(s[0])} (expected: {KYBER_N})")

    # # Step 4: Compute t = A * s + e
    # t = []
    # for i in range(KYBER_K):
    #     t_i = [0] * KYBER_N
    #     for j in range(KYBER_K):
    #         a_s = [(A[i][j][k] * s[j][k]) % KYBER_Q for k in range(KYBER_N)]
    #         t_i = [(t_i[k] + a_s[k]) % KYBER_Q for k in range(KYBER_N)]
    #     t_i = [(t_i[k] + e[i][k]) % KYBER_Q for k in range(KYBER_N)]
    #     t.append(t_i)

    # # Public key includes t and rho
    # public_key = (t, rho)

    # # Private key is the secret vector s
    # private_key = s

    #return public_key, private_key
    return N

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



def shake128(input_bytes, output_length):
    shake = shake_128()
    shake.update(input_bytes)
    return shake.digest(output_length)


### Example Usage ###

if __name__ == "__main__":
    Y = pke_keygen()
    
