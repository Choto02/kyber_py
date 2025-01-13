from hashlib import sha3_256, sha3_512, shake_128, shake_256

# Kyber-512 Parameters
KYBER_N = 256  # Polynomial degree
KYBER_Q = 3329  # Modulus
KYBER_K = 2  # Number of polynomials in the vector (k = 2 for Kyber-512)
ETA = 2  # CBD parameter for Kyber-512

def BitsToBytes(b):
    """
    Converts a bit array (of a length that is a multiple of eight) into an array of bytes. Algorithm 3
    """
    # Ensure the length of the input bit array is a multiple of 8
    if len(b) % 8 != 0:
        raise ValueError("The length of the bit array must be a multiple of 8.")
    
    l = len(b) // 8  # Determine the length of the byte array
    B = [0] * l      # Initialize byte array with zeros

    # Iterate through the bit array
    for i in range(len(b)):
        B[i // 8] += b[i] * (2 ** (i % 8))  # Update the corresponding byte value
    
    return B

def BytesToBits(B):
    """
    Converts a byte array (list of integers) into a bit array.
    """
    # Initialize an empty bit array
    b = []
    
    # Iterate over each byte in the input array
    for i in range(len(B)):
        C_i = B[i]  # Copy the current byte
        for j in range(8):  
            b.append(C_i % 2)  
            C_i //= 2  
    
    return b

def Find_d (n):
    if n <= 0:
        raise ValueError("Input must be a positive integer.")

    power = 0
    while n > 1:
        if n % 2 != 0:  # Not divisible by 2
            return None
        n //= 2  # Divide by 2
        power += 1

    return power

def ByteEncode (F,d):
    """
    Encodes an array of d-bit integers into a byte array.
    d is the subscript of ByteEncode
    """
    # d = 0

    # for i in range(len(F)):
    #     d_new = Find_d(F[i])
    #     if d_new > d:
    #         d = d_new


    # if d < 1 or d > 12:
    #     raise ValueError("d must be between 1 and 12.")
    
    # Determine modulus based on d
    m = (2 ** d) if d < 12 else KYBER_Q

    # Initialize bit array b
    b = [0] * (256 * d)  # Total bits required: 256 * d

    # Iterate through each integer in F
    for i in range(256):
        a = F[i] % m  # Reduce integer modulo m
        for j in range(d):  # Process each bit of the integer
            b[i * d + j] = a % 2  # Extract the least significant bit
            a = (a - b[i * d + j]) // 2  # Remove the bit from a

    # Convert the bit array b into a byte array B
    B = BitsToBytes(b)
    return B


def ByteDecode(B,d):
    """
    Decodes a byte array into an array of d-bit integers.
    d is the subscript of ByteDecode
    """
    # Step 1: Convert byte array B into a bit array b
    b = BytesToBits(B)
    
    # Step 2: Determine modulus m based on d
    m = (2 ** d) if d < 12 else KYBER_Q
    
    # Step 3: Initialize the integer array F
    F = [0] * 256
    
    # Step 4: Decode each integer from the bit array
    for i in range(256):
        for j in range(d):
            F[i] += b[i * d + j] * (2 ** j)  # Calculate the contribution of each bit
        F[i] %= m  # Apply modulus m
    
    return F

def XOF():
        """
        eXtendable-Output Function (XOF) described in 4.9 of FIPS 203 (page 19)
        """
        return shake_128()

def SampleNTT(B):
    """
    Takes a 32-byte seed and two indices as input and outputs a pseudorandom element of 𝑇𝑞.
    """
    ctx = XOF.Init()
    ctx = XOF.Absorb(ctx,B)
    j = 0
    a = []
    while j < 256:
        ctx, C = XOF.Squeeze(ctx,3)
        d1 = C[0] + 256 * (C[1] % 16)
        d2 = (C[1] // 16) + 16 * C[2]

        if d1 < 3329:
            a.append(d1)
            j = j + 1

        if d2 < 3329 and j < 256:
            a.append(d2)
            j = j + 1
    return a


def SamplePolyCBD(B, eta):
    """
    Takes a seed as input and outputs a pseudorandom sample from the distribution D𝜂(𝑅𝑞)
    """
    b = BytesToBits(B)
    f = []
    x, y = 0
    for i in range(256):
        j = 0
        for j in range(eta):
            x += b[2*i*eta+j]  # Calculate the contribution of each bit
        j = 0
        for j in range(eta):
            y += b[2*i*eta+eta+j]  # Calculate the contribution of each bit
        f.append((x-y) % KYBER_Q)
        x = 0
        y = 0
    return f

def NTT(f, zetas):
    """
    Computes ̂ the NTT representation 𝑓 of the given polynomial 𝑓 ∈ 𝑅𝑞.
    """
    f_hat = f
    i = 1
    l = 128

    while l >= 2:
        start = 0
        while start < 256:
            zeta = zetas[i]
            i = i + 1
            for j in range(start, start + l):
                t = zeta * f_hat[j + l]
                f_hat[j + l] = f_hat[j] - t
                f_hat[j] = f_hat[j] + t
            start = l + (j + 1)
        l = l >> 1

    for j in range(256):
        f_hat[j] = f_hat[j] % 3329

    return f_hat

def NTT_inv(f_hat, zetas):
    """
    Computes ̂ the polynomial 𝑓 ∈ 𝑅𝑞 that corresponds to the given NTT representation 𝑓 ∈ 𝑇𝑞.
    """
    
    return f_hat
