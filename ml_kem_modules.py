from hashlib import sha3_256, sha3_512, shake_128, shake_256

# Kyber-512 Parameters
KYBER_N = 256  # Polynomial degree
KYBER_Q = 3329  # Modulus
KYBER_K = 2  # Number of polynomials in the vector (k = 2 for Kyber-512)
ETA = 2  # CBD parameter for Kyber-512

zetas = [1, 1729, 2580, 3289, 2642, 630, 1897, 848,
    1062, 1919, 193, 797, 2786, 3260, 569, 1746,
    296, 2447, 1339, 1476, 3046, 56, 2240, 1333,
    1426, 2094, 535, 2882, 2393, 2879, 1974, 821,
    289, 331, 3253, 1756, 1197, 2304, 2277, 2055,
    650, 1977, 2513, 632, 2865, 33, 1320, 1915,
    2319, 1435, 807, 452, 1438, 2868, 1534, 2402,
    2647, 2617, 1481, 648, 2474, 3110, 1227, 910,
    17, 2761, 583, 2649, 1637, 723, 2288, 1100,
    1409, 2662, 3281, 233, 756, 2156, 3015, 3050,
    1703, 1651, 2789, 1789, 1847, 952, 1461, 2687,
    939, 2308, 2437, 2388, 733, 2337, 268, 641,
    1584, 2298, 2037, 3220, 375, 2549, 2090, 1645,
    1063, 319, 2773, 757, 2099, 561, 2466, 2594,
    2804, 1092, 403, 1026, 1143, 2150, 2775, 886,
    1722, 1212, 1874, 1029, 2110, 2935, 885, 2154]

zetas2 = [17, -17, 2761, -2761, 583, -583, 2649, -2649,
    1637, -1637, 723, -723, 2288, -2288, 1100, -1100,
    1409, -1409, 2662, -2662, 3281, -3281, 233, -233,
    756, -756, 2156, -2156, 3015, -3015, 3050, -3050,
    1703, -1703, 1651, -1651, 2789, -2789, 1789, -1789,
    1847, -1847, 952, -952, 1461, -1461, 2687, -2687,
    939, -939, 2308, -2308, 2437, -2437, 2388, -2388,
    733, -733, 2337, -2337, 268, -268, 641, -641,
    1584, -1584, 2298, -2298, 2037, -2037, 3220, -3220,
    375, -375, 2549, -2549, 2090, -2090, 1645, -1645,
    1063, -1063, 319, -319, 2773, -2773, 757, -757,
    2099, -2099, 561, -561, 2466, -2466, 2594, -2594,
    2804, -2804, 1092, -1092, 403, -403, 1026, -1026,
    1143, -1143, 2150, -2150, 2775, -2775, 886, -886,
    1722, -1722, 1212, -1212, 1874, -1874, 1029, -1029,
    2110, -2110, 2935, -2935, 885, -885, 2154, -2154]

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
    f = f_hat
    l = 2
    k = 127
    while l <= 128:
        start = 0
        while start < 256:
            zeta = zetas[k]
            k = k - 1
            for j in range(start, start + l):
                t = f[j]
                f[j] = t + f[j + l]
                f[j + l] = f[j + l] - t
                f[j + l] = zeta * f[j + l]
            start = j + l + 1
        l = l << 1

    for j in range(256):
        f[j] = (f[j] * 3303) % 3329
    
    return f

def MultiplyNTTs(f_hat, g_hat, zetas2):
    h_hat = []
    for i in range(128):
        h_hat[2*i], h_hat[2*i + 1] = BaseCaseMultiply(f_hat[2*i],f_hat[2*i + 1],g_hat[2*i],g_hat[2*i + 1], zetas2[i+1])
    return h_hat

def BaseCaseMultiply (a0, a1, b0, b1, gamma):
    c0 = a0*b0 + a1*b1*gamma
    c0 = c0 % 3329
    c1 = a0*b1 + a1*b0
    c1 = c1 % 3329
    return (c0,c1)

def K_PKE_KeyGen(d):
    """
    Use randomness to generate an encryption key and a corresponding decryption key 
    """
#########################################################
    rho, sigma = G(d + bytes([self.k]))

    # Generate A_hat from seed rho
    A_hat = self._generate_matrix_from_seed(rho)

    # Set counter for PRF
    N = 0

    # Generate the error vector s ∈ R^k
    s, N = self._generate_error_vector(sigma, self.eta_1, N)

    # Generate the error vector e ∈ R^k
    e, N = self._generate_error_vector(sigma, self.eta_1, N)

    # Compute public value (in NTT form)
    s_hat = s.to_ntt()
    e_hat = e.to_ntt()
    t_hat = A_hat @ s_hat + e_hat

    # Byte encode
    ek_pke = t_hat.encode(12) + rho
    dk_pke = s_hat.encode(12)

    return (ek_pke, dk_pke)
