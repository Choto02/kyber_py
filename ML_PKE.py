from hashlib import sha3_256, sha3_512, shake_128, shake_256

# Kyber-512 Parameters
KYBER_N = 256  # Polynomial degree
KYBER_Q = 3329  # Modulus
KYBER_K = 2  # Number of polynomials in the vector (k = 2 for Kyber-512)
du = 10
dv = 4
ETA1 = 3  # CBD parameter for Kyber-512
ETA2 = 2

# Zeta values for NTT and NTT_inv transforms
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

# Zeta values for NTT multiplication
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
    if len(b) % 8 != 0: # Ensure the length of input bit array multiple of 8
        raise ValueError("The length of the bit array must be a multiple of 8.")
    
    l = len(b) // 8  
    B = [0] * l      # Initialize byte array with zeros

    for i in range(len(b)):     # Iterate through the bit array
        B[i // 8] += b[i] * (2 ** (i % 8))  
    
    return B

def BytesToBits(B):
    """
    Converts a byte array (list of integers) into a bit array.
    """
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
    # Convert byte array B into a bit array b
    b = BytesToBits(B)

    # If necessary, pad b with zeros
    while len(b) < 256 * d:
        b.append(0)

    # Determine modulus m based on d
    m = (2 ** d) if d < 12 else KYBER_Q
    
    # Initialize the integer array F
    F = [0] * 256
    
    # Decode each integer from the bit array
    for i in range(256):
        for j in range(d):
            F[i] += b[i * d + j] * (2 ** j)  # Calculate the contribution of each bit
        F[i] %= m  # Apply modulus m
    
    return F

def XOF(bytes32, i, j):
    input_bytes = bytes32 + i + j
    if len(input_bytes) != 34:
        raise ValueError(
            "Input bytes should be one 32 byte array and 2 single bytes."
        )
    return shake_128(input_bytes).digest(840)

def SampleNTT(input_bytes):

    i, j = 0, 0
    a = []
    while j < KYBER_N:
        # Increasing i value gets you a fresh 3-byte array C from XOF every iteration
        d1 = input_bytes[i] + 256 * (input_bytes[i+1] % 16)
        d2 = (input_bytes[i+1] // 16) + 16 * input_bytes[i+2]

        if d1 < 3329:
            a.append(d1)
            j = j + 1

        if d2 < 3329 and j < KYBER_N:
            a.append(d2)
            j = j + 1
        i = i + 3
    return a
    
def SamplePolyCBD(B, eta):
    """
    Takes a seed as input and outputs a pseudorandom sample from the distribution D𝜂(𝑅𝑞)
    """
    b = BytesToBits(B)
    f = []
    x = 0
    y = 0
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
    f_hat = []
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
            start = start + 2*l
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
    # Ensure h_hat has the correct size
    n = len(f_hat)
    h_hat = [0] * n  # Initialize properly

    for i in range(128):  
        h_hat[2*i], h_hat[2*i + 1] = BaseCaseMultiply(
            f_hat[2*i], f_hat[2*i + 1], g_hat[2*i], g_hat[2*i + 1], zetas2[i]
        )

    return h_hat  # Make sure to return it

def BaseCaseMultiply (a0, a1, b0, b1, gamma):
    c0 = (a0*b0 + a1*b1*gamma) % 3329
    c1 = (a0*b1 + a1*b0) % 3329
    return (c0,c1)

def prf(eta, s, b):
    """
    Pseudorandom function described in 4.3 of FIPS 203 (page 18)
    """
    input_bytes = s + b
    if len(input_bytes) != 33:
        raise ValueError(
            "Input bytes should be one 32 byte array and one single byte."
        )
    return shake_256(input_bytes).digest(eta * 64)

def H(s):
    """
    Hash function described in 4.4 of FIPS 203 (page 18)
    """
    return sha3_256(s).digest()

def J(s):
    """
    Hash function described in 4.4 of FIPS 203 (page 18)
    """
    return shake_256(s).digest(32)

def G(s):
    """
    Hash function described in 4.5 of FIPS 203 (page 18)
    """
    h = sha3_512(s).digest()
    return h[:32], h[32:]

def compress_ele(x, d):
    """
    Compute round((2^d / q) * x) % 2^d
    """
    t = 1 << d
    y = (t * x + 1664) // 3329  # 1664 = 3329 // 2
    return y % t

def _decompress_ele(x, d):
    """
    Compute round((q / 2^d) * x)
    """
    t = 1 << (d - 1)
    y = (3329 * x + t) >> d
    return y

def compress(d, coeffs):
    """
    Compress every element of the input (matrix or list) to have at most `d` bits.
    """
    coeffs_new = []
    
    if isinstance(coeffs, int):  # Handle scalar input
        return compress_ele(coeffs, d)
    
    if isinstance(coeffs[0], list):  # If it's a matrix (list of lists)
        for row in coeffs:
            coeffs_new.append([compress_ele(ele, d) for ele in row])
    else:  # If it's a flat list
        coeffs_new = [compress_ele(ele, d) for ele in coeffs]
    
    return coeffs_new

def decompress(d, coeffs):
    """
    Decompress the polynomial by decompressing each coefficient

    NOTE: This as compression is lossy, we have
    x' = decompress(compress(x)), which x' != x, but is
    close in magnitude.
    """
    coeffs_new = [_decompress_ele(c, d) for c in coeffs]
    return coeffs_new

def Decode_Vector(input_bytes, d):
    # Bytes needed to decode a polynomial
    n = 32 * d

    elements = [
        ByteDecode(input_bytes[i : i + n], d)
        for i in range(0, len(input_bytes), n)
    ]

    return [item for sublist in elements for item in sublist]

def K_PKE_KeyGen(d):
    """
    Use randomness to generate an encryption key and a corresponding decryption key 
    """
    #VARIABLES
    i = 0
    j = 0
    A_hat = [[0 for _ in range(KYBER_K)] for _ in range(KYBER_K )]
    s = []
    s_hat = []
    e = []
    e_hat = []
    ek_pke = []
    dk_pke = []
    N = 0  # Set counter for PRF
    rho, sigma = G(d + bytes([KYBER_K]))  # I WAS MISSING SQUARE BRACKETS SMH

    # MATRIX A
    for i in range(KYBER_K):
        for j in range(KYBER_K):
            xof_bytes = XOF(rho, bytes([j]), bytes([i]))
            A_hat[i][j] = SampleNTT(xof_bytes)

    # VECTOR S
    for i in range(KYBER_K):
        s.append(SamplePolyCBD(prf(ETA1, sigma, bytes([N])), ETA1)) 
        N = N + 1
    
    # VECTOR E
    for i in range(KYBER_K):
        e.append(SamplePolyCBD(prf(ETA1, sigma, bytes([N])), ETA1)) 
        N = N + 1

    s_hat = [NTT(poly, zetas) for poly in s]
    e_hat = [NTT(poly, zetas) for poly in e]

    t_hat_temp = [[0 for _ in range(KYBER_N)] for _ in range(KYBER_K)]
    t_hat = [[0 for _ in range(KYBER_N)] for _ in range(KYBER_K)]
    
    # A_hat x s_hat
    for i in range(KYBER_K):
        for j in range(KYBER_K):
            temp_poly = MultiplyNTTs(A_hat[i][j], s_hat[j], zetas2)
            for k in range(256):  # Adding the two products (mat mult)
                t_hat_temp[i][k] = (t_hat_temp[i][k] + temp_poly[k]) % KYBER_Q

    # T_hat = T_hat_temp + e_hat
    for i in range(KYBER_K):
        for k in range(256):  
            t_hat[i][k] = (t_hat_temp[i][k] + e_hat[i][k]) % KYBER_Q
              
    #Encode Public Key
    for i in range(KYBER_K):
            ek_pke += (ByteEncode(t_hat[i], 12))
    ek_pke += rho

    #Encode Private Key
    for i in range(KYBER_K):
            dk_pke += (ByteEncode(s_hat[i], 12))

    # Convert lists to bytes before returning
    ek_pke_bytes = bytes(ek_pke)  
    dk_pke_bytes = bytes(dk_pke)  

    return (ek_pke_bytes, dk_pke_bytes)

def K_PKE_Decrypt(dk_pke, c):
    """
    Uses the decryption key to decrypt a ciphertext following
    Algorithm 15 (FIPS 203)
    """
    n = KYBER_K * du * 32  # Total bytes for KYBER_K compressed polynomials
    c1, c2 = c[:n], c[n:]
    u = []
    s_hat = [[0 for _ in range(KYBER_N)] for _ in range(KYBER_K)]

    # VECTOR U
    chunk_size = du * 32  # Bytes per compressed polynomial
    for i in range(KYBER_K): # Split c1 into chunks for each polynomial in  vector u
        chunk = c1[i*chunk_size : (i+1)*chunk_size]
        decoded_chunk = Decode_Vector(chunk, du)  
        decompressed_poly = decompress(du, decoded_chunk) 
        u.append(decompressed_poly)  
    
    v = decompress(dv, ByteDecode(c2, dv))

    # VECTOR S_HAT
    chunk_size = 12 * 32  # Bytes per compressed polynomial
    for i in range(KYBER_K):
        chunk = dk_pke[i*chunk_size : (i+1)*chunk_size]
        decoded_chunk = Decode_Vector(chunk, 12)  # Decode bytes to polynomial
        s_hat[i] = decoded_chunk # Append as a separate polynomial
    

    u_hat = [NTT(poly, zetas) for poly in u]
    
    # w = v′ − NTT_inv(s_hat_T x NTT(u′))
    vector_temp = [0 for _ in range(KYBER_N)] 
    for j in range(KYBER_K):
        temp_poly = MultiplyNTTs(s_hat[j], u_hat[j], zetas2)
        for k in range(256): 
            vector_temp[k] += temp_poly[k] % KYBER_Q
    vector_temp2 = NTT_inv(vector_temp,zetas)
    w = [(v[i] - vector_temp2[i]) % KYBER_Q for i in range(len(v))]
    
    m = ByteEncode(compress(1,w),1)

    return bytes(m)

def K_PKE_Encrypt(ek_pke,m,r):
    # Variables
    N = 0
    i = 0
    j = 0
    r_bold = []
    r_bold_hat = []
    AT_y_inv = []
    e1 = []
    t_hat_bytes, rho = ek_pke[:-32], ek_pke[-32:]
    t_hat = [0 for _ in range(KYBER_K)]

    # T_HAT
    for i in range(KYBER_K):
        #t_hat.append(Decode_Vector(t_hat_bytes[384*i:384*(i+1)], 12))
        t_hat[i] = Decode_Vector(t_hat_bytes[384*i:384*(i+1)], 12)
        
    # A_HAT
    A_hat = [[0 for _ in range(KYBER_K)] for _ in range(KYBER_K)]

    for i in range(KYBER_K):
        for j in range(KYBER_K):
            #A_hat[i][j] = SampleNTT(XOF(bytes(rho)+ bytes([i])+ bytes([j])))
            xof_bytes = XOF(bytes(rho), bytes([j]), bytes([i]))
            A_hat[i][j] = SampleNTT(xof_bytes)

    # R_BOLD
    i = 0
    for i in range (KYBER_K):
        r_bold.append(SamplePolyCBD(prf(ETA1,r,bytes([N])),ETA1)) 
        N = N + 1

    # e1_BOLD
    i = 0
    for i in range (KYBER_K):
        e1.append(SamplePolyCBD(prf(ETA2,r,bytes([N])),ETA2)) 
        N = N + 1

    # e2_BOLD
    e2 = SamplePolyCBD(prf(ETA2,r,bytes([N])),ETA2)

    # R_BOLD_HAT
    r_bold_hat = [NTT(poly, zetas) for poly in r_bold]

    # U_BOLD
    u_bold_temp = [[0 for _ in range(KYBER_N)] for _ in range(KYBER_K)]

    for i in range(KYBER_K):
        for j in range(KYBER_K):
            temp_poly = MultiplyNTTs(A_hat[j][i], r_bold_hat[j], zetas2)
            for k in range(256):  # Assuming each polynomial has 256 coefficients
                u_bold_temp[i][k] += temp_poly[k] % KYBER_Q

    u_bold = [NTT_inv(poly, zetas) for poly in u_bold_temp]

    for i in range(KYBER_K):
            for k in range(256):  # Assuming each polynomial has 256 coefficients
                u_bold[i][k] += e1[i][k] % KYBER_Q
    
    # MU 
    mu = decompress(1,ByteDecode(m,1))

    # v 
    v_temp = [0 for _ in range(KYBER_N)] 

    for j in range(KYBER_K):
            temp_poly = MultiplyNTTs(t_hat[j], r_bold_hat[j], zetas2)
            for k in range(256):  # Each polynomial has 256 coefficients
                v_temp[k] = (v_temp[k] + temp_poly[k]) % KYBER_Q
    
    v = NTT_inv(v_temp, zetas)

    for k in range(256):  
        v[k] += e2[k] % KYBER_Q
        v[k] += mu[k] % KYBER_Q

    # C1 and C2 
    compressed_polys = [compress(du, poly) for poly in u_bold]
    c1 = b"".join([bytes(ByteEncode(poly, du)) for poly in compressed_polys])
    c2 = bytes(ByteEncode(compress(dv, v), dv))

    return c1 + c2



if __name__ == "__main__":
    # TESTING KEYGEN
    d = b"1422"
    ek_pke, dk_pke = K_PKE_KeyGen(d)

    #TESTING ENCRYPT
    k, r = G(b"1422")
    #m = bytes(32)
    m = b"Ni hao ma".ljust(32, b'\x00')
    ciphertext = K_PKE_Encrypt(ek_pke, m,r)
    #print("Ciphertext:", ciphertext)

    #TESTING DECRYPT
    m = K_PKE_Decrypt(dk_pke, ciphertext)
    print("DECRYPTED PLAINTEXT IS: ", m)


