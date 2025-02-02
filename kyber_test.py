from hashlib import sha3_256, sha3_512, shake_128, shake_256

# Kyber-512 Parameters
KYBER_N = 256  # Polynomial degree
KYBER_Q = 3329  # Modulus
KYBER_K = 2  # Number of polynomials in the vector (k = 2 for Kyber-512)
du = 10
dv = 4
ETA1 = 3  # CBD parameter for Kyber-512
ETA2 = 2

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

# def MultiplyNTTs(f_hat, g_hat, zetas2):
#     # Ensure h_hat has the correct size
#     n = len(f_hat)
#     h_hat = [0] * n  # Initialize properly

#     print(f"Lengths - f_hat: {len(f_hat)}, g_hat: {len(g_hat)}, zetas2: {len(zetas2)}")
#     print(f"h_hat initialized with length: {len(h_hat)}")
#     for i in range(n // 2):  
#         h_hat[2*i], h_hat[2*i + 1] = BaseCaseMultiply(
#             f_hat[2*i], f_hat[2*i + 1], g_hat[2*i], g_hat[2*i + 1], zetas2[i+1]
#         )

#     return h_hat  # Make sure to return it

def MultiplyNTTs(f_hat, g_hat,zetas):
    """
    Given the coefficients of two polynomials compute the coefficients of
    their product
    """
    new_coeffs = []
    for i in range(64):
        r0, r1 = BaseCaseMultiply(
            f_hat[4 * i + 0],
            f_hat[4 * i + 1],
            g_hat[4 * i + 0],
            g_hat[4 * i + 1],
            zetas[64 + i],
        )
        r2, r3 = BaseCaseMultiply(
            f_hat[4 * i + 2],
            f_hat[4 * i + 3],
            g_hat[4 * i + 2],
            g_hat[4 * i + 3],
            -zetas[64 + i],
        )
        new_coeffs += [r0, r1, r2, r3]
    return new_coeffs

def BaseCaseMultiply (a0, a1, b0, b1, gamma):
    c0 = a0*b0 + a1*b1*gamma
    c0 = c0 % 3329
    c1 = a0*b1 + a1*b0
    c1 = c1 % 3329
    return (c0,c1)

def XOF(bytes32, j, i):
    if not isinstance(bytes32, bytes):
        raise TypeError(f"bytes32 must be a bytes object, but got {type(bytes32)}")
    
    i_bytes = i.to_bytes(2, 'little')
    j_bytes = j.to_bytes(2, 'little')
    
    input_bytes = bytes32 + i_bytes + j_bytes  # Concatenation works with bytes

    return shake_128(input_bytes).digest(840)

def SampleNTT(input_bytes):

    i, j = 0, 0
    a = []
    while j < 256:
        d1 = input_bytes[0] + 256 * (input_bytes[1] % 16)
        d2 = (input_bytes[1] // 16) + 16 * input_bytes[2]

        if d1 < 3329:
            a.append(d1)
            j = j + 1

        if d2 < 3329 and j < 256:
            a.append(d2)
            j = j + 1
        i = i + 3
    return a

    
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

def transpose_matrix(matrix):
    """
    Transposes a given matrix (list of lists).
    """
    # Use list comprehension to transpose the matrix
    transposed = [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
    return transposed

def NTT(f, zetas):
    """
    Computes ̂ the NTT representation 𝑓 of the given polynomial 𝑓 ∈ 𝑅𝑞.
    """
    print(f"Length of f: {len(f)}, Length of zetas: {len(zetas)}")
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
            #start = l + (j + 1)
            start = start + 2*l
        l = l >> 1

    for j in range(256):
        f_hat[j] = f_hat[j] % 3329

    return f_hat

def K_PKE_Encrypt(ek_pke,m,r):
    N = 0
    i = 0
    j = 0
    r_bold = []
    r_bold_hat = []
    AT_y_inv = []
    e1 = []
    t_hat_bytes, rho = ek_pke[:-32], ek_pke[-32:]
    t_hat = [0 for _ in range(KYBER_K)]


    # for i in range(KYBER_K):
    #     #t_hat.append(Decode_Vector(t_hat_bytes[384*i:384*(i+1)], 12))
    #     t_hat[i] = Decode_Vector(t_hat_bytes[384*i:384*(i+1)], 12)
    #     print("T_HAT_BYTES: ",i, "Length: ", len(t_hat_bytes[384*i:384*(i+1)]))
        
    t_hat = [[2176, 2776, 2594, 3278, 2206, 3180, 2628, 1192, 2783, 977, 2247, 2803, 3081, 1395, 1732, 3114, 1634, 962, 1751, 3253, 1101, 1646, 2611, 1045, 759, 2032, 147, 3312, 652, 2712, 1258, 3211, 1191, 1335, 2723, 1907, 453, 1496, 2981, 2099, 561, 520, 1870, 2552, 105, 222, 1295, 0, 1519, 1114, 2249, 2247, 2212, 1364, 2308, 2738, 1602, 3172, 227, 3073, 2325, 2128, 492, 189, 689, 1409, 1773, 1341, 479, 370, 822, 2076, 1698, 839, 2116, 645, 2643, 2597, 80, 3307, 2743, 1850, 378, 1339, 1252, 123, 3161, 1945, 1657, 879, 591, 464, 37, 331, 1564, 2301, 1288, 270, 3227, 1138, 1800, 971, 2422, 611, 1847, 639, 2740, 1389, 208, 1927, 2486, 129, 155, 1093, 2482, 2951, 1061, 172, 1880, 1751, 602, 913, 1750, 2402, 1364, 1700, 1924, 3045, 320, 1430, 3254, 3163, 2715, 872, 1104, 2458, 951, 1951, 380, 3264, 1021, 1180, 837, 2162, 226, 1550, 476, 2069, 2970, 2375, 1824, 2534, 262, 2956, 412, 873, 230, 1317, 637, 2971, 1886, 210, 2239, 1087, 167, 1239, 488, 1722, 1250, 532, 769, 205, 2892, 1577, 1015, 2356, 1148, 427, 1411, 341, 3163, 22, 1102, 2997, 1690, 1334, 959, 179, 3257, 1262, 3230, 1544, 259, 506, 1519, 858, 802, 2712, 1792, 2560, 2416, 2208, 344, 2664, 629, 2029, 1705, 1419, 1725, 1822, 1370, 708, 2452, 1896, 2478, 2159, 1103, 3099, 13, 954, 809, 1561, 2947, 3254, 327, 2658, 1792, 2338, 1215, 2484, 2239, 1770, 858, 766, 3118, 1405, 684, 1275, 206, 2038, 1458, 1760, 2205, 706, 1964, 2815, 2611, 2691, 742, 818, 649, 3149, 2812, 2795, 1146, 985], [1518, 2169, 2769, 2222, 1018, 2652, 3076, 1763, 126, 1213, 2274, 2818, 2154, 1682, 200, 2470, 2697, 2697, 3290, 3017, 941, 2247, 1857, 429, 635, 224, 1514, 932, 2039, 518, 2342, 2486, 3121, 1438, 210, 774, 443, 249, 2766, 700, 416, 3327, 2480, 719, 47, 1001, 2512, 2743, 1731, 1513, 1184, 649, 2730, 2105, 2246, 2496, 3205, 204, 1728, 580, 995, 1126, 591, 3196, 3092, 1756, 3049, 3001, 1689, 1245, 3025, 1019, 702, 1788, 765, 2295, 2959, 391, 944, 3197, 2084, 3023, 1247, 350, 3306, 2761, 821, 934, 1203, 668, 3222, 3042, 822, 2023, 2094, 2888, 2846, 3095, 398, 2332, 1532, 845, 863, 697, 335, 2182, 2343, 1236, 405, 1446, 2587, 1232, 1803, 3119, 2698, 3127, 3115, 1532, 444, 1385, 2555, 3086, 1494, 1829, 223, 1975, 1064, 26, 3254, 1149, 174, 2388, 100, 3203, 517, 3101, 2774, 564, 650, 2988, 1825, 1274, 1338, 2759, 3289, 3085, 1827, 2160, 1739, 1510, 2130, 2177, 2809, 3175, 1280, 1604, 0, 2147, 1054, 3220, 2846, 1065, 771, 428, 669, 1078, 1306, 138, 1650, 1932, 277, 3262, 665, 647, 898, 1864, 2828, 2786, 1526, 1371, 2152, 1275, 2376, 2350, 1343, 954, 669, 1510, 285, 3053, 1147, 1514, 2094, 2020, 1836, 666, 2298, 3166, 1493, 2862, 1411, 1535, 2696, 790, 293, 300, 2317, 124, 909, 1441, 2991, 1825, 28, 1147, 2411, 2629, 2533, 1717, 1807, 2387, 965, 1866, 350, 2419, 2665, 2143, 1615, 855, 1094, 1329, 556, 3257, 824, 2760, 699, 1818, 1822, 2916, 1090, 1881, 1665, 3124, 1790, 2173, 3144, 634, 902, 2730, 2551, 2093, 695, 552, 1229, 3230, 729, 1751]]

    # print("T_HAT IS: ")
    # print(t_hat)
    
    A_hat = [[0 for _ in range(KYBER_K)] for _ in range(KYBER_K)]

    for i in range(KYBER_K):
        for j in range(KYBER_K):
            #A_hat[i][j] = SampleNTT(XOF(bytes(rho)+ bytes([i])+ bytes([j])))
            xof_bytes = XOF(bytes(rho), j, i)
            A_hat[i][j] = SampleNTT(xof_bytes)

    i = 0
    for i in range (KYBER_K):
        r_bold.append(SamplePolyCBD(prf(ETA1,r,bytes([N])),ETA1)) 
        N = N + 1

    i = 0
    for i in range (KYBER_K):
        e1.append(SamplePolyCBD(prf(ETA2,r,bytes([N])),ETA2)) 
        N = N + 1
    
    e2 = SamplePolyCBD(prf(ETA2,r,bytes([N])),ETA2)


    r_bold_hat = [NTT(poly, zetas) for poly in r_bold]
 

   # r_bold_hat = NTT(r_bold,zetas)
    A_hat_T = transpose_matrix(A_hat)

    # Compute t_hat = A_hat_T @ r_bold_hat 
    u_bold = [[0] * len(r_bold_hat[0]) for _ in range(KYBER_K)]
    #u_bold_temp = [0 for _ in range(KYBER_K)]

    for i in range(KYBER_K):
        for j in range(KYBER_K):
            u_bold[i] += MultiplyNTTs(A_hat[j][i],r_bold_hat[j],zetas2)

    print("u_bold: ",u_bold)




if __name__ == "__main__":
    ek_pke = [128, 136, 173, 34, 234, 204, 158, 200, 198, 68, 138, 74, 223, 26, 61, 199, 56, 175, 9, 60, 87, 196, 166, 194, 98, 38, 60, 215, 86, 203, 77, 228, 102, 51, 90, 65, 247, 2, 127, 147, 0, 207, 140, 130, 169, 234, 180, 200, 167, 116, 83, 163, 58, 119, 197, 129, 93, 165, 59, 131, 49, 130, 32, 78, 135, 159, 105, 224, 13, 15, 5, 0, 239, 165, 69, 201, 120, 140, 164, 72, 85, 4, 41, 171, 66, 70, 198, 227, 16, 192, 21, 9, 133, 236, 209, 11, 177, 18, 88, 237, 214, 83, 223, 33, 23, 54, 195, 129, 162, 118, 52, 68, 88, 40, 83, 90, 162, 80, 176, 206, 183, 170, 115, 122, 177, 83, 228, 180, 7, 89, 156, 121, 121, 246, 54, 79, 2, 29, 37, 176, 20, 28, 214, 143, 8, 229, 16, 155, 44, 71, 8, 183, 60, 118, 57, 38, 55, 247, 39, 180, 218, 86, 208, 112, 120, 182, 25, 8, 155, 80, 68, 178, 121, 184, 37, 196, 10, 88, 119, 109, 90, 18, 57, 214, 38, 150, 84, 69, 106, 132, 87, 190, 64, 97, 89, 182, 188, 197, 155, 138, 54, 80, 164, 153, 183, 243, 121, 124, 1, 204, 253, 195, 73, 69, 35, 135, 226, 224, 96, 220, 81, 129, 154, 123, 148, 32, 103, 158, 6, 193, 184, 156, 145, 54, 230, 80, 82, 125, 178, 185, 94, 39, 13, 191, 248, 67, 167, 112, 77, 232, 161, 107, 226, 68, 33, 1, 211, 12, 76, 155, 98, 247, 67, 147, 124, 180, 26, 131, 85, 21, 91, 108, 1, 78, 84, 187, 154, 102, 83, 191, 51, 11, 185, 236, 78, 158, 140, 96, 3, 161, 31, 239, 165, 53, 34, 131, 169, 0, 7, 160, 112, 9, 138, 88, 129, 166, 117, 210, 126, 169, 182, 88, 189, 230, 113, 90, 69, 44, 148, 137, 118, 174, 249, 134, 79, 180, 193, 13, 160, 59, 41, 147, 97, 131, 107, 203, 71, 33, 166, 0, 39, 146, 191, 68, 155, 191, 168, 110, 90, 227, 47, 46, 220, 87, 172, 178, 79, 206, 96, 127, 178, 5, 110, 157, 40, 44, 172, 247, 175, 51, 58, 168, 230, 34, 51, 137, 210, 196, 252, 186, 174, 122, 148, 61, 238, 149, 135, 209, 234, 138, 250, 195, 165, 4, 60, 110, 126, 208, 75, 226, 40, 176, 106, 40, 105, 200, 96, 154, 137, 154, 168, 218, 156, 188, 173, 115, 140, 65, 215, 26, 123, 2, 14, 234, 69, 58, 247, 103, 32, 38, 105, 155, 49, 236, 89, 210, 96, 48, 187, 145, 15, 206, 202, 43, 160, 241, 207, 176, 249, 44, 47, 144, 62, 208, 121, 171, 195, 150, 94, 160, 148, 40, 170, 154, 131, 198, 8, 156, 133, 204, 12, 192, 70, 36, 227, 99, 70, 79, 194, 199, 20, 204, 109, 233, 155, 187, 153, 214, 77, 209, 187, 63, 190, 194, 111, 253, 114, 143, 143, 123, 24, 176, 211, 199, 36, 248, 188, 223, 228, 21, 234, 156, 172, 53, 99, 58, 179, 196, 41, 150, 44, 190, 54, 115, 126, 46, 136, 180, 30, 123, 193, 142, 193, 145, 252, 213, 52, 95, 147, 43, 79, 97, 136, 39, 73, 77, 149, 97, 90, 27, 10, 77, 11, 247, 194, 138, 122, 195, 43, 204, 95, 188, 145, 86, 251, 233, 192, 214, 85, 114, 223, 112, 123, 40, 164, 1, 182, 220, 71, 174, 64, 149, 100, 48, 200, 5, 210, 193, 214, 74, 35, 138, 194, 186, 33, 167, 79, 58, 117, 172, 217, 220, 192, 35, 7, 135, 203, 102, 94, 82, 24, 136, 249, 122, 198, 0, 69, 100, 0, 48, 134, 30, 68, 201, 30, 155, 66, 3, 195, 26, 157, 98, 67, 26, 165, 8, 114, 198, 120, 21, 225, 203, 153, 114, 40, 130, 131, 116, 12, 43, 174, 246, 181, 85, 104, 184, 79, 72, 233, 146, 63, 165, 59, 157, 98, 94, 29, 209, 190, 123, 164, 94, 46, 72, 126, 44, 167, 41, 250, 232, 197, 213, 229, 178, 131, 245, 95, 136, 106, 49, 37, 193, 18, 13, 201, 7, 141, 19, 90, 175, 27, 114, 28, 176, 71, 107, 89, 164, 229, 89, 107, 15, 55, 149, 197, 163, 116, 94, 49, 151, 105, 250, 133, 79, 118, 53, 70, 20, 83, 44, 146, 203, 56, 131, 172, 187, 162, 113, 30, 71, 182, 66, 148, 117, 129, 70, 195, 254, 214, 135, 72, 172, 39, 134, 163, 170, 247, 217, 130, 183, 130, 34, 205, 228, 201, 217, 114, 109, 203, 243, 68, 60, 114, 120, 140, 13, 148, 205, 30, 64, 144, 138, 126, 183, 69, 208, 177, 11, 162, 110, 104, 166, 38, 2, 213, 174, 238, 13, 160, 150]
    t_hat_bytes, rho = ek_pke[:-32], ek_pke[-32:]
    r = H(bytes(123))
    ciphertext = K_PKE_Encrypt(ek_pke, bytes(12345),r)
