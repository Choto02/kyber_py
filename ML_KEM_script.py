from kyber_py.ml_kem import ML_KEM_512
import os

# TESTING KEYGEN
#d = bytes(1422)
d = b"1422"
ek, dk = ML_KEM_512._k_pke_keygen(d)
#print(dk)

# TESTING ENCRYPT
k, r = ML_KEM_512._G(b"1422")
#m = bytes(32)
m = b"Ni hao ma".ljust(32, b'\x00')
c = ML_KEM_512._k_pke_encrypt(ek, m,r,)
#print("CIPHERTEXT IS: ",c)

# TESTING DECRYPT
m = ML_KEM_512._k_pke_decrypt(dk, c)
print("PLAINTEXT IS: ", m)

#key, ct = ML_KEM_512.encaps(ek)
#_key = ML_KEM_512.decaps(dk, ct)
#assert key == _key
