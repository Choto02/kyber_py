from kyber_py.ml_kem import ML_KEM_512

# TESTING KEYGEN
#d = bytes(1422)
d = b"1422"
ek, dk = ML_KEM_512._k_pke_keygen(d)
#print(ek)

# TESTING G HASH FUNCTION
#r = ML_KEM_512._G(bytes(1422))
#print("r: ",r)


#key, ct = ML_KEM_512.encaps(ek)
#_key = ML_KEM_512.decaps(dk, ct)
#assert key == _key
