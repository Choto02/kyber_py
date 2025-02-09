from kyber_py.ml_kem import ML_KEM_512

#ek, dk = ML_KEM_512.keygen()
d = bytes(1422)
ek, dk = ML_KEM_512._k_pke_keygen(d)
#print(ek)

rho, sigma = ML_KEM_512._G(d + bytes([ML_KEM_512.k]))
print(rho)

#key, ct = ML_KEM_512.encaps(ek)
#_key = ML_KEM_512.decaps(dk, ct)
#assert key == _key