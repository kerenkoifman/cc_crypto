import random
import numpy as np
# import IPython.display as display
from matplotlib import pyplot as plt
import io
import base64
from numpy.linalg import inv


from scipy.ndimage import shift

# randomize
np.random.seed(12345)


# global variables
S = np.matrix([
    [1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 1],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 1, 1]])

S_1 = np.linalg.inv(S).astype(int) % 2
# print('S-1:', S_1)


p = np.random.permutation(30)
# print("Permutation vector:\n", p)

# Convert to permutation matrix
R = np.eye(30)[p].astype(int)
# print("\nPermutation matrix:\n", R)

# Inverse permutation vector
inv_p = np.argsort(p)
# print("\nInverse permutation vector:\n", inv_p)

# Inverse permutation matrix
R_inv = np.eye(30)[inv_p].astype(int)
# print("\nInverse permutation matrix:\n", R_inv)



def generate_public_key():
    # 1+x^2
    p0 = np.array([1, 0, 1])
    # 1+x+x^2
    p1 = np.array([1, 1, 1])

    g0 = np.array([1, 1])
    g1 = np.array([0, 1])
    g2 = np.array([1, 1])

    # 1+x^7
    q0 = np.array([1, 0, 0, 0, 0, 0, 0, 1])
    # x^7
    q1 = np.array([0, 0, 0, 0, 0, 0, 0, 1])

    pq0 = np.convolve(p0, q0) % 2
    pq1 = np.convolve(p1, q1) % 2


    v = np.zeros(shape=(1, 16), dtype=int)
    v[0,0] = 1
    v[0,1] = 1
    v[0,2] = 0
    v[0,3] = 1
    v[0,4] = 1
    v[0,5] = 1

    k_rows = 6
    Gp = np.zeros(shape=(k_rows, 16), dtype=int)
    t = v.copy()
    for i in range(k_rows):
        Gp[i] = t
        t[0] = shift(t[0], 2, cval=0)
        # print('t=', t)


    # not needed due to matrix use
    # m = np.array([1, 1, 1, 0, 0, 1])
    # d = m @ Gp
    # d = d % 2
    # print("d =", d)


    pqv = np.zeros(shape=(1, 30), dtype=int)
    for i in range(10):
        pqv[0,2*i] = pq0[i]
        pqv[0,2*i+1] = pq1[i]


    Gpq = np.zeros(shape=(k_rows, 30), dtype=int)
    t = pqv.copy()
    for i in range(k_rows):
        Gpq[i] = t
        t[0] = shift(t[0], 2, cval=0)
        # print('t=', t)

    # print('pqv:', pqv)
    # print('Gpq:', Gpq)

    l0 = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    l1 = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    L = [l0, l1]

    G_hat = np.zeros((6,30), dtype=int)
    # print('G_hat-shape:', G_hat.shape)
    # print('G_hat:', G_hat)

    for i in range(k_rows):
        G_hat[i] = random.choice(L)




    # print('G_hat:', G_hat)

    G_sum = (Gpq + G_hat) % 2


    G = (S@G_sum@R) % 2

    return G


def encrypt_msg(G, m):
    codeword = (m@G) % 2

    err = np.zeros(shape=(1, 30), dtype=int)
    error_count = 3
    for i in range(error_count):
        pos = random.randint(0,30-1)
        err[0,pos] = 1


    # print('err:         ', err)

    codeword_err = (codeword + err) % 2

    # print('codeword:    ', codeword)
    # print('codeword_err:', codeword_err)

    return codeword_err


def decrypt_msg(m2):
    m3 = (m2 @ R_inv) % 2
    print('m3 =', m3, m3.shape, type(m3))

    m3 = np.array([
        0, 1, 0, 1, 0, 1, 0, 1, 
        0, 1, 0, 1, 0, 1, 1, 0, 
        0, 0, 0, 1, 1, 1, 0, 1, 
        0, 1, 0, 0, 1, 1])
    
    m3 = np.matrix([m3])
    print('m3 =', m3, type(m3))
    print('m3[0] =', m3.A1)

    # 1+x^7
    q0 = np.array([1, 0, 0, 0, 0, 0, 0, 1])
    # x^7
    q1 = np.array([0, 0, 0, 0, 0, 0, 0, 1])

    arr = np.array(m3.A1)
    u00 = arr[0::2]
    u10 = (u00 + 1) % 2
    u01 = arr[1::2]
    u11 = (u01 + 1) % 2

    # print('arr =', arr)
    print('u00 =', u00)
    print('u01 =', u01)
    print('u10 =', u10)
    print('u11 =', u11)
    print('u00-type =', type(u00))

    # convert NumPy binary array to integer
    binary_u00 = u00.dot(1 << np.arange(u00.size))
    binary_q0 = q0.dot(1 << np.arange(q0.size))
    print('binary_u00=', bin(binary_u00), type(binary_u00))
    quot00, r = binary_poly_div(int(binary_u00), int(binary_q0))

    binary_u10 = u10.dot(1 << np.arange(u10.size))
    binary_q0 = q0.dot(1 << np.arange(q0.size))
    print('binary_u10=', bin(binary_u10), type(binary_u10))
    quot10, r = binary_poly_div(int(binary_u10), int(binary_q0))

    binary_u01 = u01.dot(1 << np.arange(u01.size))
    binary_q1 = q1.dot(1 << np.arange(q1.size))
    print('binary_u01=', bin(binary_u01), type(binary_u01))
    quot01, r = binary_poly_div(int(binary_u01), int(binary_q1))

    binary_u11 = u11.dot(1 << np.arange(u11.size))
    binary_q1 = q1.dot(1 << np.arange(q1.size))
    print('binary_u11=', bin(binary_u11), type(binary_u11))
    quot11, r = binary_poly_div(int(binary_u11), int(binary_q1))


    print("Quotient (poly):", bin_to_poly(quot11))
    print("Remainder (poly):", bin_to_poly(r))


    d00 = quot00
    print('d00=', bin(d00))
    d00_m = int_to_binary_matrix(d00, 8)
    print('d00_m=', d00_m)

    d10 = quot10
    print('d10=', bin(d10))
    d10_m = int_to_binary_matrix(d10, 8)
    print('d10_m=', d10_m)


    d01 = quot01
    print('d01=', bin(d01))
    d01_m = int_to_binary_matrix(d01, 8)
    print('d01_m=', d01_m)

    d11 = quot11
    print('d11=', bin(d11))
    d11_m = int_to_binary_matrix(d11, 8)
    print('d11_m=', d11_m)



    # flatten the matrices
    d00_m_flat = d00_m.flatten()
    d01_m_flat = d01_m.flatten()

    # interleave bitwise: a[0], b[0], a[1], b[1], ...
    d0 = np.ravel(np.column_stack((d00_m_flat, d01_m_flat)))

    # convert d0 to matrix
    # print("Merged bit-wise matrix (1x16):")
    # print(d0.reshape(1, -1))

    d00_m_flat = d00_m.flatten()
    d11_m_flat = d11_m.flatten()
    d2 = np.ravel(np.column_stack((d00_m_flat, d11_m_flat)))

    d10_m_flat = d10_m.flatten()
    d01_m_flat = d01_m.flatten()
    d1 = np.ravel(np.column_stack((d10_m_flat, d01_m_flat)))

    d10_m_flat = d10_m.flatten()
    d11_m_flat = d11_m.flatten()
    d3 = np.ravel(np.column_stack((d10_m_flat, d11_m_flat)))

    print('d0=', d0)
    print('d2=', d2)
    print('d1=', d1)
    print('d3=', d3)


def binary_poly_div(dividend: int, divisor: int):
    def degree(n): return n.bit_length() - 1

    quotient = 0
    remainder = dividend

    while degree(remainder) >= degree(divisor):
        shift = degree(remainder) - degree(divisor)
        quotient ^= 1 << shift
        remainder ^= divisor << shift

    return quotient, remainder

def bin_to_poly(n):
    terms = [f"x^{i}" if i > 1 else "x" if i == 1 else "1"
             for i in reversed(range(n.bit_length())) if (n >> i) & 1]
    return " + ".join(terms) if terms else "0"

# # x^14 + x^10 + x^7
# dividend = (1 << 14) | (1 << 10) | (1 << 7)
# # x^7 + 1
# divisor  = (1 << 7) | 1

# dividend = (1 << 13) | (1 << 8) | (1 << 7)
# divisor  = (1 << 7) 


# q, r = binary_poly_div(dividend, divisor)

# print("Quotient (poly):", bin_to_poly(q))
# print("Remainder (poly):", bin_to_poly(r))


def int_to_binary_matrix(n: int, length: int = None) -> np.ndarray:
    # Convert int to binary string without '0b'
    bits = bin(n)[2:]

    # Pad with leading zeros to reach desired length
    if length is not None:
        bits = bits.zfill(length)

    # Convert to NumPy array of ints (LSB on the left)
    bit_array = np.array([int(b) for b in reversed(bits)])
    return np.array([bit_array])  # Wrap as 1xN matrix



def cccrypto():

    G = generate_public_key()

    # define message - plaintext
    m = np.array([1, 1, 1, 0, 0, 1])

    # m2 is the encrypted message
    m2 = encrypt_msg(G, m)

    print(f'msg = {m}')
    print(f'encrypt-msg = {m2}')

    m_org = decrypt_msg(m2)

    print('m_org = ', m_org)


def test():
    # print('p0:', p0)
    # print('g0:', g0)
    print('v:' , v)
    print('Gp:' , Gp)



# define main function
def main():
    print("in main")

    # test()

    cccrypto()


# using the special variable __main__
if __name__ == "__main__":
    main()
