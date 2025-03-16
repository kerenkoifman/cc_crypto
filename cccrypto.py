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
    p0 = np.array([1, 0, 0, 0, 0, 0, 0, 1])
    # x^7
    p1 = np.array([0, 0, 0, 0, 0, 0, 0, 1])

    pq0 = np.array([1, 0, 1, 0, 0, 0, 0, 1, 0, 1])
    pq1 = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])




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

    # print('m3 =', m3)
    # print('m3[0] =', m3.A1)

    arr = np.array(m3.A1)
    u00 = arr[0::2]
    u10 = (u00 + 1) % 2
    u01 = arr[1::2]
    u11 = (u01 + 1) % 2

    # print('arr =', arr)
    # print('u00 =', u00)
    # print('u01 =', u01)
    # print('u10 =', u10)
    # print('u11 =', u11)


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