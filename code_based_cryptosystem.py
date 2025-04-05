import random
import numpy as np

# import IPython.display as display
from matplotlib import pyplot as plt
from trellis import *
import io
import base64
from numpy.linalg import inv
from scipy.ndimage import shift

# randomize
np.random.seed(12345)

# global variables
S = np.matrix(
    [
        [1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 1],
    ]
)

S_1 = np.linalg.inv(S).astype(int) % 2
p = np.random.permutation(30)

# Convert to permutation matrix
R = np.eye(30)[p].astype(int)

# Inverse permutation vector
inv_p = np.argsort(p)

# Inverse permutation matrix
R_inv = np.eye(30)[inv_p].astype(int)

def generate_Gp():
    # Define the convolutional code polynomials p0 and p1
    # 1+x^2
    p0 = np.array([1, 0, 1])
    # 1+x+x^2
    p1 = np.array([1, 1, 1])

    # Define coefficient matrices
    g0 = np.array([1, 1])
    g1 = np.array([0, 1])
    g2 = np.array([1, 1])
    
    # Parameters
    k_rows = 6
    n = 2  # Number of output bits per input bit (rate 1/2)
    p = 2  # Memory length (p0 and p1 are degree 2 polynomials)
    
    # Create the generator matrix
    Gp = np.zeros((k_rows, (k_rows+p)*n), dtype=int)
    
    # Fill the matrix with coefficients according to the convolutional structure
    coefficients = [g0, g1, g2]
    for i, g in enumerate(coefficients):
        for row in range(k_rows):
            col_pos = (row + i) * n
            if col_pos < Gp.shape[1]:
                Gp[row, col_pos:col_pos+n] = g
    
    return Gp

def generate_Gpq():
    # Define the convolutional code polynomials p0 and p1
    p0 = np.array([1, 0, 1])  # 1+x^2
    p1 = np.array([1, 1, 1])  # 1+x+x^2

    # Define high-memory polynomials
    q0 = np.array([1, 0, 0, 0, 0, 0, 0, 1])  # 1+x^7
    q1 = np.array([0, 0, 0, 0, 0, 0, 0, 1])  # x^7

    # Multiply the polynomials and take mod 2
    pq0 = np.convolve(p0, q0) % 2
    pq1 = np.convolve(p1, q1) % 2

    # Parameters
    k_rows = 6
    n = 2  # Number of output bits per input bit (rate 1/2)

    pqv = np.zeros(shape=(1, 30), dtype=int)
    for i in range(len(pq0)):
        if 2*i < len(pqv[0]):
            pqv[0, 2*i] = pq0[i]
        if 2*i+1 < len(pqv[0]):
            pqv[0, 2*i+1] = pq1[i]

    Gpq = np.zeros(shape=(k_rows, 30), dtype=int)
    t = pqv.copy()
    for i in range(k_rows):
        Gpq[i] = t
        t[0] = shift(t[0], 2, cval=0)
    
    return Gpq

def encrypt_msg(G, m):
    codeword = (m @ G) % 2 # multiplies the plaintext message m with the generator matrix G using matrix multiplication

    # create error vector
    err = np.zeros(shape=(1, 30), dtype=int)
    error_count = 3
    # randomly introduce errors in the error vector
    for i in range(error_count):
        pos = random.randint(0, 30 - 1)
        err[0, pos] = 1

    # print('err:         ', err)

    codeword_err = (codeword + err) % 2

    # print('codeword:    ', codeword)
    # print('codeword_err:', codeword_err)

    return codeword_err


def decrypt_msg(encrypted_message):
    inverse_permuted_msg = (encrypted_message @ R_inv) % 2
    print("\ninverse permuted message =", inverse_permuted_msg.flatten())
    inverse_permuted_msg = np.matrix(inverse_permuted_msg)
    print("inverse permuted message =", inverse_permuted_msg.A1)

    # 1+x^7
    q0 = np.array([1, 0, 0, 0, 0, 0, 0, 1])
    # x^7
    q1 = np.array([0, 0, 0, 0, 0, 0, 0, 1])

    # create masks
    arr = np.array(inverse_permuted_msg.A1)
    even_bits_zeros = arr[0::2]
    even_bits_ones = (even_bits_zeros + 1) % 2
    odd_bits_zeros = arr[1::2]
    odd_bits_ones = (odd_bits_zeros + 1) % 2

    print("\neven_bits_zeros =", even_bits_zeros)
    print("odd_bits_zeros =", odd_bits_zeros)
    print("even_bits_ones =", even_bits_ones)
    print("odd_bits_ones =", odd_bits_ones)
    print("even_bits_zeros-type =", type(even_bits_zeros))

    # convert NumPy binary array to integer
    binary_even_bits_zeros = even_bits_zeros.dot(1 << np.arange(even_bits_zeros.size))
    binary_q0 = q0.dot(1 << np.arange(q0.size))
    print("\nbinary_even_bits_zeros=", bin(binary_even_bits_zeros), type(binary_even_bits_zeros))
    quot00, r = binary_poly_div(int(binary_even_bits_zeros), int(binary_q0))

    binary_even_bits_ones = even_bits_ones.dot(1 << np.arange(even_bits_ones.size))
    binary_q0 = q0.dot(1 << np.arange(q0.size))
    print("binary_even_bits_ones=", bin(binary_even_bits_ones), type(binary_even_bits_ones))
    quot10, r = binary_poly_div(int(binary_even_bits_ones), int(binary_q0))

    binary_odd_bits_zeros = odd_bits_zeros.dot(1 << np.arange(odd_bits_zeros.size))
    binary_q1 = q1.dot(1 << np.arange(q1.size))
    print("binary_odd_bits_zeros=", bin(binary_odd_bits_zeros), type(binary_odd_bits_zeros))
    quot01, r = binary_poly_div(int(binary_odd_bits_zeros), int(binary_q1))

    binary_odd_bits_ones = odd_bits_ones.dot(1 << np.arange(odd_bits_ones.size))
    binary_q1 = q1.dot(1 << np.arange(q1.size))
    print("binary_odd_bits_ones=", bin(binary_odd_bits_ones), type(binary_odd_bits_ones))
    quot11, r = binary_poly_div(int(binary_odd_bits_ones), int(binary_q1))

    print("\nQuotient (poly):", bin_to_poly(quot11))
    print("Remainder (poly):", bin_to_poly(r))

    d00 = quot00
    print("\nd00=", bin(d00))
    d00_m = int_to_binary_matrix(d00, 8)
    print("d00_m=", d00_m)

    d10 = quot10
    print("d10=", bin(d10))
    d10_m = int_to_binary_matrix(d10, 8)
    print("d10_m=", d10_m)

    d01 = quot01
    print("d01=", bin(d01))
    d01_m = int_to_binary_matrix(d01, 8)
    print("d01_m=", d01_m)

    d11 = quot11
    print("d11=", bin(d11))
    d11_m = int_to_binary_matrix(d11, 8)
    print("d11_m=", d11_m)

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

    print(f"\nd0 = {d0}")
    print(f"d1 = {d1}")
    print(f"d2 = {d2}")
    print(f"d3 = {d3}\n")
    return d0,d1,d2,d3


def binary_poly_div(dividend: int, divisor: int):
    def degree(n):
        return n.bit_length() - 1

    quotient = 0
    remainder = dividend

    while degree(remainder) >= degree(divisor):
        shift = degree(remainder) - degree(divisor)
        quotient ^= 1 << shift
        remainder ^= divisor << shift

    return quotient, remainder


def bin_to_poly(n):
    terms = [
        f"x^{i}" if i > 1 else "x" if i == 1 else "1"
        for i in reversed(range(n.bit_length()))
        if (n >> i) & 1
    ]
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


def test():
    # print('p0:', p0)
    # print('g0:', g0)
    print("v:", v)
    print("Gp:", Gp)

def print_polynomials(p0, p1):
    # Define coefficient matrices
    g0 = np.array([1, 1])
    g1 = np.array([0, 1])
    g2 = np.array([1, 1])
    
    # Print polynomials
    print("\nPolynomials:")
    print(f"p0 = {p0} (1+x^2)")
    print(f"p1 = {p1} (1+x+x^2)")
    
    # Print coefficient matrices
    print("\nCoefficient Matrices:")
    print(f"g0 = {g0}")
    print(f"g1 = {g1}")
    print(f"g2 = {g2}")
    print(f"0  = [0 0]")
    
def print_matrices(Gp, Gpq):
    print("\nGenerator Matrix Gp:")
    print("-----------------------------------------")
    for row in Gp:
        print(' '.join(map(str, row)))
    
    print("\nHigh-Memory Generator Matrix Gpq:")
    print("----------------------------------")
    for row in Gpq:
        print(' '.join(map(str, row)))

def print_permutation_matrices():
    print("\nPermutation Matrix R:")
    print("-----------------------------------------")
    for row in R:
        print(' '.join(map(str, row)))
    
    print("\nInverse Permutation Matrix R_inv:")
    print("-----------------------------------------")
    for row in R_inv:
        print(' '.join(map(str, row)))
    
    print("\nPermutation vector p:")
    print(p)
    
    print("\nInverse permutation vector inv_p:")
    print(inv_p)

def find_min_path(trellis):
    num_cols = len(trellis[0])
    min_weight = 50
    best_path = None

    def dfs(node, curr_weight, curr_infobits):
        nonlocal min_weight, best_path
        # base case
        if node.col == num_cols - 1:
            if curr_weight <= min_weight:
                min_weight = curr_weight
                best_path = curr_infobits.copy()
            return
            
        if len(node.connections) == 0:
            return

        for next_node, info_bit, output, weight in node.connections:
            curr_infobits.append(info_bit)
            dfs(next_node, curr_weight + weight, curr_infobits)
            curr_infobits.pop()

    # start only from initial state '00' at column 0
    dfs(trellis[0][0], 0, [])

    return min_weight, best_path

# define main function
def main():
    # Define the convolutional code polynomials
    p0 = [1, 0, 1]  # 1+x^2
    p1 = [1, 1, 1]  # 1+x+x^2
    K = 3
    input_len = 6

    d0 = [0,0,0,0,0,1,1,1,0,1,0,1,0,0,1,1]
    d1 = [0,1,0,1,0,0,1,0,0,0,0,0,0,1,1,0]
    d2 = [0,0,1,0,1,1,0,1,1,1,1,1,1,0,0,1]
    d3 = [0,1,1,1,1,0,0,0,1,0,1,0,1,1,0,0]
    d_i = [d0, d1, d2, d3]
    print_polynomials(p0,p1)

    Gp = generate_Gp()
    Gpq = generate_Gpq()
        
    print_matrices(Gp, Gpq)
    msg = np.array([1, 1, 1, 0, 0, 1])
    encrypted_message = encrypt_msg(Gpq, msg)

    print_permutation_matrices() 
    print(f"\nmessage = {msg}")
    print(f"\nencrypted message = {encrypted_message.flatten()}")
    quotient_arr = decrypt_msg(encrypted_message)

    overall_min_dist = 100
    i = 0
    for d in d_i:
        trellis = create_trellis(p0,p1,input_len,K,d)
        min_dist, info_word = find_min_path(trellis)
        transformed_plaintext = np.array(info_word[:-2], dtype=int).reshape(1, 6)
        original_plain_text_m = (transformed_plaintext @ S_1) % 2
        if min_dist < overall_min_dist:
            min_decoder = f"d{i}"

        print(f"d{i} = {d}")
        print(f"minimal distance = {min_dist}")
        print(f"info_word = {info_word}")
        print(f"transformed plaintext = {transformed_plaintext.flatten()}") # m\hat S
        print(f"original plain text (m) = {np.array(original_plain_text_m).flatten()}\n")
        i+=1


    print(f"The valid viterbi decoder is: {min_decoder}\n")

    visualize_trellis(create_trellis(p0,p1,input_len,K,d3))

# using the special variable __main__
if __name__ == "__main__":
    main()

