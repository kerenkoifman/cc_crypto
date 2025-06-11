import random
import numpy as np
from matplotlib import pyplot as plt
from trellis import *

# randomize
np.random.seed(12345)

def generate_random_nonsingular_binary_matrix(K):
    while True:
        S = np.random.randint(0, 2, size=(K, K), dtype=int)
        try:
            S_inv = np.linalg.inv(S)
            S_inv = (np.round(S_inv) % 2).astype(int)
            if (S @ S_inv % 2 == np.eye(K, dtype=int)).all():
                disp_matrix(S, "Random nonsingular binary matrix S")
                disp_matrix(S_inv, "Inverse nonsingular Matrix S⁻¹")
                return S, S_inv
        except np.linalg.LinAlgError:
            continue

def binary_poly_divideide(dividend: int, divisor: int):
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

# LSB on the left
def int_to_binary_matrix(n: int, length: int = None) -> np.ndarray:
    # Convert int to binary string without '0b'
    bits = bin(n)[2:]

    # Pad with leading zeros to reach desired length
    if length is not None:
        bits = bits.zfill(length)

    # Convert to NumPy array of ints (LSB on the left)
    bit_array = np.array([int(b) for b in reversed(bits)])
    return np.array([bit_array])  # Wrap as 1xN matrix

# MSB on the left
def int_to_binary_matrix2(n, length=None):
    # Convert to binary string without '0b'
    bin_str = bin(n)[2:]
    
    # Optionally pad with zeros to a fixed length
    if length:
        bin_str = bin_str.zfill(length)
    
    # Convert to NumPy array of integers
    return np.array([int(bit) for bit in bin_str])


def add_binary_polynomials(p1, p2):
    # Pad the shorter one with leading zeros
    max_len = max(len(p1), len(p2))
    p1 = np.pad(p1, (max_len - len(p1), 0))
    p2 = np.pad(p2, (max_len - len(p2), 0))
    
    return (p1 ^ p2)  # XOR for GF(2) addition

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

def apply_crc_to_message(msg):
    # multiply by x^3
    crc_r = [1, 0, 0, 0]  # x^3, r = 3
    crc_p = [1, 0, 1, 1]  # x^3+x+1

    crc_msg = np.convolve(msg, crc_r) % 2
    print(f'crc_msg={crc_msg}')

    # Convert list to binary string and then to integer
    crc_p_bin_str = ''.join(str(bit) for bit in crc_p)
    crc_p_int = int(crc_p_bin_str, 2)

    crc_msg_bin_str = ''.join(str(bit) for bit in crc_msg)
    crc_msg_int = int(crc_msg_bin_str, 2)

    # crc_msg_int = crc_msg.dot(1 << np.arange(crc_msg.size))
    # crc_p_int = np_crc_p.dot(1 << np.arange(np_crc_p.size))
    # print("crc_msg_int=", bin(crc_msg_int), type(crc_msg_int))
    # print("crc_p_int=", bin(crc_p_int), type(crc_p_int))
    quot, rem = binary_poly_divide(int(crc_msg_int), int(crc_p_int))
    # print()
    # print("Quotient (poly):", bin_to_poly(quot))
    # print("Remainder (poly):", bin_to_poly(rem))
    rem_m = int_to_binary_matrix2(rem, 3)
    # print("Remainder (matrix):", rem_m)
    crc_msg2 = add_binary_polynomials(crc_msg, rem_m)
    # print("crc_msg2:", crc_msg2)
    return crc_msg2

def extract_polynomial_coefficients(p0_lst, p1_lst):
    """
    Extract coefficient matrices g_i from polynomial representations
    
    Args:
        p0_lst: coefficients of p0(x) as list [a0, a1, a2, ...]
        p1_lst: coefficients of p1(x) as list [b0, b1, b2, ...]
    
    Returns:
        list of g_i matrices where g_i = [coeff_of_x^i_in_p0, coeff_of_x^i_in_p1]
    """
    # Convert to numpy arrays and ensure same length
    p0 = np.array(p0_lst)
    p1 = np.array(p1_lst)
    
    # Pad shorter polynomial with zeros
    max_len = max(len(p0), len(p1))
    if len(p0) < max_len:
        p0 = np.pad(p0, (0, max_len - len(p0)))
    if len(p1) < max_len:
        p1 = np.pad(p1, (0, max_len - len(p1)))
    
    # Extract coefficient matrices
    g_matrices = []
    for i in range(max_len):
        g_i = np.array([p0[i], p1[i]])
        g_matrices.append(g_i)
        
    print("Coefficient matrices:")
    for i, g in enumerate(g_matrices):
        print(f"g{i} = [{' '.join(map(str, g))}]")
    print("0 = [0 0]\n")
        
    return g_matrices

def generate_Gp(p0_lst, p1_lst, K, n):
    """
    Generate convolutional code generator matrix using proper coefficient extraction
    
    Args:
        p0_lst: polynomial p0 coefficients [a0, a1, a2, ...]
        p1_lst: polynomial p1 coefficients [b0, b1, b2, ...]
        K: number of information bits (rows)
        n: number of output bits per input (columns per shift)
    
    Returns:
        Gp: K x N generator matrix
    """
    # Extract coefficient matrices
    g_matrices = extract_polynomial_coefficients(p0_lst, p1_lst)
    p = len(g_matrices) - 1  # Memory length (degree of polynomials)
    
    # Calculate matrix dimensions
    N = n * (K + p)  # Total number of columns
    
    # Initialize generator matrix
    Gp = np.zeros((K, N), dtype=int)
    
    # Fill the matrix
    for row in range(K):
        for i, g_i in enumerate(g_matrices):
            col_start = (row + i) * n
            if col_start + n <= N:
                Gp[row, col_start:col_start + n] = g_i
    
    print("=" * 80)  
    disp_matrix(Gp, "Generator Matrix Gp")
        
    return Gp

def poly_degree(poly):
    """Return the degree of a binary polynomial (LSB first)."""
    nonzero_indices = np.nonzero(poly)[0]
    return nonzero_indices[-1] if len(nonzero_indices) > 0 else -1  # -1 means zero polynomial

def multiply_polynomials_gf2(p_poly, q_poly):
    """
    Multiply two polynomials in GF(2) (binary field)
    
    Args:
        p_poly: first polynomial coefficients
        q_poly: second polynomial coefficients
    
    Returns:
        product polynomial coefficients
    """
    # Convolution followed by mod 2
    result = np.convolve(p_poly, q_poly) % 2
    return result

def generate_Gpq(p_polynomials, q_polynomials, K):
    """
    Create the high-memory generator matrix Gpq
    
    Args:
        p_polynomials: list of p polynomial coefficient arrays
        q_polynomials: list of q polynomial coefficient arrays  
        K: number of information bits (rows)
    
    Returns:
        Gpq matrix
    """
    n = len(p_polynomials)
    
    # Multiply corresponding p and q polynomials
    pq_products = []
    for i in range(n):
        pq_product = multiply_polynomials_gf2(p_polynomials[i], q_polynomials[i])
        pq_products.append(pq_product)
    
    # Calculate dimensions
    max_degree = max(len(pq) - 1 for pq in pq_products)
    N = n * (K + max_degree)
    
    # Create coefficient matrices from products
    g_matrices = []
    max_len = max(len(pq) for pq in pq_products)
    
    for degree in range(max_len):
        g_i = []
        for pq in pq_products:
            if degree < len(pq):
                g_i.append(pq[degree])
            else:
                g_i.append(0)
        g_matrices.append(np.array(g_i))
    
    # Build the matrix with diagonal structure
    Gpq = np.zeros((K, N), dtype=int)
    
    for row in range(K):
        for i, g_i in enumerate(g_matrices):
            col_start = (row + i) * n
            if col_start + n <= N:
                Gpq[row, col_start:col_start + n] = g_i
    
    disp_matrix(Gpq, "Generator matrix Gpq:")
    
    return Gpq

def generate_masking_matrix(K, N, n=2):
    """
    Simple version for integration into your existing code
    
    Args:
        K: number of rows 
        N: number of columns
        n: rate parameter (default 2 for rate 1/2)
    
    Returns:
        G_tilde: K x N masking matrix
    """
    
    # Create the two alternating patterns from the paper
    l0 = np.array([1, 0] * (N // 2 + 1))[:N]  # [1,0,1,0,1,0,...]
    l1 = np.array([0, 1] * (N // 2 + 1))[:N]  # [0,1,0,1,0,1,...]
    L = [l0, l1]
    
    # Create masking matrix by randomly choosing rows from L
    G_tilde = np.zeros((K, N), dtype=int)
    for row in range(K):
        G_tilde[row] = random.choice(L)
    
    disp_matrix(G_tilde, "Masked Generator Matrix G̃")
    
    return G_tilde

def compute_unmasked_variants(even_bits, odd_bits):
    # Convert to numpy arrays and ensure int type
    even = np.array(even_bits, dtype=int)
    odd = np.array(odd_bits, dtype=int)
    
    # Create mask vectors (all zeros and all ones)
    zeros_mask = np.zeros(len(even), dtype=int)
    ones_mask = np.ones(len(even), dtype=int)
    # Compute four variants by subtracting masks (XOR in GF(2))
    variants = {}
    
    # Variant 1: (c̃(x) - 0(x))_0 - even bits minus zeros mask
    variants['even_minus_zeros'] = (even - zeros_mask) % 2
    print(f"(c̃(x)-0(x))₀ = [" + " ".join(map(str, variants['even_minus_zeros'])) + "]")
    
    # Variant 2: (c̃(x) - 1(x))_0 - even bits minus ones mask  
    variants['even_minus_ones'] = (even - ones_mask) % 2
    print(f"(c̃(x)-1(x))₀ = [" + " ".join(map(str, variants['even_minus_ones'])) + "]")
    
    # Variant 3: (c̃(x) - 0(x))_1 - odd bits minus zeros mask
    variants['odd_minus_zeros'] = (odd - zeros_mask[:len(odd)]) % 2
    print(f"(c̃(x)-0(x))₁ = [" + " ".join(map(str, variants['odd_minus_zeros'])) + "]")
    
    # Variant 4: (c̃(x) - 1(x))_1 - odd bits minus ones mask
    variants['odd_minus_ones'] = (odd - ones_mask[:len(odd)]) % 2
    print(f"(c̃(x)-1(x))₁ = [" + " ".join(map(str, variants['odd_minus_ones'])) + "]\n")
    
    return variants
def binary_vector_to_int(binary_vector):
    """
    Convert binary vector to integer (LSB first convention)
    
    Args:
        binary_vector: list/array of binary coefficients [a0, a1, a2, ...]
                      where polynomial is a0 + a1*x + a2*x^2 + ...
    
    Returns:
        integer representation of the polynomial
    """
    result = 0
    for i, bit in enumerate(binary_vector):
        if bit == 1:
            result |= (1 << i)
    return result

def int_to_binary_vector(n, length=None):
    """
    Convert integer back to binary vector (LSB first convention)
    
    Args:
        n: integer representation of polynomial
        length: desired length of output vector (pad with zeros if needed)
    
    Returns:
        binary vector [a0, a1, a2, ...] where polynomial is a0 + a1*x + a2*x^2 + ...
    """
    if n == 0:
        return [0] * (length if length else 1)
    
    # Get binary representation
    binary_str = bin(n)[2:]  # Remove '0b' prefix
    
    # Convert to list (reverse for LSB first)
    binary_vector = [int(bit) for bit in reversed(binary_str)]
    
    # Pad with zeros if needed
    if length and len(binary_vector) < length:
        binary_vector.extend([0] * (length - len(binary_vector)))
    
    return binary_vector

def binary_poly_divide(dividend, divisor):
    """
    Polynomial division in GF(2) - copied from your existing function
    """
    def degree(n):
        return n.bit_length() - 1

    quotient = 0
    remainder = dividend

    while degree(remainder) >= degree(divisor):
        shift = degree(remainder) - degree(divisor)
        quotient ^= 1 << shift
        remainder ^= divisor << shift

    return quotient, remainder

def polynomial_to_string(binary_vector):
    """Convert binary vector to polynomial string representation"""
    terms = []
    for i, coeff in enumerate(binary_vector):
        if coeff == 1:
            if i == 0:
                terms.append("1")
            elif i == 1:
                terms.append("x")
            else:
                terms.append(f"x^{i}")
    
    return " + ".join(terms) if terms else "0"

def polynomial_division(variants, q0_poly, q1_poly):
    """    
    Args:
        variants: dictionary with four unmasked variants from step 2
        q0_poly: q0 polynomial coefficients [a0, a1, a2, ...]
        q1_poly: q1 polynomial coefficients [b0, b1, b2, ...]
    
    Returns:
        quotients: dictionary with four quotient results
    """
    # Convert q polynomials to integers for division
    q0_int = binary_vector_to_int(q0_poly)
    q1_int = binary_vector_to_int(q1_poly)
    
    quotients = {}
    # Variant 0: (c̃(x) - 0(x))_0 / q0(x)
    var0_int = binary_vector_to_int(variants['even_minus_zeros'])
    quot0, rem0 = binary_poly_divide(var0_int, q0_int)
    quotients['d0_0'] = int_to_binary_vector(quot0, 8)  # 8 bits as in paper
    
    # Variant 1: (c̃(x) - 1(x))_0 / q0(x)
    var1_int = binary_vector_to_int(variants['even_minus_ones'])
    quot1, rem1 = binary_poly_divide(var1_int, q0_int)
    quotients['d1_0'] = int_to_binary_vector(quot1, 8)
    
    # Variant 2: (c̃(x) - 0(x))_1 / q1(x)
    var2_int = binary_vector_to_int(variants['odd_minus_zeros'])
    quot2, rem2 = binary_poly_divide(var2_int, q1_int)
    quotients['d0_1'] = int_to_binary_vector(quot2, 8)
    
    # Variant 3: (c̃(x) - 1(x))_1 / q1(x)
    var3_int = binary_vector_to_int(variants['odd_minus_ones'])
    quot3, rem3 = binary_poly_divide(var3_int, q1_int)
    quotients['d1_1'] = int_to_binary_vector(quot3, 8)
    
    return quotients

def interleave_two_vectors(vec1, vec2):
    """
    Interleave two vectors: vec1 ⋏ vec2
    Result: [vec1[0], vec2[0], vec1[1], vec2[1], vec1[2], vec2[2], ...]
    
    Args:
        vec1: first vector
        vec2: second vector
    
    Returns:
        interleaved vector
    """
    
    # Ensure both vectors have the same length
    min_len = min(len(vec1), len(vec2))
    max_len = max(len(vec1), len(vec2))
    
    # Pad shorter vector with zeros if needed
    if len(vec1) < max_len:
        vec1 = list(vec1) + [0] * (max_len - len(vec1))
    if len(vec2) < max_len:
        vec2 = list(vec2) + [0] * (max_len - len(vec2))
    
    # Interleave: [a[0], b[0], a[1], b[1], ...]
    result = []
    for i in range(max_len):
        result.append(vec1[i])
        result.append(vec2[i])
    
    return result

def quotient_interleaving(quotients):
    """    
    Args:
        quotients: dict with keys 'd0_0', 'd1_0', 'd0_1', 'd1_1' from step 3
    
    Returns:
        d_vectors: dict with four interleaved d vectors
    """    
    # Extract the quotients
    d0_0 = quotients['d0_0']  # (d0)_0
    d1_0 = quotients['d1_0']  # (d1)_0  
    d0_1 = quotients['d0_1']  # (d0)_1
    d1_1 = quotients['d1_1']  # (d1)_1
    
    print(f"(d₀)₀ = {d0_0}")
    print(f"(d₁)₀ = {d1_0}")
    print(f"(d₀)₁ = {d0_1}")
    print(f"(d₁)₁ = {d1_1}")
    print()
    
    # Create the four interleaved combinations
    d_vectors = {}
    
    print("Interleaving quotients to create d vectors:\n")
    # d0 = (d0)_0 ⋏ (d0)_1
    d_vectors['d0'] = interleave_two_vectors(d0_0, d0_1)
    print(f"d₀ = (d₀)₀⋏(d₀)₁ = {d_vectors['d0']}")
    
    # d1 = (d0)_0 ⋏ (d1)_1
    d_vectors['d1'] = interleave_two_vectors(d0_0, d1_1)
    print(f"d₁ = (d₀)₀⋏(d1)₁ = {d_vectors['d1']}")
    
    # d2 = (d1)_0 ⋏ (d0)_1
    d_vectors['d2'] = interleave_two_vectors(d1_0, d0_1)
    print(f"d₂ = (d₁)₀⋏(d₀)₁ = {d_vectors['d2']}")
    
    # d3 = (d1)_0 ⋏ (d1)_1
    d_vectors['d3'] = interleave_two_vectors(d1_0, d1_1)
    print(f"d₃ = (d₁)₀⋏(d₁)₁ = {d_vectors['d3']}")
    
    return d_vectors

def find_best_d_vector(d_vectors, p0, p1, K, register_size):
    print("\n" + "=" * 80)
    print("\nViterbi Decoding:")
    min_weight = float('inf')  # Start with infinite weight
    best_index = -1
    best_path = None
    best_d = None
    
    # Test each d vector 
    for key, d in d_vectors.items():
        # Extract the index from the key (e.g., 'd0' -> 0)
        i = int(key[1])  # Get the number after 'd'        
        try:
            # Create trellis and find minimal path
            trellis = create_trellis(p0, p1, K, register_size, d)
            weight, path = find_min_path(trellis)
            print(f"  d{i}: ,Minimal weight: {weight}, Best path: {path}")
            
            # Check if this is the new minimum
            if weight < min_weight:
                min_weight = weight
                best_index = i
                best_path = path
                best_d = d
                
        except Exception as e:
            print(f"  Error processing {key}: {e}")
            continue
    
    if best_index == -1:
        print(f"\n❌ No valid d vector found!")
        return None
    
    print(f"🏆 WINNER: d{best_index} with minimal weight {min_weight}\n")
    
    return {
        'index': best_index,
        'd_vector': best_d, 
        'weight': min_weight,
        'path': best_path
    }

def parallel_viterbi_decoding(d_vectors, p0, p1, K, register_size, S_inv):    
    # Find the d vector with minimal weight
    best = find_best_d_vector(d_vectors, p0, p1, K, register_size)
    
    if best['path'] is None:
        print("❌ No valid decoding found!")
        return None
    
    print("=" * 80)
    print("\nDecoding the msg:\n")    
    # Extract information bits (remove termination bits)
    info_word = best['path']
    print(f"Raw info word: {info_word}")
    
    # Remove termination bits (last register_size zeros)
    if len(info_word) > (register_size-1):
        info_bits = info_word[:-(register_size-1)]
    else:
        info_bits = info_word
        
    print(f"Info bits (after removing {register_size-1} termination bits): {info_bits}")
    
    # Ensure exactly K bits
    if len(info_bits) > K:
        info_bits = info_bits[:K]
    elif len(info_bits) < K:
        info_bits.extend([0] * (K - len(info_bits)))
            
    # Apply inverse transformation: m = (m̂S) × S^(-1)
    transformed_plaintext = np.array(info_bits, dtype=int).reshape(1, K)
    original_plaintext = (transformed_plaintext @ S_inv) % 2
    
    print(f"Transformed plaintext (m̂S): {transformed_plaintext.flatten()}")
    print(f"Original plaintext (m): {original_plaintext.flatten()}")
    
    return original_plaintext

def ask_user_choice():
    """
    Ask user if they want to use default values or input custom values
    
    Returns:
        bool: True if user wants defaults, False if user wants to input custom values
    """
    print("=" * 80)
    print("    HIGH-MEMORY MASKED CONVOLUTIONAL CODES FOR POST-QUANTUM CRYPTOGRAPHY")
    print("=" * 80)
    print()
    print("Choose your input method:")
    print("1. Use default values (quick test)")
    print("2. Enter custom values (interactive)")
    print()
    
    while True:
        choice = input("Enter your choice (1 or 2): ").strip()
        
        if choice == '1':
            print("\n✓ Using default values for quick testing...")
            print("-" * 40)
            return True
        elif choice == '2':
            print("\n✓ Using interactive input mode...")
            print("-" * 40)
            return False
        else:
            print("❌ Invalid choice. Please enter 1 or 2.")
            
def display_summary(params):
    """
    Display a summary of the input parameters
    
    Args:
        params: dictionary containing all input parameters
    """
    print("\n" + "=" * 60)
    print("                 INPUT PARAMETER SUMMARY")
    print("=" * 60)
    print(f"Message:          {params['msg']}")
    print(f"Register size:    {params['register_size']}")
    print(f"Default rate:     1/2 (n = 2)")
    print(f"p₀(x) = {bin_to_poly(binary_vector_to_int(params['p0']))}")
    print(f"p₁(x) = {bin_to_poly(binary_vector_to_int(params['p1']))}")
    print(f"q₀(x) = {bin_to_poly(binary_vector_to_int(params['q0']))}")
    print(f"q₁(x) = {bin_to_poly(binary_vector_to_int(params['q1']))}")
    print("=" * 60)
    print()
    
def get_input_from_user(use_defaults=False):
    def get_poly_input(name):
        """Helper function to get polynomial input"""
        return list(map(int, input(f"Enter {name} polynomial as space-separated bits (e.g., 1 0 1): ").split()))
    
    if use_defaults:
        print("Using default values...")
        
        # Default values
        msg = np.array([1, 1, 1, 0, 0, 1])
        register_size = 3
        p0 = [1, 0, 1]  # p0(x) = 1 + x^2
        p1 = [1, 1, 1]  # p1(x) = 1 + x + x^2
        q0 = [1, 0, 0, 0, 0, 0, 0, 1]  # q0(x) = 1 + x^7
        q1 = [0, 0, 0, 0, 0, 0, 0, 1]  # q1(x) = x^7
        
    else:
        print("=== Convolutional Cryptosystem Input ===")
        print()
        
        # Get binary message
        msg_input = input("Enter a binary message (e.g., 1 1 1 0 0 1): ")
        msg = np.array(list(map(int, msg_input.split())))
        
        # Get register size
        register_size = int(input("Enter memory length of the convolutional code (register size) (e.g., 3): "))
        
        # Get polynomials
        print("\nEnter convolutional code polynomials:")
        p0 = get_poly_input("p0")
        p1 = get_poly_input("p1")
        
        print("\nEnter high-memory polynomials:")
        q0 = get_poly_input("q0")
        q1 = get_poly_input("q1")
    
    # Create polynomial lists
    p_polynomials = [p0, p1]
    q_polynomials = [q0, q1]
    
    # Return all parameters as a dictionary
    return {
        'msg': msg,
        'register_size': register_size,
        'p0': p0,
        'p1': p1,
        'q0': q0,
        'q1': q1,
        'p_polynomials': p_polynomials,
        'q_polynomials': q_polynomials
    }
    
def disp_matrix(matrix, name):
    print(f"\n{name}:")
    for row in matrix:
        print(' '.join(map(str, row)))
    print()            
    print("=" * 80)  

def generate_permutation_matrix(use_defaults,N):
    # generate permutation vector
    if use_defaults:
        perm = [13, 24, 8, 17, 29, 7, 20, 0, 9, 28, 4, 25, 2, 10, 22, 27, 14, 1, 6, 11, 19, 5, 16, 3, 26, 15, 23, 12, 21, 18]
    else:
        perm = np.random.permutation(N)
        
    # generate permutation matrix R 
    R = np.zeros((N, N), dtype=int)
    for new_col, old_col in enumerate(perm):
        R[old_col, new_col] = 1
        
    # generate inverse permutation matrix R_inv
    R_inv_vec = [0] * N
    for new_pos, old_pos in enumerate(perm):
        R_inv_vec[old_pos] = new_pos
        
    # print
    perm_1based = [col + 1 for col in perm]
    print(f"\nPermutation Vector:")
    print(f"[{' '.join(map(str, perm_1based))}]")
    disp_matrix(R, "Permutation Matrix R")
    
    return R, R_inv_vec

def get_error_vector(codeword_length, use_defaults):
    if use_defaults:
        return [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    else:
        while True:
            try:
                num_errors = int(input(f"Enter number of errors (0 to {codeword_length}): "))
                if 0 <= num_errors <= codeword_length:
                    break
                else:
                    print(f"❌ Number must be between 0 and {codeword_length}")
            except ValueError:
                print("❌ Please enter a valid integer")
        error_vector = np.zeros(codeword_length, dtype=int)
        if num_errors == 0:
            print("✓ No errors - returning zero vector")
            return np.zeros(codeword_length, dtype=int)
        error_positions = random.sample(range(codeword_length), num_errors)
        for pos in error_positions:
            error_vector[pos] = 1
        
        return error_vector
    
def check_recovery_with_crc(original_message, recovered_message, original_msg_length):
    """
    Check if recovery was successful, handling both CRC and non-CRC cases
    
    Args:
        original_message: The message that was encoded
        recovered_message: The message that was recovered
        use_crc: Whether CRC was used
        original_msg_length: Length of the original message before CRC
    
    Returns:
        dict: Recovery results and analysis
    """
    print("=" * 70)
    print("                    RECOVERY ANALYSIS")
    print("=" * 70)
    
    results = {}
    
    recovered_flat = recovered_message.flatten()
    
    # Extract original 3-bit message from both
    original_3bit = np.array(original_message[:original_msg_length])
    recovered_3bit = np.array(recovered_flat[:original_msg_length])
    
    # Check if original message was recovered
    original_match = np.array_equal(original_3bit, recovered_3bit)
    
    print(f"\nCRC Analysis:")
    print(f"Original 3-bit:        {original_3bit}")
    print(f"Recovered 3-bit:       {recovered_3bit}")
    print(f"Original match:        {'✅ YES' if original_match else '❌ NO'}")
            
    results.update({
        'original_3bit_recovered': original_match,
        'original_3bit': original_3bit,
        'recovered_3bit': recovered_3bit
    })
    
    # Overall success for CRC case
    overall_success = original_match 
    
    print(f"\nOverall Result:        {'🎉 SUCCESS' if overall_success else '❗ FAILURE'}")
    
    results.update({
        'success': overall_success,
    })
    
    print("=" * 70)
    return results

def main():
    # Ask user if they want to use defaults or custom input
    use_defaults = ask_user_choice()
    # Get input parameters based on user choice
    params = get_input_from_user(use_defaults=use_defaults)
    # Display summary of inputs
    display_summary(params)
    # Extract parameters from dictionary
    raw_msg = params['msg']
    register_size = params['register_size'] 
    p0, p1 = params['p0'], params['p1']
    q0, q1 = params['q0'], params['q1']
    p_polynomials = params['p_polynomials']
    q_polynomials = params['q_polynomials']
    if use_defaults:
        msg = raw_msg
    else:
        msg = apply_crc_to_message(raw_msg)  # You'd need to implement this
    
    K = len(msg) # Number of information bits (rows in Gp and Gpq)
    p_deg = poly_degree(p0)  # Degree of p0
    q_deg = poly_degree(q0)  # Degree of q0
    n = 2  # Number of output bits per input bit (rate 1/2)
    N = n*(K+p_deg+q_deg) # Total number of columns in Gp and Gpq
    
    q_coefficient_matrices = extract_polynomial_coefficients(q0, q1)
    Gp = generate_Gp(p0, p1, K, n)    
    Gpq = generate_Gpq(p_polynomials, q_polynomials, K)
    G_tilde = generate_masking_matrix(K, N, n)
    Gpq_plus_G_tilde = (Gpq + G_tilde) % 2
    disp_matrix(Gpq_plus_G_tilde, "Matrix - Gpq + G̃")
    S, S_inv = generate_random_nonsingular_binary_matrix(K)
    S_times_Gpq_plus_G_tilde = (S @ Gpq_plus_G_tilde) % 2
    disp_matrix(Gpq_plus_G_tilde, "Matrix - S(Gpq + G̃)")
    R, R_inv_vec = generate_permutation_matrix(use_defaults, N)
    G = S_times_Gpq_plus_G_tilde @ R
    disp_matrix(G, "Permuted Matrix - S(Gpq + G̃)R")
    # the codeword c of the MCC corresponding to G
    c = (msg @ G) % 2
    print(f"\nThe codeword c of the MCC corresponding to G:")
    print(f"c = mG = [{' '.join(map(str, c))}]\n")
    e = get_error_vector(G.shape[1], use_defaults)
    c_e = (c + e) % 2
    print(f"cₑ = c + e = [{' '.join(map(str, c_e))}]\n")

    # decryption
    # step 1: inverse permutation
    c_tilde = [0] * N
    for new_pos, old_pos in enumerate(R_inv_vec):
        c_tilde[new_pos] = c_e[old_pos]    
    print("c̃ = CₑR⁻¹ =  [" + " ".join(map(str, c_tilde)) + "]")
    c_tilde = np.array(c_tilde, dtype=int)

    # step 2: unmasking    
    # Extract even and odd positioned bits
    even_bits = c_tilde[0::2]  # positions 0, 2, 4, 6, ...
    odd_bits = c_tilde[1::2]   # positions 1, 3, 5, 7, ...
    print(f"Even positions (0,2,4,...): [" + " ".join(map(str, even_bits)) + "]")
    print(f"Odd positions  (1,3,5,...): [" + " ".join(map(str, odd_bits)) + "]\n")
    print("Deinterleaving c̃ to its polynomials constitutes and computing the four possible unmasked variants:\n")
    variants = compute_unmasked_variants(even_bits, odd_bits)
    
    # step 3: inverting the high-memory polynomial multiplication
    print("Inverting the High-Memory Polynomial Multiplication:\n")
    quotients = polynomial_division(variants, q0, q1)

    # step 4: quotient interleaving
    d_vectors = quotient_interleaving(quotients)
    
    # step 5+6: parallel Viterbi decoding and plaintext recovery
    original_message = parallel_viterbi_decoding(
        d_vectors, p0, p1, K, register_size, S_inv
    )
    if original_message is not None:
        if (np.array_equal(msg, original_message.flatten())):
            print(f"\n🎉 SUCCESS!")
        else: 
            print(f"\n❗️ Decryption mismatch!")
        print(f"Original message:  {msg}")
        print(f"Recovered message: {original_message.flatten()}\n")
    else:
        print("❌ Decryption failed!")
    
    results = check_recovery_with_crc(original_message.flatten(), raw_msg, len(raw_msg))


    
if __name__ == "__main__":
    main()
