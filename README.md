# High-Memory Masked Convolutional Codes

A post-quantum cryptography implementation based on convolutional codes.

## Overview

This project implements a novel post-quantum encryption scheme using high-memory masked convolutional codes. Unlike traditional code-based methods that rely on block codes with fixed parameters, this approach:

- Supports arbitrary plaintext lengths
- Uses convolutional codes with strong error-correction capabilities
- Employs a dual-layer error mechanism for enhanced security
- Scales efficiently with linear decryption complexity

## Features

- **Stronger Security**: Offers significantly enhanced cryptanalytic resistance compared to Classic McEliece
- **Flexible Code Selection**: Supports various convolutional codes to meet specific security needs
- **Scalable Key Length**: Adaptable to different security requirements
- **Efficient Hardware Implementation**: Uses the Viterbi algorithm for decoding

## Getting Started

```python
# Generate a public key
G = generate_public_key()

# Define a message (plaintext)
m = np.array([1, 1, 1, 0, 0, 1])

# Encrypt the message
encrypted_msg = encrypt_msg(G, m)

# Decrypt the message
decrypted_msg = decrypt_msg(encrypted_msg)
```

## Requirements

- Python 3.6+
- NumPy
- SciPy

## Installation

```bash
git clone https://github.com/yourusername/high-memory-cc-crypto.git
cd high-memory-cc-crypto
pip install -r requirements.txt
```

## License

MIT

## References

Based on "High-Memory Masked Convolutional Codes for Post-Quantum Cryptography" by Meir Ariel.
