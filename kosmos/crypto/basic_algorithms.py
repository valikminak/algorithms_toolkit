import hashlib
import os
import hmac
import base64
import math
from typing import Tuple, Optional, List
import secrets
import random


def caesar_cipher(message: str, shift: int, encrypt: bool = True) -> str:
    """
    Caesar cipher encryption and decryption.

    Args:
        message: The message to encrypt or decrypt
        shift: The number of positions to shift each character
        encrypt: True for encryption, False for decryption

    Returns:
        Encrypted or decrypted message
    """
    # Adjust shift for decryption
    if not encrypt:
        shift = -shift

    result = ""

    for char in message:
        if char.isalpha():
            # Determine the ASCII offset based on case
            ascii_offset = ord('A') if char.isupper() else ord('a')

            # Apply the shift with modulo to wrap around the alphabet
            shifted = (ord(char) - ascii_offset + shift) % 26 + ascii_offset
            result += chr(shifted)
        else:
            # Keep non-alphabetic characters unchanged
            result += char

    return result


def vigenere_cipher(message: str, key: str, encrypt: bool = True) -> str:
    """
    Vigenère cipher encryption and decryption.

    Args:
        message: The message to encrypt or decrypt
        key: The encryption key
        encrypt: True for encryption, False for decryption

    Returns:
        Encrypted or decrypted message
    """
    result = ""
    key = key.upper()
    key_length = len(key)
    key_as_int = [ord(k) - ord('A') for k in key]

    i = 0
    for char in message:
        if char.isalpha():
            # Determine the ASCII offset based on case
            is_upper = char.isupper()
            ascii_offset = ord('A') if is_upper else ord('a')

            # Calculate the shift for this character
            key_index = i % key_length
            key_shift = key_as_int[key_index]

            # Apply the shift (add for encrypt, subtract for decrypt)
            if encrypt:
                shifted = (ord(char) - ascii_offset + key_shift) % 26 + ascii_offset
            else:
                shifted = (ord(char) - ascii_offset - key_shift) % 26 + ascii_offset

            result += chr(shifted)
            i += 1
        else:
            # Keep non-alphabetic characters unchanged
            result += char

    return result


def transposition_cipher(message: str, key: int, encrypt: bool = True) -> str:
    """
    Columnar transposition cipher encryption and decryption.

    Args:
        message: The message to encrypt or decrypt
        key: Number of columns (for encryption) or rows (for decryption)
        encrypt: True for encryption, False for decryption

    Returns:
        Encrypted or decrypted message
    """
    if key <= 0:
        return message

    if encrypt:
        # Add padding if necessary
        padding_length = (key - len(message) % key) % key
        padded_message = message + ' ' * padding_length

        # Arrange the message in a grid with 'key' columns
        grid = [padded_message[i:i + key] for i in range(0, len(padded_message), key)]

        # Read the message column by column
        result = ''
        for col in range(key):
            for row in range(len(grid)):
                result += grid[row][col]

        return result
    else:
        # Calculate dimensions for the grid
        length = len(message)
        rows = math.ceil(length / key)

        # Create an empty grid
        grid = [[' ' for _ in range(key)] for _ in range(rows)]

        # Calculate the number of complete columns
        full_cols = length % rows
        if full_cols == 0:
            full_cols = rows

        # Fill the grid column by column
        index = 0
        for col in range(key):
            col_height = rows if col < full_cols else rows - 1
            for row in range(col_height):
                if index < length:
                    grid[row][col] = message[index]
                    index += 1

        # Read the message row by row
        result = ''
        for row in grid:
            result += ''.join(row)

        return result


def xor_cipher(message: str, key: str) -> str:
    """
    XOR cipher (symmetric encryption).

    Args:
        message: The message to encrypt or decrypt
        key: The encryption key

    Returns:
        Encrypted or decrypted message
    """
    # Convert message and key to bytes if they are strings
    if isinstance(message, str):
        message = message.encode('utf-8')
    if isinstance(key, str):
        key = key.encode('utf-8')

    # Create a repeating key of the same length as the message
    key_repeated = key * (len(message) // len(key) + 1)
    key_bytes = key_repeated[:len(message)]

    # XOR each byte in the message with the corresponding byte in the key
    result_bytes = bytes([message[i] ^ key_bytes[i] for i in range(len(message))])

    # Try to decode as UTF-8, fallback to base64 if not valid UTF-8
    try:
        return result_bytes.decode('utf-8')
    except UnicodeDecodeError:
        return base64.b64encode(result_bytes).decode('ascii')


def diffiehellman_key_exchange(prime: int, generator: int) -> Tuple[int, int, int]:
    """
    Simulate Diffie-Hellman key exchange protocol.

    Args:
        prime: A large prime number (p)
        generator: A primitive root modulo p (g)

    Returns:
        Tuple of (Alice's private key, Bob's private key, shared secret)
    """
    # Alice generates a private key
    alice_private = random.randint(2, prime - 2)

    # Alice computes her public key
    alice_public = pow(generator, alice_private, prime)

    # Bob generates a private key
    bob_private = random.randint(2, prime - 2)

    # Bob computes his public key
    bob_public = pow(generator, bob_private, prime)

    # Alice computes the shared secret
    alice_shared_secret = pow(bob_public, alice_private, prime)

    # Bob computes the shared secret
    bob_shared_secret = pow(alice_public, bob_private, prime)

    # Verify that both parties computed the same shared secret
    assert alice_shared_secret == bob_shared_secret, "Key exchange failed"

    return alice_private, bob_private, alice_shared_secret


def sha256_hash(message: str) -> str:
    """
    Calculate SHA-256 hash of a message.

    Args:
        message: The message to hash

    Returns:
        Hexadecimal digest of the SHA-256 hash
    """
    sha256 = hashlib.sha256()
    sha256.update(message.encode('utf-8'))
    return sha256.hexdigest()


def hmac_sha256(key: str, message: str) -> str:
    """
    Calculate HMAC-SHA256 of a message.

    Args:
        key: The secret key
        message: The message to authenticate

    Returns:
        Hexadecimal digest of the HMAC-SHA256
    """
    return hmac.new(
        key.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()


def generate_random_key(length: int = 32) -> str:
    """
    Generate a secure random key.

    Args:
        length: Length of the key in bytes

    Returns:
        Base64-encoded random key
    """
    random_bytes = secrets.token_bytes(length)
    return base64.b64encode(random_bytes).decode('ascii')


class RSA:
    """
    RSA encryption implementation (simplified).

    Warning: This is a simplified implementation for educational purposes only.
    For real-world cryptography, use established libraries.
    """

    @staticmethod
    def generate_keypair(bits: int = 1024) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Generate an RSA key pair.

        Args:
            bits: Key size in bits

        Returns:
            Tuple of ((e, n), (d, n)) representing public and private keys
        """

        def is_prime(n: int, k: int = 5) -> bool:
            """Miller-Rabin primality test."""
            if n <= 1:
                return False
            if n <= 3:
                return True
            if n % 2 == 0:
                return False

            # Write n-1 as 2^r * d
            r, d = 0, n - 1
            while d % 2 == 0:
                r += 1
                d //= 2

            # Witness loop
            for _ in range(k):
                a = random.randint(2, n - 2)
                x = pow(a, d, n)

                if x == 1 or x == n - 1:
                    continue

                for _ in range(r - 1):
                    x = pow(x, 2, n)
                    if x == n - 1:
                        break
                else:
                    return False

            return True

        def generate_prime(bits: int) -> int:
            """Generate a prime number with the specified number of bits."""
            while True:
                p = random.getrandbits(bits)
                # Ensure the number has the correct number of bits
                p |= (1 << bits - 1) | 1
                if is_prime(p):
                    return p

        def gcd(a: int, b: int) -> int:
            """Calculate the greatest common divisor of a and b."""
            while b:
                a, b = b, a % b
            return a

        def mod_inverse(e: int, phi: int) -> int:
            """Calculate the modular multiplicative inverse of e (mod phi)."""

            def extended_gcd(a, b):
                if a == 0:
                    return b, 0, 1
                else:
                    gcd, x, y = extended_gcd(b % a, a)
                    return gcd, y - (b // a) * x, x

            gcd, x, y = extended_gcd(e, phi)
            if gcd != 1:
                raise ValueError("Modular inverse does not exist")
            else:
                return x % phi

        # Generate two large prime numbers
        p = generate_prime(bits // 2)
        q = generate_prime(bits // 2)

        # Calculate n and phi(n)
        n = p * q
        phi = (p - 1) * (q - 1)

        # Choose e
        e = 65537  # Common choice for e

        # If e is not coprime to phi, find another e
        while gcd(e, phi) != 1:
            e = random.randrange(2, phi)

        # Calculate d
        d = mod_inverse(e, phi)

        # Return the key pair
        public_key = (e, n)
        private_key = (d, n)

        return public_key, private_key

    @staticmethod
    def encrypt(message: int, public_key: Tuple[int, int]) -> int:
        """
        Encrypt a message using the RSA algorithm.

        Args:
            message: Integer message to encrypt
            public_key: Public key tuple (e, n)

        Returns:
            Encrypted message
        """
        e, n = public_key
        if message < 0 or message >= n:
            raise ValueError("Message must be between 0 and n-1")

        return pow(message, e, n)

    @staticmethod
    def decrypt(ciphertext: int, private_key: Tuple[int, int]) -> int:
        """
        Decrypt a message using the RSA algorithm.

        Args:
            ciphertext: Encrypted message
            private_key: Private key tuple (d, n)

        Returns:
            Decrypted message
        """
        d, n = private_key
        return pow(ciphertext, d, n)

    @staticmethod
    def encrypt_string(message: str, public_key: Tuple[int, int]) -> List[int]:
        """
        Encrypt a string message using RSA.

        Args:
            message: String message to encrypt
            public_key: Public key tuple (e, n)

        Returns:
            List of encrypted blocks
        """
        # Convert the message to bytes
        message_bytes = message.encode('utf-8')

        # Calculate the maximum block size (in bytes)
        # n.bit_length() // 8 - 1 ensures we stay below n
        e, n = public_key
        block_size = n.bit_length() // 16  # Conservative block size

        # Split the message into blocks and encrypt each block
        ciphertext_blocks = []
        for i in range(0, len(message_bytes), block_size):
            block = message_bytes[i:i + block_size]
            # Convert the block to an integer
            block_int = int.from_bytes(block, byteorder='big')
            # Encrypt the integer
            encrypted_block = RSA.encrypt(block_int, public_key)
            ciphertext_blocks.append(encrypted_block)

        return ciphertext_blocks

    @staticmethod
    def decrypt_string(ciphertext_blocks: List[int], private_key: Tuple[int, int],
                       block_size: Optional[int] = None) -> str:
        """
        Decrypt a string message using RSA.

        Args:
            ciphertext_blocks: List of encrypted blocks
            private_key: Private key tuple (d, n)
            block_size: Original block size in bytes (if known)

        Returns:
            Decrypted message
        """
        d, n = private_key

        # If block_size is not provided, estimate it
        if block_size is None:
            block_size = n.bit_length() // 16  # Conservative block size

        # Decrypt each block and concatenate
        decrypted_bytes = bytearray()
        for block in ciphertext_blocks:
            decrypted_block = RSA.decrypt(block, private_key)

            # Convert the integer back to bytes
            block_bytes = decrypted_block.to_bytes(
                (decrypted_block.bit_length() + 7) // 8, byteorder='big'
            )

            decrypted_bytes.extend(block_bytes)

        # Convert bytes back to string
        try:
            return decrypted_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # In case of decoding issues, return the raw bytes as a string
            return str(decrypted_bytes)


class AES:
    """
    Simple AES-inspired block cipher implementation.

    Warning: This is a simplified implementation for educational purposes only.
    For real-world cryptography, use established libraries.
    """

    # S-box for SubBytes operation (simplified)
    SBOX = [
        0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
        0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
        0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
        0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
        0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
        0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
        0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
        0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
        0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
        0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
        0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
        0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
        0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
        0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
        0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
        0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
    ]

    # Inverse S-box for InvSubBytes operation (simplified)
    INV_SBOX = [
        0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
        0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
        0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
        0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
        0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
        0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
        0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
        0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
        0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
        0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
        0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
        0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
        0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
        0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
        0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
        0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
    ]

    def __init__(self, key: bytes):
        """
        Initialize AES cipher with a key.

        Args:
            key: Encryption key (16, 24, or 32 bytes for AES-128/192/256)
        """
        if len(key) not in (16, 24, 32):
            raise ValueError("Key must be 16, 24, or 32 bytes")

        self.key = key
        self.rounds = {16: 10, 24: 12, 32: 14}[len(key)]
        self.key_schedule = self._key_expansion(key)

    def _key_expansion(self, key: bytes) -> List[List[int]]:
        """
        Expand the key into a key schedule.

        Args:
            key: The original encryption key

        Returns:
            Key schedule (list of round keys)
        """
        # Simplified key expansion for educational purposes
        # In a real implementation, this would follow the AES key schedule algorithm
        key_bytes = list(key)
        key_schedule = []

        # Split the key into 16-byte blocks
        for i in range(0, len(key_bytes), 16):
            round_key = key_bytes[i:i + 16]
            key_schedule.append(round_key)

        # Generate additional round keys
        while len(key_schedule) <= self.rounds:
            prev_key = key_schedule[-1]
            new_key = []

            # Simple key derivation (not the actual AES key schedule)
            for j in range(16):
                # Mix with S-box
                new_key.append(self.SBOX[(prev_key[j] + j) % 256])

            key_schedule.append(new_key)

        return key_schedule[:self.rounds + 1]

    def _sub_bytes(self, state: List[int]) -> List[int]:
        """
        SubBytes transformation - substitute each byte with its S-box value.

        Args:
            state: Current state array

        Returns:
            Transformed state
        """
        return [self.SBOX[b] for b in state]

    def _inv_sub_bytes(self, state: List[int]) -> List[int]:
        """
        InvSubBytes transformation - substitute each byte with its inverse S-box value.

        Args:
            state: Current state array

        Returns:
            Transformed state
        """
        return [self.INV_SBOX[b] for b in state]

    def _shift_rows(self, state: List[int]) -> List[int]:
        """
        ShiftRows transformation - cyclically shift rows of the state.

        Args:
            state: Current state array

        Returns:
            Transformed state
        """
        # For simplicity, we're representing the state as a flat list
        # In a real implementation, this would work on a 4x4 matrix

        # First row: no shift
        # Second row: shift left by 1
        # Third row: shift left by 2
        # Fourth row: shift left by 3
        return [
            # Row 0 (no shift)
            state[0], state[4], state[8], state[12],
            # Row 1 (shift by 1)
            state[5], state[9], state[13], state[1],
            # Row 2 (shift by 2)
            state[10], state[14], state[2], state[6],
            # Row 3 (shift by 3)
            state[15], state[3], state[7], state[11]
        ]

    def _inv_shift_rows(self, state: List[int]) -> List[int]:
        """
        InvShiftRows transformation - inverse of ShiftRows.

        Args:
            state: Current state array

        Returns:
            Transformed state
        """
        return [
            # Row 0 (no shift)
            state[0], state[4], state[8], state[12],
            # Row 1 (shift right by 1)
            state[7], state[11], state[15], state[3],
            # Row 2 (shift right by 2)
            state[14], state[2], state[6], state[10],
            # Row 3 (shift right by 3)
            state[13], state[1], state[5], state[9]
        ]

    def _mix_columns(self, state: List[int]) -> List[int]:
        """
        MixColumns transformation - mix data within each column for diffusion.

        Args:
            state: Current state array

        Returns:
            Transformed state
        """

        # Simplified MixColumns for educational purposes
        # In a real implementation, this involves Galois field multiplication

        def mix_column(column):
            a, b, c, d = column
            return [
                (a << 1) ^ b ^ c ^ d,
                a ^ (b << 1) ^ c ^ d,
                a ^ b ^ (c << 1) ^ d,
                a ^ b ^ c ^ (d << 1)
            ]

        result = []
        for i in range(0, 16, 4):
            column = [state[i], state[i + 1], state[i + 2], state[i + 3]]
            mixed = mix_column(column)
            result.extend(mixed)

        # Apply modulo 256 to ensure byte values
        return [x % 256 for x in result]

    def _inv_mix_columns(self, state: List[int]) -> List[int]:
        """
        InvMixColumns transformation - inverse of MixColumns.

        Args:
            state: Current state array

        Returns:
            Transformed state
        """

        # Simplified InvMixColumns for educational purposes
        # This is a very simplified version that doesn't accurately reflect
        # the mathematical inverse of MixColumns

        def inv_mix_column(column):
            a, b, c, d = column
            return [
                (a << 2) ^ (b << 1) ^ c ^ d,
                a ^ (b << 2) ^ (c << 1) ^ d,
                a ^ b ^ (c << 2) ^ (d << 1),
                (a << 1) ^ b ^ c ^ (d << 2)
            ]

        result = []
        for i in range(0, 16, 4):
            column = [state[i], state[i + 1], state[i + 2], state[i + 3]]
            mixed = inv_mix_column(column)
            result.extend(mixed)

        # Apply modulo 256 to ensure byte values
        return [x % 256 for x in result]

    def _add_round_key(self, state: List[int], round_key: List[int]) -> List[int]:
        """
        AddRoundKey transformation - XOR the state with the round key.

        Args:
            state: Current state array
            round_key: Current round key

        Returns:
            Transformed state
        """
        return [s ^ k for s, k in zip(state, round_key)]

    def encrypt_block(self, block: bytes) -> bytes:
        """
        Encrypt a single 16-byte block.

        Args:
            block: 16-byte plaintext block

        Returns:
            16-byte encrypted block
        """
        if len(block) != 16:
            raise ValueError("Block size must be 16 bytes")

        # Convert block to state array
        state = list(block)

        # Initial round key addition
        state = self._add_round_key(state, self.key_schedule[0])

        # Main encryption rounds
        for i in range(1, self.rounds):
            state = self._sub_bytes(state)
            state = self._shift_rows(state)
            state = self._mix_columns(state)
            state = self._add_round_key(state, self.key_schedule[i])

        # Final round (no MixColumns)
        state = self._sub_bytes(state)
        state = self._shift_rows(state)
        state = self._add_round_key(state, self.key_schedule[self.rounds])

        # Convert state back to bytes
        return bytes(state)

    def decrypt_block(self, block: bytes) -> bytes:
        """
        Decrypt a single 16-byte block.

        Args:
            block: 16-byte ciphertext block

        Returns:
            16-byte decrypted block
        """
        if len(block) != 16:
            raise ValueError("Block size must be 16 bytes")

        # Convert block to state array
        state = list(block)

        # Initial round (reverse of final encryption round)
        state = self._add_round_key(state, self.key_schedule[self.rounds])
        state = self._inv_shift_rows(state)
        state = self._inv_sub_bytes(state)

        # Main decryption rounds
        for i in range(self.rounds - 1, 0, -1):
            state = self._add_round_key(state, self.key_schedule[i])
            state = self._inv_mix_columns(state)
            state = self._inv_shift_rows(state)
            state = self._inv_sub_bytes(state)

        # Final round key addition
        state = self._add_round_key(state, self.key_schedule[0])

        # Convert state back to bytes
        return bytes(state)

    def encrypt(self, data: bytes, mode: str = 'ECB') -> bytes:
        """
        Encrypt data using the specified mode of operation.

        Args:
            data: Data to encrypt
            mode: Mode of operation ('ECB' or 'CBC')

        Returns:
            Encrypted data
        """
        # Pad data to a multiple of 16 bytes
        padded_data = self._pad(data)

        if mode == 'ECB':
            # Electronic Codebook mode - encrypt each block independently
            result = bytearray()
            for i in range(0, len(padded_data), 16):
                block = padded_data[i:i + 16]
                encrypted_block = self.encrypt_block(block)
                result.extend(encrypted_block)
            return bytes(result)

        elif mode == 'CBC':
            # Cipher Block Chaining mode
            # Use a random IV for CBC mode
            iv = os.urandom(16)
            result = bytearray(iv)  # Prepend IV to result

            prev_block = iv
            for i in range(0, len(padded_data), 16):
                block = padded_data[i:i + 16]
                # XOR with previous ciphertext block (or IV for first block)
                xored_block = bytes(b1 ^ b2 for b1, b2 in zip(block, prev_block))
                encrypted_block = self.encrypt_block(xored_block)
                result.extend(encrypted_block)
                prev_block = encrypted_block

            return bytes(result)

        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def decrypt(self, data: bytes, mode: str = 'ECB') -> bytes:
        """
        Decrypt data using the specified mode of operation.

        Args:
            data: Data to decrypt
            mode: Mode of operation ('ECB' or 'CBC')

        Returns:
            Decrypted data
        """
        if mode == 'ECB':
            # Electronic Codebook mode - decrypt each block independently
            result = bytearray()
            for i in range(0, len(data), 16):
                block = data[i:i + 16]
                decrypted_block = self.decrypt_block(block)
                result.extend(decrypted_block)

            # Remove padding
            return self._unpad(result)

        elif mode == 'CBC':
            # Cipher Block Chaining mode
            if len(data) < 16:
                raise ValueError("Invalid ciphertext (too short for CBC mode)")

            # Extract IV from the beginning
            iv = data[:16]
            ciphertext = data[16:]

            result = bytearray()
            prev_block = iv

            for i in range(0, len(ciphertext), 16):
                block = ciphertext[i:i + 16]
                decrypted_block = self.decrypt_block(block)
                # XOR with previous ciphertext block (or IV for first block)
                xored_block = bytes(b1 ^ b2 for b1, b2 in zip(decrypted_block, prev_block))
                result.extend(xored_block)
                prev_block = block

            # Remove padding
            return self._unpad(result)

        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def _pad(self, data: bytes) -> bytes:
        """
        Pad the data to a multiple of 16 bytes using PKCS#7 padding.

        Args:
            data: Data to pad

        Returns:
            Padded data
        """
        padding_length = 16 - (len(data) % 16)
        padding = bytes([padding_length] * padding_length)
        return data + padding

    def _unpad(self, data: bytes) -> bytes:
        """
        Remove PKCS#7 padding from the data.

        Args:
            data: Padded data

        Returns:
            Unpadded data
        """
        padding_length = data[-1]
        if padding_length > 16:
            return data  # Invalid padding, return data as is

        # Check if padding is valid
        for i in range(1, padding_length + 1):
            if data[-i] != padding_length:
                return data  # Invalid padding, return data as is

        return data[:-padding_length]