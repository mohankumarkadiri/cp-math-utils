"""
math_utils.py
A utility module providing essential mathematical and number theory functions for competitive programming.

Includes:
- gcd, lcm
- Modular arithmetic (add, multiply, inverse, exponentiation)
- Sieve of Eratosthenes (normal & segmented)
- Primality tests (basic & Miller-Rabin)
- Euler's Totient function
- Prime factorization
- Chinese Remainder Theorem (CRT)
- Modular combinatorics (nCr mod p with precomputation)
- Additional helpers like factorial, inverse factorial with memoization

Author: Kadiri Mohan Kumar
Date: 2025
"""

from typing import List, Tuple

# -------------------------
# Basic Math Functions
# -------------------------

def gcd(a: int, b: int) -> int:
    """
    Computes the Greatest Common Divisor (GCD) of two integers using Euclid's algorithm.

    Time Complexity: O(log(min(a, b)))
    Space Complexity: O(1)

    Args:
        a (int): First number.
        b (int): Second number.

    Returns:
        int: GCD of a and b.
    """
    while b:
        a, b = b, a % b
    return abs(a)

def lcm(a: int, b: int) -> int:
    """
    Computes the Least Common Multiple (LCM) of two integers using GCD.

    Time Complexity: O(log(min(a, b))) due to gcd call
    Space Complexity: O(1)

    Args:
        a (int): First number.
        b (int): Second number.

    Returns:
        int: LCM of a and b.
    """
    return abs(a // gcd(a, b) * b)

# -------------------------
# Modular Arithmetic
# -------------------------

def mod_add(a: int, b: int, mod: int) -> int:
    """
    Modular addition: (a + b) % mod

    Time Complexity: O(1)
    Space Complexity: O(1)
    """
    return (a % mod + b % mod) % mod

def mod_sub(a: int, b: int, mod: int) -> int:
    """
    Modular subtraction: (a - b) % mod

    Time Complexity: O(1)
    Space Complexity: O(1)
    """
    return (a % mod - b % mod + mod) % mod

def mod_mul(a: int, b: int, mod: int) -> int:
    """
    Modular multiplication: (a * b) % mod

    Time Complexity: O(1)
    Space Complexity: O(1)
    """
    return (a % mod) * (b % mod) % mod

def mod_exp(base: int, exp: int, mod: int) -> int:
    """
    Modular exponentiation: (base^exp) % mod using binary exponentiation.

    Time Complexity: O(log(exp))
    Space Complexity: O(1)

    Args:
        base (int): Base integer.
        exp (int): Exponent (non-negative).
        mod (int): Modulus.

    Returns:
        int: (base^exp) mod mod.
    """
    result = 1
    base %= mod
    while exp > 0:
        if exp & 1:
            result = (result * base) % mod
        base = (base * base) % mod
        exp >>= 1
    return result

def mod_inv(a: int, mod: int) -> int:
    """
    Computes modular inverse of a under modulo mod (mod must be prime) using Fermat's Little Theorem:
    a^(mod-2) % mod

    Time Complexity: O(log(mod))
    Space Complexity: O(1)

    Args:
        a (int): Number to find inverse for.
        mod (int): Prime modulus.

    Returns:
        int: Modular inverse of a under mod.
    """
    return mod_exp(a, mod - 2, mod)

# -------------------------
# Sieve of Eratosthenes
# -------------------------

def sieve(n: int) -> List[bool]:
    """
    Computes prime status of numbers from 0 to n using Sieve of Eratosthenes.

    Time Complexity: O(n log log n)
    Space Complexity: O(n)

    Args:
        n (int): Upper limit.

    Returns:
        List[bool]: List where index i is True if i is prime, else False.
    """
    prime = [True] * (n + 1)
    prime[0], prime[1] = False, False

    for i in range(2, int(n**0.5) + 1):
        if prime[i]:
            for j in range(i*i, n+1, i):
                prime[j] = False
    return prime

def segmented_sieve(low: int, high: int) -> List[int]:
    """
    Generate primes in range [low, high] using segmented sieve.

    Time Complexity: O((high - low + 1) log log high)
    Space Complexity: O(high - low + 1)

    Args:
        low (int): Start of range.
        high (int): End of range.

    Returns:
        List[int]: List of primes in the range.
    """
    import math

    limit = int(math.sqrt(high)) + 1
    prime = sieve(limit)
    primes = [i for i, val in enumerate(prime) if val]

    segment = [True] * (high - low + 1)

    for p in primes:
        start = max(p*p, ((low + p - 1)//p)*p)
        for j in range(start, high + 1, p):
            segment[j - low] = False

    result = []
    for i in range(low, high + 1):
        if i > 1 and segment[i - low]:
            result.append(i)
    return result

# -------------------------
# Primality Tests
# -------------------------

def is_prime_basic(n: int) -> bool:
    """
    Check if number is prime using trial division.

    Time Complexity: O(sqrt(n))
    Space Complexity: O(1)

    Args:
        n (int): Number to check.

    Returns:
        bool: True if prime, False otherwise.
    """
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def miller_rabin(n: int, k: int = 5) -> bool:
    """
    Miller-Rabin probabilistic primality test.

    Time Complexity: O(k * log^3 n)
    Space Complexity: O(1)

    Args:
        n (int): Number to test.
        k (int): Number of iterations for accuracy.

    Returns:
        bool: True if probably prime, False if composite.
    """
    import random

    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0:
        return False

    # write n-1 as d*2^r
    r, d = 0, n - 1
    while d % 2 == 0:
        d //= 2
        r += 1

    def check(a, d, n, r):
        x = mod_exp(a, d, n)
        if x == 1 or x == n - 1:
            return True
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                return True
        return False

    for _ in range(k):
        a = random.randint(2, n - 2)
        if not check(a, d, n, r):
            return False
    return True

# -------------------------
# Euler’s Totient Function
# -------------------------

def phi(n: int) -> int:
    """
    Compute Euler's Totient function: count of numbers <= n that are coprime with n.

    Time Complexity: O(sqrt(n))
    Space Complexity: O(1)

    Args:
        n (int): Number.

    Returns:
        int: Totient of n.
    """
    result = n
    p = 2
    while p * p <= n:
        if n % p == 0:
            while n % p == 0:
                n //= p
            result -= result // p
        p += 1
    if n > 1:
        result -= result // n
    return result

# -------------------------
# Prime Factorization
# -------------------------

def prime_factors(n: int) -> List[int]:
    """
    Return the prime factors of n in ascending order (with repetition).

    Time Complexity: O(sqrt(n))
    Space Complexity: O(log n) (worst-case number of factors)

    Args:
        n (int): Number to factor.

    Returns:
        List[int]: Prime factors.
    """
    factors = []
    while n % 2 == 0:
        factors.append(2)
        n //= 2
    f = 3
    while f * f <= n:
        while n % f == 0:
            factors.append(f)
            n //= f
        f += 2
    if n > 1:
        factors.append(n)
    return factors

# -------------------------
# Chinese Remainder Theorem (CRT)
# -------------------------

def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """
    Extended Euclidean Algorithm.
    Returns gcd, x, y where ax + by = gcd(a,b).

    Time Complexity: O(log(min(a,b)))
    Space Complexity: O(1)
    """
    if b == 0:
        return a, 1, 0
    gcd_val, x1, y1 = extended_gcd(b, a % b)
    x = y1
    y = x1 - (a // b) * y1
    return gcd_val, x, y

def crt(remainders: List[int], moduli: List[int]) -> int:
    """
    Solve system of congruences using Chinese Remainder Theorem.
    Assumes moduli are pairwise coprime.

    Time Complexity: O(k log M), where k = number of congruences, M = product of moduli
    Space Complexity: O(1)

    Args:
        remainders (List[int]): List of remainders.
        moduli (List[int]): List of moduli.

    Returns:
        int: The unique solution modulo product of moduli.
    """
    assert len(remainders) == len(moduli)
    prod = 1
    for m in moduli:
        prod *= m

    result = 0
    for r, m in zip(remainders, moduli):
        p = prod // m
        gcd_val, inv, _ = extended_gcd(p, m)
        assert gcd_val == 1, "Moduli must be coprime"
        result += r * inv * p
    return result % prod

# -------------------------
# Modular Combinatorics
# -------------------------

class ModComb:
    """
    Class to precompute factorials and inverse factorials modulo mod for fast nCr calculations.

    Precomputation Time Complexity: O(n)
    Query Time Complexity: O(1)
    Space Complexity: O(n)

    Args:
        max_n (int): Maximum n for nCr.
        mod (int): Prime modulus.
    """
    def __init__(self, max_n: int, mod: int):
        self.mod = mod
        self.fact = [1] * (max_n + 1)
        self.inv_fact = [1] * (max_n + 1)
        for i in range(2, max_n + 1):
            self.fact[i] = (self.fact[i-1] * i) % mod
        self.inv_fact[max_n] = mod_inv(self.fact[max_n], mod)
        for i in range(max_n - 1, 0, -1):
            self.inv_fact[i] = (self.inv_fact[i + 1] * (i + 1)) % mod

    def nCr(self, n: int, r: int) -> int:
        """
        Compute nCr % mod.

        Time Complexity: O(1)
        Space Complexity: O(1)

        Args:
            n (int): Number of items.
            r (int): Number of selections.

        Returns:
            int: nCr modulo mod.
        """
        if r > n or r < 0:
            return 0
        return (self.fact[n] * self.inv_fact[r] % self.mod) * self.inv_fact[n - r] % self.mod

# -------------------------
# Additional Helpers
# -------------------------

def factorial(n: int) -> int:
    """
    Compute factorial of n.

    Time Complexity: O(n)
    Space Complexity: O(1)

    Args:
        n (int): Number.

    Returns:
        int: n!
    """
    result = 1
    for i in range(2, n+1):
        result *= i
    return result

def inverse_factorial(n: int, mod: int) -> int:
    """
    Compute modular inverse of factorial n! under prime mod.

    Time Complexity: O(n)
    Space Complexity: O(1)

    Args:
        n (int): Number.
        mod (int): Prime modulus.

    Returns:
        int: (n!)^(-1) mod mod.
    """
    return mod_inv(factorial(n) % mod, mod)

