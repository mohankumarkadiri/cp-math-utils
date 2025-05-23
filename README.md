# math_lib

A comprehensive Python library offering essential **mathematical** and **competitive programming** utilities.  
Includes number theory functions, data structures, graph algorithms, string helpers, and more — all designed to be efficient and easy to use.

---

## Features

- Modular arithmetic utilities (GCD, LCM, Modular Exponentiation, Inverse mod, Chinese Remainder Theorem)
- Prime testing (optimized and normal)
- Sieve of Eratosthenes for prime generation
- Data structures: Segment Trees, Fenwick Trees, Disjoint Set Union (Union-Find)
- Graph utilities: BFS, DFS, Reverse Graph (Transpose), shortest paths
- String utilities: KMP algorithm, Z-algorithm, palindrome checks
- Competitive programming helpers: direction vectors, sliding window maximum/minimum, prefix sums, frequency counters
- Well documented with time and space complexities

---

## Installation

You can install the package via pip:

```bash
pip install cp-math-utils

or clone this repo and install manually

git clone https://github.com/mohankumarkadiri/cp-math-utils.git
cd cp-math-utils
pip install .


```python
from math_lib import math_utils, helpers

print(math_utils.is_prime_optimized(101))

print(math_utils.mod_exp(2, 10, 1000))  # 1024 % 1000 = 24

for nx, ny in helpers.neighbors(1, 1, 3, 3):
    print(nx, ny)



