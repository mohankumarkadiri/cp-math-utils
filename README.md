# cp\_math\_utils

A comprehensive Python library offering essential **mathematical** and **competitive programming** utilities.
Includes number theory functions, data structures, graph algorithms, string helpers, and more — all designed to be efficient and easy to use.

---

## Features

* Modular arithmetic utilities (GCD, LCM, Modular Exponentiation, Inverse mod, Chinese Remainder Theorem)
* Prime testing (optimized and normal)
* Sieve of Eratosthenes for prime generation
* Data structures: Segment Trees, Fenwick Trees, Disjoint Set Union (Union-Find)
* Graph utilities: BFS, DFS, Reverse Graph (Transpose), shortest paths
* String utilities: KMP algorithm, Z-algorithm, palindrome checks
* Competitive programming helpers: direction vectors, sliding window maximum/minimum, prefix sums, frequency counters
* Well documented with time and space complexities

---

## Installation

You can install the package via pip:

```bash
pip install cp-math-utils
```

Or clone this repo and install manually:

```bash
git clone https://github.com/mohankumarkadiri/cp-math-utils.git
cd cp-math-utils
pip install .
```

---

## Examples

### Prime Check & Modular Exponentiation

```python
from cp_math_utils import math_utils

print(math_utils.is_prime_optimized(101))        # True
print(math_utils.mod_exp(2, 10, 1000))           # 24
```

### Neighboring Cells (Grid Movement)

```python
from cp_math_utils import helpers

for nx, ny in helpers.neighbors(1, 1, 3, 3):
    print(nx, ny)
# Outputs: (0, 1), (1, 0), (1, 2), (2, 1)
```

### Using Segment Tree

```python
from cp_math_utils.ds import SegmentTree

arr = [1, 3, 5, 7, 9, 11]
st = SegmentTree(arr)
print(st.query(1, 3))  # Sum from index 1 to 3
st.update(1, 10)       # Update index 1 to value 10
print(st.query(1, 3))
```

---

## Documentation

Each module and function is documented with:

* Clear description
* Time and space complexity
---

## 🔗 License

This project is licensed under the [MIT License](LICENSE).