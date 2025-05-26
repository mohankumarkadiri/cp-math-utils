"""
ds.py

Data structures for competitive programming.
Includes Union-Find (DSU), Segment Tree, Fenwick Tree (BIT), Sparse Table, GenericSegmentTree

Author: Kadiri Mohan Kumar
"""

from typing import List, Callable
import math

# ------------------------------
# ✅ Union-Find (Disjoint Set Union - DSU)
# ------------------------------

class DSU:
    """
    Disjoint Set Union (Union-Find) with Path Compression and Union by Rank.

    Time Complexity:
        - find(): O(α(n)) [inverse Ackermann, nearly constant]
        - union(): O(α(n))
    Space Complexity: O(n)
    """

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        xr, yr = self.find(x), self.find(y)
        if xr == yr:
            return False
        if self.rank[xr] < self.rank[yr]:
            self.parent[xr] = yr
        elif self.rank[xr] > self.rank[yr]:
            self.parent[yr] = xr
        else:
            self.parent[yr] = xr
            self.rank[xr] += 1
        return True


# ------------------------------
# ✅ Fenwick Tree (Binary Indexed Tree)
# ------------------------------

class FenwickTree:
    """
    Fenwick Tree for Range Sum Queries and Point Updates.

    Time Complexity:
        - update(): O(log n)
        - query(): O(log n)
    Space Complexity: O(n)
    """

    def __init__(self, size: int):
        self.size = size
        self.tree = [0] * (size + 1)

    def update(self, index: int, delta: int):
        index += 1  # BIT is 1-indexed
        while index <= self.size:
            self.tree[index] += delta
            index += index & -index

    def query(self, index: int) -> int:
        # Sum from [0..index]
        index += 1
        result = 0
        while index > 0:
            result += self.tree[index]
            index -= index & -index
        return result

    def range_query(self, l: int, r: int) -> int:
        return self.query(r) - self.query(l - 1)


# ------------------------------
# ✅ Segment Tree (Sum Range)
# ------------------------------

class SegmentTree:
    """
    Segment Tree for Range Sum Query with Point Update.

    Time Complexity:
        - build(): O(n)
        - update(): O(log n)
        - query(): O(log n)
    Space Complexity: O(n)
    """

    def __init__(self, data: List[int]):
        self.n = len(data)
        self.tree = [0] * (2 * self.n)
        for i in range(self.n):
            self.tree[self.n + i] = data[i]
        for i in range(self.n - 1, 0, -1):
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]

    def update(self, index: int, value: int):
        index += self.n
        self.tree[index] = value
        while index > 1:
            index //= 2
            self.tree[index] = self.tree[2 * index] + self.tree[2 * index + 1]

    def query(self, left: int, right: int) -> int:
        # Query sum in [left, right)
        result = 0
        left += self.n
        right += self.n
        while left < right:
            if left % 2:
                result += self.tree[left]
                left += 1
            if right % 2:
                right -= 1
                result += self.tree[right]
            left //= 2
            right //= 2
        return result


# ------------------------------
# ✅ Sparse Table (RMQ - min)
# ------------------------------

class SparseTable:
    """
    Sparse Table for Static Range Minimum Query (RMQ).

    Time Complexity:
        - build(): O(n log n)
        - query(): O(1)
    Space Complexity: O(n log n)
    """

    def __init__(self, arr: List[int]):
        self.n = len(arr)
        self.P = int(math.log2(self.n)) + 1
        self.st = [[0] * self.P for _ in range(self.n)]

        for i in range(self.n):
            self.st[i][0] = arr[i]

        for j in range(1, self.P):
            for i in range(self.n - (1 << j) + 1):
                self.st[i][j] = min(self.st[i][j - 1], self.st[i + (1 << (j - 1))][j - 1])

    def query(self, l: int, r: int) -> int:
        # Query min in [l, r]
        length = r - l + 1
        k = int(math.log2(length))
        return min(self.st[l][k], self.st[r - (1 << k) + 1][k])


class GenericSegmentTree:
    """
    Generic Segment Tree with any associative merge function.

    Time Complexity:
        - build(): O(n)
        - update(): O(log n)
        - query(): O(log n)
    Space Complexity: O(n)

    Args:
        data: List of elements
        func: Merge function (e.g., sum, min, max)
        identity: Identity element for func (e.g., 0 for sum, inf for min)
    """

    def __init__(self, data: List[int], func: Callable, identity):
        self.n = len(data)
        self.func = func
        self.identity = identity
        self.tree = [identity] * (2 * self.n)

        # Build tree
        for i in range(self.n):
            self.tree[self.n + i] = data[i]
        for i in range(self.n - 1, 0, -1):
            self.tree[i] = func(self.tree[2 * i], self.tree[2 * i + 1])

    def update(self, index: int, value: int):
        index += self.n
        self.tree[index] = value
        while index > 1:
            index //= 2
            self.tree[index] = self.func(self.tree[2 * index], self.tree[2 * index + 1])

    def query(self, left: int, right: int) -> int:
        # Query in [left, right)
        result = self.identity
        left += self.n
        right += self.n
        while left < right:
            if left % 2:
                result = self.func(result, self.tree[left])
                left += 1
            if right % 2:
                right -= 1
                result = self.func(result, self.tree[right])
            left //= 2
            right //= 2
        return result
