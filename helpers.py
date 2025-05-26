"""
helpers.py

Essential competitive programming helpers.

Includes:
- Direction vectors for 2D grids
- Valid neighbor generator
- Reverse a Graph


Author: Kadiri Mohan Kumar

"""

from typing import List, Tuple

# Common 2D directions (up, down, left, right)
DIR4: List[Tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# 8 directions (includes diagonals)
DIR8: List[Tuple[int, int]] = DIR4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]

# Knight moves for chess-based problems
KNIGHT_MOVES: List[Tuple[int, int]] = [
    (-2, -1), (-2, 1), (-1, -2), (-1, 2),
    (1, -2), (1, 2), (2, -1), (2, 1)
]

def neighbors(x: int, y: int, n: int, m: int, directions: List[Tuple[int, int]] = DIR4):
    """
    Yields valid neighboring coordinates in a grid.

    Time: O(1) per call
    """
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < n and 0 <= ny < m:
            yield (nx, ny)


def reverse_graph(adj: List[List[int]], n: int) -> List[List[int]]:
    """
    Returns the transpose (reverse) of a directed graph.

    Parameters:
    - adj: adjacency list of the original graph, where adj[u] contains all v such that u -> v
    - n: number of vertices

    Returns:
    - rev_adj: adjacency list of the reversed graph, where all edges are reversed

    Time: O(n + E), Space: O(n + E)
    """
    rev_adj = [[] for _ in range(n)]
    for u in range(n):
        for v in adj[u]:
            rev_adj[v].append(u)
    return rev_adj
