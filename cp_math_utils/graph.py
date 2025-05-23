"""
graph.py

Essential graph algorithms for competitive programming and backend use.

Includes:
- BFS, DFS
- Dijkstra's Algorithm
- Topological Sort
- Union-Find (Disjoint Set Union)
- Detect Cycle (Undirected & Directed)
- Shortest Path (Unweighted)

Author: Kadiri Mohan Kumar

"""

from collections import deque, defaultdict
import heapq

def bfs(graph, start):
    """
    Breadth-First Search (BFS)

    Time: O(V + E), Space: O(V)
    """
    visited = set()
    queue = deque([start])
    order = []

    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                queue.append(neighbor)
    return order

def dfs(graph, start, visited=None, order=None):
    """
    Depth-First Search (DFS)

    Time: O(V + E), Space: O(V)
    """
    if visited is None:
        visited = set()
    if order is None:
        order = []

    visited.add(start)
    order.append(start)
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited, order)
    return order

def dijkstra(graph, start):
    """
    Dijkstra's Algorithm (for weighted graphs)

    Time: O((V + E) log V), Space: O(V)
    Returns: distances dict
    """
    dist = defaultdict(lambda: float('inf'))
    dist[start] = 0
    pq = [(0, start)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))
    return dict(dist)

def topological_sort(graph):
    """
    Topological Sort (Kahn's Algorithm)

    Time: O(V + E), Space: O(V)
    """
    in_degree = defaultdict(int)
    for u in graph:
        for v in graph[u]:
            in_degree[v] += 1

    queue = deque([u for u in graph if in_degree[u] == 0])
    order = []

    while queue:
        u = queue.popleft()
        order.append(u)
        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    if len(order) != len(graph):
        raise ValueError("Graph has a cycle")
    return order

class UnionFind:
    """
    Union-Find / Disjoint Set Union (DSU)

    Time: O(α(N)) per op (amortized), Space: O(N)
    """
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if x != self.parent[x]:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        xr = self.find(x)
        yr = self.find(y)
        if xr == yr:
            return False
        if self.rank[xr] < self.rank[yr]:
            self.parent[xr] = yr
        else:
            self.parent[yr] = xr
            if self.rank[xr] == self.rank[yr]:
                self.rank[xr] += 1
        return True

def detect_cycle_undirected(graph, n):
    """
    Detect cycle in an undirected graph using Union-Find

    Time: O(E * α(N)), Space: O(N)
    """
    uf = UnionFind(n)
    for u in graph:
        for v in graph[u]:
            if u < v:  # avoid double counting
                if not uf.union(u, v):
                    return True
    return False

def detect_cycle_directed(graph):
    """
    Detect cycle in a directed graph (DFS approach)

    Time: O(V + E), Space: O(V)
    """
    visited = set()
    stack = set()

    def dfs(u):
        visited.add(u)
        stack.add(u)
        for v in graph[u]:
            if v not in visited:
                if dfs(v):
                    return True
            elif v in stack:
                return True
        stack.remove(u)
        return False

    for u in graph:
        if u not in visited:
            if dfs(u):
                return True
    return False

def shortest_path_unweighted(graph, start):
    """
    Shortest path in unweighted graph using BFS

    Time: O(V + E), Space: O(V)
    Returns: dict of node -> distance from start
    """
    dist = {start: 0}
    queue = deque([start])
    while queue:
        u = queue.popleft()
        for v in graph[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist
