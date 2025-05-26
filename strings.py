"""
strings.py

Essential string algorithms for competitive programming and backend utilities.

Includes:
- KMP Pattern Matching
- Z-Algorithm
- Rabin-Karp Hashing
- Manacher's Algorithm
- Longest Prefix Suffix (LPS) Array
- Palindrome Checks
- Anagram Checks

Author: Kadiri Mohan Kumar

"""

from typing import List

def build_lps(pattern: str) -> List[int]:
    """
    Builds the LPS (Longest Prefix Suffix) array for KMP algorithm.

    Time: O(n), Space: O(n)
    """
    lps = [0] * len(pattern)
    length = 0  # length of previous longest prefix suffix

    for i in range(1, len(pattern)):
        while length > 0 and pattern[i] != pattern[length]:
            length = lps[length - 1]
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
    return lps

def kmp_search(text: str, pattern: str) -> List[int]:
    """
    Knuth-Morris-Pratt (KMP) Pattern Matching

    Time: O(n + m), Space: O(m)
    Returns all starting indices where pattern is found in text.
    """
    lps = build_lps(pattern)
    i = j = 0
    positions = []

    while i < len(text):
        if text[i] == pattern[j]:
            i += 1
            j += 1
        if j == len(pattern):
            positions.append(i - j)
            j = lps[j - 1]
        elif i < len(text) and text[i] != pattern[j]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    return positions

def z_algorithm(s: str) -> List[int]:
    """
    Z-Algorithm for pattern matching and prefix queries.

    Time: O(n), Space: O(n)
    Returns Z-array where Z[i] is the length of the longest substring
    starting at i which is also a prefix of s.
    """
    n = len(s)
    z = [0] * n
    l = r = 0
    for i in range(1, n):
        if i <= r:
            z[i] = min(r - i + 1, z[i - l])
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
        if i + z[i] - 1 > r:
            l, r = i, i + z[i] - 1
    return z

def rabin_karp(text: str, pattern: str, base: int = 256, mod: int = 10**9 + 7) -> List[int]:
    """
    Rabin-Karp Rolling Hash Pattern Matching

    Time: O(n + m), Space: O(1)
    Returns all indices where pattern is found in text.
    """
    n, m = len(text), len(pattern)
    if m > n:
        return []

    hpattern = 0
    htext = 0
    h = 1
    for i in range(m - 1):
        h = (h * base) % mod

    for i in range(m):
        hpattern = (base * hpattern + ord(pattern[i])) % mod
        htext = (base * htext + ord(text[i])) % mod

    positions = []
    for i in range(n - m + 1):
        if hpattern == htext:
            if text[i:i + m] == pattern:
                positions.append(i)
        if i < n - m:
            htext = (base * (htext - ord(text[i]) * h) + ord(text[i + m])) % mod
            if htext < 0:
                htext += mod
    return positions

def manacher(s: str) -> List[int]:
    """
    Manacher's Algorithm for Longest Palindromic Substring lengths

    Time: O(n), Space: O(n)
    Returns array of palindrome radii centered at each character.
    """
    t = '#' + '#'.join(s) + '#'
    n = len(t)
    p = [0] * n
    c = r = 0
    for i in range(n):
        mirror = 2 * c - i
        if i < r:
            p[i] = min(r - i, p[mirror])
        while i + 1 + p[i] < n and i - 1 - p[i] >= 0 and t[i + 1 + p[i]] == t[i - 1 - p[i]]:
            p[i] += 1
        if i + p[i] > r:
            c, r = i, i + p[i]
    return p

def is_palindrome(s: str) -> bool:
    """
    Check if a string is a palindrome.

    Time: O(n), Space: O(1)
    """
    return s == s[::-1]

def are_anagrams(s1: str, s2: str) -> bool:
    """
    Check if two strings are anagrams.

    Time: O(n), Space: O(1)
    """
    return sorted(s1) == sorted(s2)
