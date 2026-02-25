#!/usr/bin/env python3
"""
Example usage of the sassy Python bindings.
"""

import sassy

# Random text of length 1000000000
import random
import time

sassy.features()

print("gen")
n = 100000
m = 20
k = 1
text = bytes(random.choices(b"ACGT", k=n))
pattern = bytes(random.choices(b"ACGT", k=m))
# time instant now

searcher = sassy.Searcher("dna", rc=False)
for _ in range(10):
    start = time.time()
    matches = searcher.search(pattern, text, k=k)
    print(f"GB/s: {n / (time.time()-start) / 10**9}")

# Example 1: Simple DNA search
print("=== DNA Search Example ===")
pattern = b"ATCGATCG"
text = b"GGGGATCGATCGTTTT"

# Search with extended IUPAC alphabet and allow overhang
matches = sassy.Searcher("iupac", alpha=0.5).search(pattern, text, k=0)

print(f"Pattern: {pattern.decode()}")
print(f"Text:  {text.decode()}")
print(f"Found {len(matches)} matches:")

for i, match in enumerate(matches):
    print(f"  Match {i+1}:")
    print(f"    Start: {match.text_start}")
    print(f"    End: {match.text_end}")
    print(f"    Cost: {match.cost}")
    print(f"    Strand: {match.strand}")
    print(f"    CIGAR: {match.cigar}")

# Example 2: Reverse complement search
print("\n=== Reverse Complement Example ===")
searcher = sassy.Searcher("dna")

pattern2 = b"GCTAGCTA"
text2 = b"AAAAAGCTAGCTAAAAA"

matches2 = searcher.search(pattern2, text2, k=1)

print(f"Pattern: {pattern2.decode()}")
print(f"Text:  {text2.decode()}")
print(f"Found {len(matches2)} matches with k=1:")

for i, match in enumerate(matches2):
    print(f"  Match {i+1}: cost={match.cost}, strand={match.strand}")

# Example 3: ASCII search
print("\n=== ASCII Search Example ===")
pattern3 = b"hello"
text3 = b"world hello there"

matches3 = sassy.Searcher("ascii").search(pattern3, text3, k=0)

print(f"Pattern: {pattern3.decode()}")
print(f"Text:  {text3.decode()}")
print(f"Found {len(matches3)} matches:")

for i, match in enumerate(matches3):
    print(
        f"  Match {i+1}: start={match.text_start}, end={match.text_end}, cost={match.cost}"
    )

# Example 4: Search multiple patterns in multiple strings
print("\n=== Search_Many Example ===")
patterns4 = [b"hello", b"world"]
texts4 = [b"hello world", b"hi Hello there!", b"the world wide web is full of words"]

# mode controls the backend; see the documentation of `search::SearchMode`.
matches4 = sassy.Searcher("ascii").search_many(
    patterns4, texts4, k=1, mode="single", threads=2
)

for i, match in enumerate(matches4):
    print(
        f"  Match {i+1}: pattern={match.pattern_idx} text={match.text_idx} start={match.text_start}, end={match.text_end}, cost={match.cost}"
    )

# Example 5: Search all end positions
print("\n=== Search All Example ===")
pattern5 = b"ATCGATCG"
text5 = b"GGGGATCGATCGTTTT"

searcher5 = sassy.Searcher("dna", rc=False)
matches_default = searcher5.search(pattern5, text5, k=2)
matches_all = searcher5.search_all(pattern5, text5, k=2)

print(f"Pattern: {pattern5.decode()}")
print(f"Text:    {text5.decode()}")
print(f"search() returned {len(matches_default)} match(es)")
print(f"search_all() returned {len(matches_all)} match(es)")

for i, match in enumerate(matches_all):
    print(
        f"  Match {i+1}: start={match.text_start}, end={match.text_end}, cost={match.cost}, cigar={match.cigar}"
    )
