#!/usr/bin/env python3
"""
Demonstrate search_all_alignments: enumerate every valid alignment at every matched
end position, not just the single greedy best one.

Use case: when you need all distinct CIGAR strings / text_start values that
explain a match (e.g. for probabilistic scoring, read-graph construction, or
debugging ambiguous alignments).

The `margin` parameter (default 0) extends enumeration to sub-optimal alignments:
each group yields alignments with cost <= optimal_cost + margin.  This is useful
when the best alignment is ambiguous and near-optimal alternatives are meaningful.

Run after building the bindings:
    maturin develop --features python
    python python/sassy/example_search_all_alignments.py
"""

import re
import time
from math import comb
from sassy import AllAlignmentsAtPosIter, Match, Searcher


def render_alignment(m: Match, text: bytes, pattern: bytes) -> str:
    """
    Return a three-line ASCII alignment diagram for a Match:

        text row:    text bases, '-' where the pattern is inserted
        middle row:  '|' match, '.' mismatch, ' ' gap
        pattern row: pattern bases, '-' where the text is deleted

    CIGAR ops:
        '=' match      — both advance, '|'
        'X' mismatch   — both advance, '.'
        'D' deletion   — text advances, pattern gets '-'
        'I' insertion  — pattern advances, text gets '-'
    """
    text_row, mid_row, pat_row = [], [], []
    ti, pi = m.text_start, m.pattern_start

    for n, op in re.findall(r"(\d+)([=XDI])", m.cigar):
        for _ in range(int(n)):
            if op == "=":
                text_row.append(chr(text[ti]))
                mid_row.append("|")
                pat_row.append(chr(pattern[pi]))
                ti += 1
                pi += 1
            elif op == "X":
                text_row.append(chr(text[ti]))
                mid_row.append(".")
                pat_row.append(chr(pattern[pi]))
                ti += 1
                pi += 1
            elif op == "D":  # gap in pattern
                text_row.append(chr(text[ti]))
                mid_row.append(" ")
                pat_row.append("-")
                ti += 1
            elif op == "I":  # gap in text
                text_row.append("-")
                mid_row.append(" ")
                pat_row.append(chr(pattern[pi]))
                pi += 1

    return "".join(text_row) + "\n" + "".join(mid_row) + "\n" + "".join(pat_row)


def print_alignment(
    m: Match, text: bytes, pattern: bytes, indent: str = "    "
) -> None:
    label = f"strand={m.strand}  text[{m.text_start}:{m.text_end}]  cost={m.cost}  cigar={m.cigar}"
    print(f"{indent}{label}")
    for line in render_alignment(m, text, pattern).splitlines():
        print(f"{indent}  {line}")
    print()


# ---------------------------------------------------------------------------
# 1. Minimal example — show that one end position can have multiple alignments
# ---------------------------------------------------------------------------
print("=== Multiple alignments at one end position ===\n")

# Pattern "AT" vs text "ACT" at k=1:
#   cost=1 at text_end=3, reachable via three distinct paths:
#     Sub  : text[1:3]="CT"  — substitute C>A
#     Del  : text[0:3]="ACT" — delete C from text
#     Ins  : text[2:3]="T"   — insert A into text
text1, pattern1 = b"ACT", b"AT"
searcher = Searcher("dna", rc=False)
groups: list[AllAlignmentsAtPosIter] = searcher.search_all_alignments(
    pattern1, text1, k=1, margin=0
)

print(f"  pattern: {pattern1.decode()}   text: {text1.decode()}   k=1")
print(f"  {len(groups)} end position(s) with cost <= 1\n")
for group in groups:
    alignments: list[Match] = list(group)
    print(
        f"  text_end={alignments[0].text_end}  cost={alignments[0].cost}  "
        f"n_alignments={len(alignments)}"
    )
    for m in alignments:
        print_alignment(m, text1, pattern1)

# ---------------------------------------------------------------------------
# 2. Combinatorial explosion — why early termination is essential
# ---------------------------------------------------------------------------
print("=== Combinatorial explosion ===\n")

# The DFS branches whenever multiple ops are valid at the same matrix cell.
# Match checks cost[i-1,j-1] == g; Ins checks cost[i,j-1] == g-1 — different
# cells, so BOTH can be valid at once.  For a fully repetitive sequence every
# step branches, giving C(pattern_len, k) distinct alignments.
#
# pattern = "A" * (t + k), text = "A" * t, k insertions required.
# Alignments at the single end position = C(t+k, k).

t, k_exp = 13, 12
pattern_exp = b"A" * (t + k_exp)  # 25 A's
text_exp = b"A" * t  # 13 A's
expected = comb(t + k_exp, k_exp)  # C(25, 12) = 5,200,300

print(f"  pattern: {'A' * (t + k_exp)}  (len={t + k_exp})")
print(f"  text:    {'A' * t}  (len={t})")
print(
    f"  k={k_exp}  ->  C({t + k_exp},{k_exp}) = {expected:,} alignments at the one end position"
)
print()

# Good: take only the first alignment — the DFS yields it after ~(t+k) steps.
s_fast = Searcher("dna", rc=False)
t0 = time.perf_counter()
first_only = [
    next(iter(g))
    for g in s_fast.search_all_alignments(pattern_exp, text_exp, k_exp, margin=0)
]
t_fast = time.perf_counter() - t0
print(f"  First alignment only : {len(first_only)} result(s) in {t_fast * 1e3:.2f} ms")

# Bad: exhaust every alignment — forces the DFS to visit all C(25,12) leaves.
s_slow = Searcher("dna", rc=False)
t0 = time.perf_counter()
total_exp = sum(
    sum(1 for _ in g)
    for g in s_slow.search_all_alignments(pattern_exp, text_exp, k_exp, margin=0)
)
t_slow = time.perf_counter() - t0
print(f"  Exhaustive enumeration: {total_exp:,} alignments in {t_slow:.2f} s")
print(f"  Slowdown vs first-only: {t_slow / t_fast:.0f}x")
print()

# ---------------------------------------------------------------------------
# 3. Early termination — avoid enumerating exponentially many paths
# ---------------------------------------------------------------------------
print("=== Early termination (short-circuit) ===\n")

# A highly repetitive sequence can produce exponentially many alignments.
# Break out of the inner iterator as soon as you have enough.
MAX_PER_POS = 2
text2, pattern2 = b"AAAAAA", b"AAAA"
searcher2 = Searcher("dna", rc=False)
groups2 = searcher2.search_all_alignments(pattern2, text2, k=2, margin=0)

print(f"  pattern: {pattern2.decode()}   text: {text2.decode()}   k=2")
print(f"  (collecting at most {MAX_PER_POS} alignment(s) per end position)\n")
for group in groups2:
    collected: list[Match] = []
    for m in group:  # lazy — only advances as far as needed
        collected.append(m)
        if len(collected) >= MAX_PER_POS:
            break  # remaining paths are never computed
    first = collected[0]
    capped = len(collected) == MAX_PER_POS
    print(
        f"  text_end={first.text_end}  cost={first.cost}  "
        f"collected={len(collected)}{'+ (capped)' if capped else ''}"
    )
    for m in collected:
        print_alignment(m, text2, pattern2)

# ---------------------------------------------------------------------------
# 4. Reverse complement — strand field distinguishes Fwd / RC matches
# ---------------------------------------------------------------------------
print("=== Reverse complement search ===\n")

# AAGT appears on the + strand; its RC (ACTT) appears on the - strand.
text3, pattern3 = b"TTAAGTAGTACTT", b"AAGT"
searcher3 = Searcher("dna", rc=True)
groups3 = searcher3.search_all_alignments(pattern3, text3, k=0, margin=0)

print(f"  pattern: {pattern3.decode()}   text: {text3.decode()}   k=0  rc=True\n")
for group in groups3:
    for m in group:
        print_alignment(m, text3, pattern3)

# ---------------------------------------------------------------------------
# 5. Contrast with search() and search_all()
# ---------------------------------------------------------------------------
print("=== Comparison: search / search_all / search_all_alignments ===\n")

text4, pattern4, k4 = b"GCATGGCATG", b"GCATG", 1
s = Searcher("dna", rc=False)
best = s.search(pattern4, text4, k4)
all_ends = s.search_all(pattern4, text4, k4)
# Count total alignments across all end positions.
total = sum(
    sum(1 for _ in g) for g in s.search_all_alignments(pattern4, text4, k4, margin=0)
)

print(f"  pattern: {pattern4.decode()}   text: {text4.decode()}   k={k4}\n")
print(
    f"  search()                : {len(best):2d} match(es) [one best per local-minimum end position]\n"
    f"  search_all()            : {len(all_ends):2d} match(es) [one best alignment per end position]\n"
    f"  search_all_alignments() : {total:2d} alignment(s) [all alignments across all end positions]\n"
)

# Show every alignment for the first end position that has more than one.
groups4 = s.search_all_alignments(pattern4, text4, k4, margin=0)
for group in groups4:
    aligns = list(group)
    if len(aligns) > 1:
        print(f"  End position {aligns[0].text_end} has {len(aligns)} alignments:")
        for m in aligns:
            print_alignment(m, text4, pattern4)
        break

# ---------------------------------------------------------------------------
# 6. margin — sub-optimal alignments within a budget above optimal
# ---------------------------------------------------------------------------
print("=== Sub-optimal alignments with margin ===\n")

# margin=M tells the DFS to also yield alignments with cost <= optimal + M.
# The constraint optimal + margin <= k is enforced automatically (the budget
# is clamped to k so the search never exceeds the filled matrix window).
#
# Use case: when the optimal alignment is ambiguous and you want to see
# near-optimal alternatives ranked by cost, e.g. for multi-mapping reads.

text5, pattern5, k5 = b"ACGT", b"AGT", 2
s5 = Searcher("dna", rc=False)

print(f"  pattern: {pattern5.decode()}   text: {text5.decode()}   k={k5}\n")

for margin in (0, 1, 2):
    groups5 = s5.search_all_alignments(pattern5, text5, k5, margin=margin)
    aligns5 = [m for g in groups5 for m in g]
    by_cost: dict[int, int] = {}
    for m in aligns5:
        by_cost[m.cost] = by_cost.get(m.cost, 0) + 1
    cost_summary = "  ".join(f"cost={c}: {n}" for c, n in sorted(by_cost.items()))
    print(f"  margin={margin}  total={len(aligns5)}  [{cost_summary}]")

print()

# Show the alignments for margin=1 so cost breakdown is visible.
s5b = Searcher("dna", rc=False)
print("  Alignments with margin=1 (cost <= optimal + 1):\n")
for group in s5b.search_all_alignments(pattern5, text5, k5, margin=1):
    for m in group:
        print_alignment(m, text5, pattern5)

# optimal_cost() lets you inspect the minimum cost at each end position
# before iterating, which is useful for computing the margin threshold.
print("  optimal_cost() per end position:\n")
s5c = Searcher("dna", rc=False)
for group in s5c.search_all_alignments(pattern5, text5, k5, margin=0):
    opt = group.optimal_cost()  # call before consuming the iterator
    aligns = list(group)
    print(f"    text_end={aligns[0].text_end}  optimal_cost={opt}")
