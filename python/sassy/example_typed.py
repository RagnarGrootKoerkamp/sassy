#!/usr/bin/env python3
"""Typed example for validating sassy type stubs with mypy/pyright."""

from sassy import Match, Searcher, features

# features() returns None
features()

# Searcher with all parameter variants
searcher_dna: Searcher = Searcher(alphabet="dna", rc=True)
searcher_iupac: Searcher = Searcher(alphabet="iupac", rc=False, alpha=0.5)
searcher_ascii: Searcher = Searcher(alphabet="ascii")

# search returns list[Match]
matches: list[Match] = searcher_dna.search(b"ATCGATCG", b"GGGGATCGATCGTTTT", k=1)

# Match properties
for m in matches:
    idx: int = m.pattern_idx
    tidx: int = m.text_idx
    start: int = m.text_start
    end: int = m.text_end
    pstart: int = m.pattern_start
    pend: int = m.pattern_end
    cost: int = m.cost
    strand: str = m.strand
    cigar: str = m.cigar
    rep: str = repr(m)

# search_many returns list[Match]
many_matches: list[Match] = searcher_ascii.search_many(
    patterns=[b"hello", b"world"],
    texts=[b"hello world", b"hi there"],
    k=1,
    threads=1,
    mode="single",
)
