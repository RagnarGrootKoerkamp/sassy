import sassy
pattern = b"ATCGATCG"
text = b"GGGGATCGATCGTTTT"
# alphabet: ascii, dna, iupac
searcher = sassy.Searcher("dna")
matches = searcher.search(pattern, text, k=1)
for i, match in enumerate(matches):
    print(f"Match {i+1}:")
    print(f"    Start: {match.text_start}")
    print(f"    End: {match.text_end}")
    print(f"    Cost: {match.cost}")
    print(f"    Strand: {match.strand}")
    print(f"    CIGAR: {match.cigar}")