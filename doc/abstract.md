**Motivation.**
Approximate string matching (ASM) is the problem of finding all occurrences of a
pattern in a text while allowing up to k errors. Instances of this problem are
ubiquitous in bioinformatics, but surprisingly few tools solve it:
some do semi-global alignment and only find a single best match, others 
are approximate and based on seed-chain-extend, while exact search schemes such as
Columba and Sahara require an index.

**Methods.**
Sassy is a library and tool for ASM of short patterns.
Sassy1 is optimized for searching a single pattern in long texts, and does so by
using AVX2 SIMD and splitting the text into 4 chunks to process one chunk per SIMD
lane. This also enables a grep-like CLI.
Sassy2 is optimized for the typical case of batch-searching many short equal-length
patterns: it searches multiple patterns in parallel, one per AVX-512 lane, and first
does a _suffix filter_ that finds locations where e.g. the 16 bp suffix
matches, to increase parallelism.

**Results.**
Sassy1 is 4-15x faster than Edlib for patterns up to length 1000 bp, and
reaches a text throughput up to 2 Gbp/s. For CRISPR off-target detection of 61 guides
in a human genome, Sassy is 100x faster than SWOffinder and only slightly slower
than CHOPOFF, while not requiring an index.
When searching many short patterns, Sassy2 is an additional 10-50x faster than
Sassy1 for short (<200 bp) texts and 2.5-4.5x faster for long (>8 kbp) texts.
With 16 threads, Sassy2 reaches over 100 Gbp/s of total text
throughput when searching barcodes in Nanopore reads.

**Availability.**
Sassy is implemented in Rust and available at github.com/RagnarGrootKoerkamp/sassy.
