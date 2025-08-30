[![crates.io](https://img.shields.io/crates/v/sassy.svg)](https://crates.io/crates/sassy)
[![docs.rs](https://img.shields.io/docsrs/sassy.svg)](https://docs.rs/sassy)
[![PyPI](https://img.shields.io/pypi/v/sassy-rs.svg)](https://pypi.org/project/sassy-rs/)

# Sassy: SIMD-accelerated Approximate String Matching

Sassy is a library and tool for searching short strings in texts,
a problem that goes by many names:
- approximate string matching,
- pattern matching,
- fuzzy searching.

The motivating application is searching short (length 20 to 100) DNA sequences
in a human genome or e.g. in a set of reads.
Sassy generally works well for patterns/queries up to length 1000,
and supports both ASCII and DNA.

Highlights:
- Sassy uses bitpacking and SIMD (both AVX and NEON supported).
  Its main novelty is tiling these in the text direction.
- Support for _overhang_ alignments where the pattern extends beyond the text.
- Support for (case-insensitive) ASCII, DNA (`ACGT`), and
  [IUPAC](https://www.bioinformatics.org/sms/iupac.html) (=`ACGT+NYR...`) alphabets.
- Rust library (`cargo add sassy`), binary (`cargo install sassy`), Python
  bindings (`pip install sassy-rs`), and C bindings (see below).

See **the paper** below, and corresponding evals in [evals/](evals/).

> Rick Beeloo and Ragnar Groot Koerkamp.  
> Sassy: Searching Short DNA Strings in the 2020s.  
> bioRxiv, July 2025.
> https://doi.org/10.1101/2025.07.22.666207.


## Usage

### 0. Rust library

A larger example can be found in [`src/lib.rs`](src/lib.rs).

```rust
use sassy::{Searcher, Match, profiles::{Dna}, Strand};

let pattern = b"ATCG";
let text = b"AAAATTGAAA";
let k = 1;

let mut searcher = Searcher::<Dna>::new_fwd();
let matches = searcher.search(pattern, &text, k);

assert_eq!(matches.len(), 1);

assert_eq!(matches[0].text_start, 3);
assert_eq!(matches[0].text_end, 7);
assert_eq!(matches[0].cost, 1);
assert_eq!(matches[0].strand, Strand::Fwd);
assert_eq!(matches[0].cigar.to_string(), "2=1X1=");
```

### 1. Command-line interface (CLI)

**Build and install** using `cargo`:

```bash
cargo install sassy
```

**Search a pattern** `ATGAGCA` in `text.fasta` with ≤1 edit:
```bash
sassy search --pattern ATGAGCA --alphabet dna -k 1 text.fasta
```
or search all records of a fasta file with `--pattern-fasta <fasta-file>` instead of `--pattern`.

For the alphabets see [supported alphabets](#supported-alphabets)

**CRISPR off-target search** for one or more guides in `guides.txt`:
```bash
sassy crispr --threads 8 --guide guides.txt --k 5 --max-n-frac 0.1 --output hits.tsv hg38.fasta
```

Allows `<= k` edits in the sgRNA, and the PAM (the last 3 characters of each guide) has to match exactly, unless `--allow-pam-edits` is given.

Output of the `crispr` command is a tab-delimited file with one row per hit, e.g.:

```text
guide                    text_id  cost  strand  start     end       match_region             cigar
GAGTCCGAGCAGAAGAAGAANGG  chr21    5     +       5024135   5024154   GAGGCCACAGAGAAGAGGG      3=1X2=1D1=1D3=1D5=1D4=
GAGTCCGAGCAGAAGAAGAANGG  chr21    3     +       21087337  21087359  gagaccgaggagaagaaaaagg   3=1X5=1X7=1D5=
GAGTCCGAGCAGAAGAAGAANGG  chr21    3     -       9701297   9701320   GACTCGAGCATGAAGAAGAAAGG  2=1X1=1D6=1I12=
GAGTCCGAGCAGAAGAAGAANGG  chr21    5     -       46396975  46396998  CAGTCCCAGCAGACGACGGACGG  1X5=1X6=1X2=1X1=1X4=
```

The `start` and `end` are 0-based open-ended (i.e. 0-based inclusive of the
start, but exclusive of the end), and `start` is always less then `end`
(regardless of the strand).  The 
`match_region` reported will be the sequence from the target file when `strand` is `+`, or the reverse complement
of the sequence from the target file when `strand` is `-`, so that it matches the `guide` sequence.
The `cigar` is always oriented to read left-to-right with the provided guide and `match_region` sequences.

Note that this searches for approximate occurrences of the guide
sequence itself, and _not_ for reverse-complement _binding_ sites.
If binding sites are to be found, please reverse-complement the input or output manually.

### 2. Python bindings

PyPI wheels can be installed with:

```bash
pip install sassy-rs 
```

```python
import sassy

pattern = b"ACTG"
text    = b"ACGGCTACGCAGCATCATCAGCAT"

searcher = sassy.Searcher("dna") # ascii / dna / iupac
matches  = searcher.search(pattern, text, k=1)

for m in matches:
    print(m)
```

See [python/README.md](python/README.md) for more details.

### 3. C library

See [c/README.md](c/README.md) for details. Quick example:

```c
#include "sassy.h"

int main() {
    const char* pattern = "ACTG";
    const char* text    = "ACGGCTACGCAGCATCATCAGCAT";

    // DNA alphabet, with reverse complement, without overhang.
    sassy_SearcherType* searcher = sassy_searcher("dna", true, NAN);
    sassy_Match* out_matches = NULL;
    size_t n_matches = search(searcher,
                              pattern, strlen(pattern),
                              text, strlen(text),
                              1, // k=1
                              &out_matches);

    sassy_matches_free(out_matches, n_matches);
    sassy_searcher_free(searcher);
}
```
