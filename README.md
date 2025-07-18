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
in a human genome or e.g. sets of reads, but it generally works well for
patterns/queries up to length 1000, and also supports plain ASCII alphabet.

Highlights:
- Sassy uses bitpacking and SIMD, and its main novelty is tiling these in the
  text direction.
- Support for _overhang_ alignments where the pattern extends beyond the text.
- Support for (case-insensitive) ASCII, DNA (`ACGT`), and
  [IUPAC](https://www.bioinformatics.org/sms/iupac.html) (=DNA+`NYR`...) alphabets.
- Rust library (`cargo add sassy`), binary (`cargo install sassy`), Python
  bindings (`pip install sassy-rs`), and C bindings (see below).

**The paper** can be found at TODO, and evals are in [evals/](evals/).

## Usage

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

**CRISPR off-target search** for guides in `guides.txt`:
```bash
sassy crispr --guide guides.txt --k 1  text.fasta
```
Allows `<= k` edits in the sgRNA, and the PAM has to match exactly, unless
`--allow-pam-edits` is given.

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
