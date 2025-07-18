[![PyPI version](https://img.shields.io/pypi/v/sassy-rs.svg)](https://pypi.org/project/sassy-rs/)

# Sassy: SIMD-accelerated Approximate String Searching

Sassy is a library and tool for approximately searching short patterns in texts,
the problem that goes by many names:
- approximate string matching,
- pattern searching,
- fuzzy searching.

The motivating application is searching short (~20bp) DNA fragments in a human genome (3GB), but it also works well for longer patterns up to ~1Kbp, and shorter texts.

---

## Key features

* Highly optimized with SIMD
* Bindings: **CLI**, **Python**, and **C**
* Supports different alphabets, `Ascii`, `Dna`, and `Iupac` or you can implement your own `Profile`
* Support overhang cost when alignments go past text boundaries

---

## Usage

> Pick the interface that best suits your workflow.

### 1. Command-line interface (CLI)

Install from source (requires Rust ≥1.73):

```bash
cargo install --git https://github.com/RagnarGrootKoerkamp/sassy
```

Explore the sub-commands:

```bash
sassy --help
```

#### Search patterns
Searching a single pattern (`ATGAGCA`) in `text.fasta` with ≤1 edit:

```bash
sassy search --pattern ATGAGCA --alphabet dna -k 1 text.fasta
```
or search with a multi-fasta file with `--pattern-fasta <fasta-file>` instead of `--pattern`.

For the alphabets see [supported alphabets](#supported-alphabets)

#### Crispr off-target
CRISPR off-target search for guides in `guides.txt`:

```bash
sassy crispr --guide guides.txt --k 1  text.fasta
```
Allows at most 1 error in the sgRNA (not the PAM), if `--allow-pam-edits` is enabled
it allows at most 1 edits across the sgRNA+PAM.

For additional CLI options see `sassy <command> --help`.

### 2. Python bindings

We regularly publish wheels on PyPi which can be installed with:

```bash
pip install sassy-rs 
```

```python
import sassy

pattern = b"ATCGATCG"
text    = b"GGGGATCGATCGTTTT"

searcher = sassy.Searcher("dna")    # ascii / dna / iupac
matches  = searcher.search(pattern, text, k=1)

for m in matches:
    print(m)
```

See [python/README.md](python/README.md) for more details.

### 3. C library

Enable the `c` feature when building, then link against the generated static
library:

```bash
cargo build --release --features c
```

Minimal usage example:

```c
#include <stdio.h>
#include "sassy.h"

int main(void) {
    const char *pattern = "ATCG";
    const char *text    = "AAAATCGT";

    SassySearcher *s = sassy_searcher_new(SASSY_ALPHABET_DNA, /*rc=*/true);
    SassyMatches  *m = sassy_search(s, pattern, 4, text, 8, /*k=*/1);

    sassy_matches_print(m);

    sassy_matches_free(m);
    sassy_searcher_free(s);
    return 0;
}
```

Detailed API documentation and build instructions are in
[c/README.md](c/README.md).

---

## Supported alphabets

| Alphabet | Description                               |
| -------- | ----------------------------------------- |
| ASCII    | Exact character equality                  |
| DNA      | Case-insensitive `A C G T`                |
| IUPAC<sup>1</sup>    | Extended IUPAC codes (e.g. `N`, `Y`, `R`) |

<sup>1</sup> See [IUPAC nucleotide codes](https://www.bioinformatics.org/sms/iupac.html) for details.

When using Sassy as Rust library you can also implement a custom `Profile`.

---

## Evals

For the evals see [evals/README.md](evals/README.md).

---

## License

MIT