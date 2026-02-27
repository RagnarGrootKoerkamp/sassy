[![crates.io](https://img.shields.io/crates/v/sassy.svg)](https://crates.io/crates/sassy)
[![Conda version](https://img.shields.io/conda/v/bioconda/sassy?label=bioconda)](https://anaconda.org/bioconda/sassy)
[![PyPI](https://img.shields.io/pypi/v/sassy-rs.svg)](https://pypi.org/project/sassy-rs/)
[![docs.rs](https://img.shields.io/docsrs/sassy.svg?label=docs.rs)](https://docs.rs/sassy)
[![biorXiv preprint](https://img.shields.io/badge/biorXiv-10.1101/2025.07.22.666207-green)](https://doi.org/10.1101/2025.07.22.666207)

# Sassy: SIMD-accelerated Approximate String Matching

Sassy is a library and tool for searching short strings in texts,
a problem that goes by many names:
- approximate string matching,
- pattern matching,
- fuzzy searching.

The motivating application is searching short (length 20 to 100) DNA sequences
in a human genome or e.g. in a set of reads.
Sassy generally works well for patterns/queries up to length 1000,
and supports both ASCII, DNA, and IUPAC.

It has a `grep`-like mode for quick human inspection, as well as `search` to
report locations of matches, and `filter` to only output (non)-matching records.

![gif of `sassy grep`](fig/sassy-grep.gif)

Feature highlights:
- Sassy uses bitpacking and SIMD (both AVX2 and NEON supported).
  Its main novelty is tiling these in the text direction.
- Support for _overhang_ alignments where the pattern extends beyond the text.
- Support for (case-insensitive) ASCII, DNA (`ACGT`), and
  [IUPAC](https://www.bioinformatics.org/sms/iupac.html) (=`ACGT+NYR...`) alphabets.
- Rust library (`cargo add sassy`), binary (`cargo install sassy`, see details below), Python
  bindings (`pip install sassy-rs`), and C bindings (see below).

See **the paper**, [detailed docs on docs.rs](https://docs.rs/sassy/latest/sassy/), and corresponding evals in [evals/](evals/):

> Rick Beeloo and Ragnar Groot Koerkamp.  
> Sassy: Searching Short DNA Strings in the 2020s.  
> bioRxiv, July 2025.
> https://doi.org/10.1101/2025.07.22.666207.

## Installation

### Prebuilt binaries 
See the latest [release](https://github.com/RagnarGrootKoerkamp/sassy/releases).

You can also get these via
``` sh
cargo binstall sassy
```
or via conda/mamba/pixi:

``` sh
conda install -c bioconda sassy
```

### Build from source
``` sh
RUSTFLAGS="-C target-cpu=native" cargo install sassy
```

Sassy uses AVX2 or NEON instructions performance reasons, which requires either
`target-cpu=native` or `target-cpu=x86-64-v3` on x64 machines.
See [this README](https://github.com/ragnargrootkoerkamp/ensure_simd) for details and [this
blog](https://curiouscoding.nl/posts/distributing-rust-simd-binaries/) for background.
The same restrictions apply when using the sassy library in a larger project.

Sassy requires Rust 1.91 or newer. Get it via `rustup update`. (Switch to
[rustup](https://rustup.rs) when your system installation is too old).

## Usage

Sassy can be used via the CLI, or as Rust, Python, or C library.

### 0. Rust library

The library can be used to search for ASCII or DNA strings.
A larger example can be found in [`src/lib.rs`](src/lib.rs).

```rust
// cargo add sassy
use sassy::{Searcher, Match, profiles::Iupac, Strand};

let pattern = b"ATCG";
let text = b"AAAATTGAAA";
let k = 1;

// The Iupac profile supports N and YR... characters.
// If you are sure you only have ACGT input, then `profiles::Dna` is slightly faster.
let mut searcher = Searcher::<Iupac>::new_fwd();
let matches = searcher.search(pattern, &text, k);

assert_eq!(matches.len(), 1);

assert_eq!(matches[0].text_start, 3);
assert_eq!(matches[0].text_end, 7);
assert_eq!(matches[0].cost, 1);
assert_eq!(matches[0].strand, Strand::Fwd);
assert_eq!(matches[0].cigar.to_string(), "2=1X1=");
```

### 1. Command-line interface (CLI)

The CLI can be used via:
1. `sassy grep`: to show nicely coloured output.
2. `sassy search`: to write a `.tsv` of matching locations.
3. `sassy filter`: to write a `.fasta`/`.fastq` of (non)-matching records.
4. `sassy crispr`: to search for CRISPR guides.

`grep`, `search`, and `filter` all take the same arguments, and are implemented
by forwarding to `grep`. Thus, they can all be combined via e.g.

```sh
sassy grep -p ACGTCAAACCTA -k 3 --matches matches.tsv --output filtered.fastq reads.fastq.gz
```

#### 1.1: Grep for a pattern

**Search a pattern** `ATGAGCA` in `text.fasta` with ≤1 edit:
```bash
sassy search --pattern ATGAGCA -k 1 text.fasta
```
or search all records of a fasta file with `--pattern-fasta <fasta-file>` instead of `--pattern`.

The `grep` output is coloured:
- green shows matching characters,
- orange shows mismatches,
- red shows deleted characters (in pattern but not in text),
- blue shows inserted characters (in text but not in pattern).
![screenshot of sassy grep output](fig/grep.png)

#### 1.2: TSV output for matches

```sh
sassy search -p GTACAGAAACGAGCGGATGGAAAGAGTAGTGAGCGCCTCGCG -k 2 reads.fa > matches.tsv
# or
sassy search -p GTACAGAAACGAGCGGATGGAAAGAGTAGTGAGCGCCTCGCG -k 2 reads.fa --matches matches.tsv
# or
sassy grep   -p GTACAGAAACGAGCGGATGGAAAGAGTAGTGAGCGCCTCGCG -k 2 reads.fa --matches matches.tsv
```
gives `.tsv` output like this:

```tsv
pat_id	text_id	cost	strand	start	end	match_region	cigar
pattern	AC_000001.1__1_1	0	+	6	48	GTACAGAAACGAGCGGATGGAAAGAGTAGTGAGCGCCTCGCG	42=
pattern	AC_000001.1__1_35	0	+	897	939	GTACAGAAACGAGCGGATGGAAAGAGTAGTGAGCGCCTCGCG	42=
pattern	AC_000001.1__1_49	1	+	866	908	GTACAGAAACGAGCGGATGGAAAGAGTAGTGAGCGCCGCGCG	37=1X4=
pattern	AC_000001.1__1_64	0	-	1267	1309	GTACAGAAACGAGCGGATGGAAAGAGTAGTGAGCGCCTCGCG	42=
pattern	AC_000001.1__1_67	0	+	600	642	GTACAGAAACGAGCGGATGGAAAGAGTAGTGAGCGCCTCGCG	42=
pattern	AC_000001.1__1_68	0	-	1826	1868	GTACAGAAACGAGCGGATGGAAAGAGTAGTGAGCGCCTCGCG	42=
pattern	AC_000001.1__1_78	3	-	4381	4425	GTACAGAAACGAGCGGATGGAAAATAGTAGTGAGCGGCCTCGCG	23=1X1I10=1I8=
pattern	AC_000001.1__1_92	0	-	6554	6596	GTACAGAAACGAGCGGATGGAAAGAGTAGTGAGCGCCTCGCG	42=
pattern	AC_000001.1__1_94	0	-	6413	6455	GTACAGAAACGAGCGGATGGAAAGAGTAGTGAGCGCCTCGCG	42=
pattern	AC_000001.1__1_115	2	+	2091	2131	GTACAGAAACGAGCATGGAAAGAGTAGTGAGCGCCTCGCG	14=2D26=
pattern	AC_000001.1__1_118	0	-	3062	3104	GTACAGAAACGAGCGGATGGAAAGAGTAGTGAGCGCCTCGCG	42=
pattern	AC_000001.1__1_123	0	+	1416	1458	GTACAGAAACGAGCGGATGGAAAGAGTAGTGAGCGCCTCGCG	42=
pattern	AC_000001.1__1_127	0	+	27	69	GTACAGAAACGAGCGGATGGAAAGAGTAGTGAGCGCCTCGCG	42=
```

#### 1.3: Filter matching records
```sh
sassy filter -p GTACAGAAACGAGCGGATGGAAAGAGTAGTGAGCGCCTCGCG -k 2 reads.fq > filtered.fq
# or
sassy filter -p GTACAGAAACGAGCGGATGGAAAGAGTAGTGAGCGCCTCGCG -k 2 reads.fq -o filtered.fq
# or
sassy grep   -p GTACAGAAACGAGCGGATGGAAAGAGTAGTGAGCGCCTCGCG -k 2 reads.fq -o filtered.fq
```
Writes a file containing only matching records. Use `--invert` to only
write non-matching records.

#### 1.4: CRISPR off-target search

Search for one or more guides in `guides.txt`:
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
start, but exclusive of the end), and `start` is always less than `end`
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
