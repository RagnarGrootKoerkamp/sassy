# Sassy Python Bindings

🐍 The python bindings for Sassy

## Installation

**Pip.**
```bash
pip install sassy-rs
```
(as sassy was already taken we chose sassy-rs)

**From source.**
In the root of the repository, run: 
```bash
maturin develop --features python
```
You need Maturin for this, see [maturin](https://github.com/PyO3/maturin):


## Usage

A simple usage is as follows:

``` python
import sassy
pattern = b"ATCGATCG"
text = b"GGGGATCGATCGTTTT"
# alphabet: ascii, dna, uipac
searcher = sassy.Searcher("dna")
matches = searcher.search(pattern, text, k=1)
for i, match in enumerate(matches):
    print(f"Match {i+1}:")
    print(f"    Start: {match.text_start}")
    print(f"    End: {match.text_end}")
    print(f"    Cost: {match.cost}")
    print(f"    Strand: {match.strand}")
    print(f"    CIGAR: {match.cigar}")
```

This finds 3 matches:

``` text
Match 1:
    Start: 4
    End: 12
    Cost: 0
    Strand: +
    CIGAR: 8=
Match 2:
    Start: 6
    End: 14
    Cost: 1
    Strand: -
    CIGAR: 6=X=
Match 3:
    Start: 2
    End: 10
    Cost: 1
    Strand: -
    CIGAR: X7=
```

Further options are `sassy.Searcher(alpha=0.5)` to allow overhang alignments,
and `sassy.Searcher("dna", rc=False)` to disable reverse complements for DNA
or IUPAC strings. `searcher.search` is the simple function to search one pattern
in one text, while `searcher.search_many` takes multiple patterns and multiple
texts and searches each pattern in each text, possibly in multiple threads.

See [sassy/example.py](sassy/example.py) for a larger example.

## Type Hints

This package ships [PEP 561](https://peps.python.org/pep-0561/) type stubs (`sassy/py.typed` + `sassy/sassy.pyi`) so that type checkers and IDEs can provide autocomplete and type checking out of the box.

See [sassy/example_typed.py](sassy/example_typed.py) for a typed example that exercises all public APIs.

To run the typed example with [`uv`](https://docs.astral.sh/uv/):
```console
uv run \
    --with maturin \
    sh -c 'maturin develop && python python/sassy/example_typed.py'
```

To run type checking on the typed example with [`uv`](https://docs.astral.sh/uv/):
```console
uv run \
    --with mypy \
    sh -c 'mypy python/sassy/example_typed.py'
```

**Note on auto-generation.** The `.pyi` stubs are currently maintained by hand.
PyO3's [type stub introspection](https://pyo3.rs/main/type-stub.html) is a work-in-progress and does not yet support the function-based `#[pymodule]` declaration used here.
Maturin does not generate stubs itself — it only packages them.
Until PyO3 introspection matures, the manual stubs are the most reliable approach and should be kept in sync with `src/python.rs` when the API changes.

## Troubleshooting


**1. I could install `sassy-rc` but no modules/functions are found.**

When creating an issue please include the output of `print(dir(sassy))` if you were able to install `sassy-rs` but no functions/modules were found. 

Your output might look like:
```python
['__all__', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__', '__spec__']
```
Whereas it should look like:
```python
['Searcher', '__all__', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__', '__spec__', 'features', 'sassy']
```

**2. Other sassy issues.**

If you were able to install sassy, but have other issues please also add the output of `sassy.features()`.
