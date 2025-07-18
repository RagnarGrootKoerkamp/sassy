# Sassy C bindings

First clone this repo and build using `cargo build --release --features c`. 
Then create your C file, see [example.c](example.c).

**Example usage.**

``` c
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


**Compile executable.**
Make sure `-L` points to the `target/release/` where the `.so` file is.
```bash
gcc -std=c11 -I. example.c \
    -L ../target/release -lsassy -lm \
    -o sassy_example
```

**Running executable.**
```bash
LD_LIBRARY_PATH=../target/release ./sassy_example
```

### Setup
The Rust source for the C bindings is in `src/c.rs` when the `c` feature is
enabled. `cbindgen` can be used to generate the corresponding [`sassy.h`](sassy.h) header:

```bash
cbindgen --config cbindgen.toml --output c/sassy.h
```
