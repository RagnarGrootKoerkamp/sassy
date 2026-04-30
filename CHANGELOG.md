# Changelog

<!-- next-header -->

## git
- `search_many` and `search_texts` now accept any slice of where the elements
  implement `RcSearchable`, not just a slice of references to a type implementing
  `RcSearchable`. This means that in some cases there is no need to create a
  temporary Vec just to hold texts references.

## 0.2.2
- Build GitHub x86-64 binaries for both AVX2 and AVX-512 using [`cargo multivers`](https://github.com/ronnychevalier/cargo-multivers).

## 0.2.1

- **Breaking:** Tracebacks now prefer substitutions over indels, as this is
  more practically useful: [#46](https://github.com/RagnarGrootKoerkamp/sassy/pull/46)
  (Previous behaviour was to prefer matches, and otherwise deletion over
  substitution over insertion.)
- docs: Add Rust and CLI examples for v2 to the readme
- docs: Clarify the tsv output of `sassy search` in the readme.
- docs: Clarify overhang [#37](https://github.com/RagnarGrootKoerkamp/sassy/issues/37)
- feat: Add `sassy grep --v2` for searching based on the algorithm of the v2 paper.
- feat: Add `sassy search --sam` flag to report negative-strand matches and CIGARs in the
  text direction, rather than the default of reverse-complementing them to match
  the pattern direction. [#49](https://github.com/RagnarGrootKoerkamp/sassy/issues/49)
- feat: Add `sassy crispr --pam-length <len>` to allow changing the default of 3.
- python: Add python typing [#43](https://github.com/RagnarGrootKoerkamp/sassy/pull/43)
- misc: Introduce default-enabled `cli` and `diagnostics` features to allow
  pruning dependencies and building for WASM. [#50](https://github.com/RagnarGrootKoerkamp/sassy/issues/50) [#51](https://github.com/RagnarGrootKoerkamp/sassy/pull/51)
  redundant dependencies.
- misc: Refactors to share traceback code between v1 and v2.
- evals: Updates to v1 and v2 evals.


## 0.2.0
The main new feature of this v2 release is `search_encoded_patterns`, written by @RickBeeloo,
for searching many short and equal length patterns (say 23 bp; should be at most
32 or else 64) in parallel. This is 2x or more faster than eg `search_patterns`,
because it packs deltas in the pattern (rather than text) direction, which
removes a lot of overhead in the 'early break check' and makes the 'profile'
much faster.

v2 also adds new variants of the existing `Searcher::search` method:
- `search_patterns`: search multiple similar-length patterns in a single text.
  Useful for searching in short texts where chunking them in pieces via the
  normal `search` has large overhead. This uses one SIMD lane per pattern.
- `search_texts`: search a single pattern in multiple similar-length texts.
  This uses one SIMD lane per text.
- `search_many`: a more high-level wrapper that takes a list of
  patterns and texts and does an all-vs-all search on multiple threads, using
  a user-specified underlying algorithm.
  
Further changes:
- **Breaking**: Cigar output now follows the SAM spec and reversed `I` and `D`
  compared to before. `I` now means the pattern contains a character that is not
  in the text.
- feat: Support AVX512 for pattern tiling.
- feat: moved pretty printing from `bin/grep.rs` to publicly available `Match::pretty_print`.
- feat: `Match` now contains `text_idx` and `pattern_idx` for multi-search variants.
- feat: Add `Searcher::only_best_match` and `Searcher::without_trace`.
- perf: Collect matches into an internal Vec before returning that.
- fix: `sassy grep` would crash on printing reverse complement matches with overhang.
- fix: fix issues with duplicate reported matches in overhang
- misc: debug-printing a `Match` now uses stringified cigar.
- misc: Add python bindings for `search_many`

## 0.1.10
- **bugfix**: `sassy grep` had a bug in the batching logic, making it skip a
  record after every 1MB of input. So for human genomes it would only search
  every other record.
- docs: Recommend to use `Profile::Iupac` instead of `Profile::Dna` in docs and examples. ([#38](https://github.com/RagnarGrootKoerkamp/sassy/issues/38))
- misc: set up `cargo release`

## 0.1.9
- Implement `RcSearchable` for `?Sized` types such as `[u8]`, so that `seach(_, text, _)` now works for `text: &[u8]`, instead of only `text: &&[u8]`.
- Migrate back from `cargo-dist` to a small manual `release.yaml`.
- Abort on panic.

## 0.1.8
- Sassy is now available on bioconda!
- ci: add aarch64-linux target for github releases, drop windows targets
- ci: by default `target-cpu=native`, but `config-portable.yaml` for CI builds
- doc: readme updates for `target-cpu`
- update to `ensure_simd` for compile-time AVX2 check and add a run-time check
  as well.

## 0.1.7
- Improved compile error message when not using `-target-cpu=native`. ([#37](https://github.com/RagnarGrootKoerkamp/sassy/issues/37))
- Added installation section to README, explaining the use of `RUSTFLAGS` and
  minimal required rust version. ([#37](https://github.com/RagnarGrootKoerkamp/sassy/issues/37))
- Assert whether the selected profile supports overhang.
- Add `cargo-dist` for binary release artefacts.
- Move some internal features from feature-gated to `#[doc(hidden)] mod private {}`.

## 0.1.6
- feat: Use `wide` instead of `portable-simd` so that `sassy` now works on
  stable Rust ([#26](https://github.com/RagnarGrootKoerkamp/sassy/issues/26)). It's slightly (<5%) slower and has slightly ugly code, but
  good enough for now.

## 0.1.5
- feat: Add `sassy search`, `sassy filter`, and `sassy grep` ([#35](https://github.com/RagnarGrootKoerkamp/sassy/issues/35), see updated readme).
- perf: Improvements when searching short (len ~16) patterns, by avoiding
  redundant expensive `find_mininima` call.
- perf: Improvements when searching short texts without overhang, by avoiding redundant
  floating point operations.
- misc: Bump `pa-types` to `1.2.0` for `Cigar::to_char_pairs` to conveniently
  iterate over corresponding characters.
- misc: `derive(Clone)` for `Searcher` ([#36](https://github.com/RagnarGrootKoerkamp/sassy/pull/36))
- misc: Bugfix for mixed-case IUPAC input.
- docs: Minor documentation & readme fixes.

## 0.1.4
- Improve docs for `sassy crispr` ([#34](https://github.com/RagnarGrootKoerkamp/sassy/pull/34) by @tfenne).
- Require value for `--max-n-frac` ([#33](https://github.com/RagnarGrootKoerkamp/sassy/pull/33) by @tfenne).
- Check that AVX2 or NEON instructions are enabled; otherwise `-F scalar` is required.
- Non-x86 support: Use `swizzle_dyn` instead of hardcoding `_mm256_shuffle_epi8`.
- Add fallback for non-BMI2 instruction sets; 5-20% slower.
- Update `pa-types` to `1.1.0` for CIGAR output that always includes `1` (eg `1=`).
- Fix/invert `sassy crispr --no-rc` flag.
- Ensure output columns of `sassy crispr` match content ([#31](https://github.com/RagnarGrootKoerkamp/sassy/pull/31) by @tfenne).

## 0.1.3
- Include source code in pypi distribution.

## 0.1.2
- First public release on crates.io and pypi.
