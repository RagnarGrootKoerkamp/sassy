# Changelog

## 0.1.7
- Improved compile error message when not using `-target-cpu=native`. (#37)
- Added installation section to README, explaining the use of `RUSTFLAGS` and
  minimal required rust version. (#37)
- Assert whether the selected profile supports overhang.
- Add `cargo-dist` for binary release artifacts.

## 0.1.6
- feat: Use `wide` instead of `portable-simd` so that `sassy` now works on
  stable Rust (#26). It's slightly (<5%) slower and has slightly ugly code, but
  good enough for now.

## 0.1.5
- feat: Add `sassy search`, `sassy filter`, and `sassy grep` (#35, see updated readme).
- perf: Improvements when searching short (len ~16) patterns, by avoiding
  redundant expensive `find_mininima` call.
- perf: Improvements when searching short texts without overhang, by avoiding redundant
  floating point operations.
- misc: Bump `pa-types` to `1.2.0` for `Cigar::to_char_pairs` to conveniently
  iterate over corresponding characters.
- misc: `derive(Clone)` for `Searcher` (#36)
- misc: Bugfix for mixed-case IUPAC input.
- docs: Minor documentation & readme fixes.

## 0.1.4
- Improve docs for `sassy crispr` (#34 by @tfenne).
- Require value for `--max-n-frac` (#33 by @tfenne).
- Check that AVX2 or NEON instructions are enabled; otherwise `-F scalar` is required.
- Non-x86 support: Use `swizzle_dyn` instead of hardcoding `_mm256_shuffle_epi8`.
- Add fallback for non-BMI2 instruction sets; 5-20% slower.
- Update `pa-types` to `1.1.0` for CIGAR output that always includes `1` (eg `1=`).
- Fix/invert `sassy crispr --no-rc` flag.
- Ensure output columns of `sassy crispr` match content (#31 by @tfenne).

## 0.1.3
- Include source code in pypi distribution.

## 0.1.2
- First public release on crates.io and pypi.
