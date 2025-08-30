# Changelog

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
