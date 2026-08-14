# Elementwise robustness worklog

Date: 2026-08-14

## Summary

Implemented issue [#6](https://github.com/tensor4all/AlternatingCrossInterpolation.jl/issues/6) on top of the tolerance-scaling fix from PR [#5](https://github.com/tensor4all/AlternatingCrossInterpolation.jl/pull/5). The public `elementwise` return shape remains unchanged.

## References reviewed

- `src/elementwise.jl`, `src/randomtt.jl`, and `test/test_elementwise.jl`
- TensorCrossInterpolation.jl `globalpivotfinder.jl`, `globalsearch.jl`, `cachedtensortrain.jl`, and `tensorci2.jl`
- tensor4all-rs ACI PRs [#591](https://github.com/tensor4all/tensor4all-rs/pull/591), [#609](https://github.com/tensor4all/tensor4all-rs/pull/609), [#617](https://github.com/tensor4all/tensor4all-rs/pull/617), and [#619](https://github.com/tensor4all/tensor4all-rs/pull/619)

## Decisions

- Reuse TensorCrossInterpolation.jl's floating-zone walk rather than maintaining a second implementation.
- Cache immutable input tensor-train evaluations for the full ACI run; rebuild the solution cache once per global search because the solution changes between sweeps.
- Inject global pivots through ACI's existing left/right frames and zero-pad only bonds that can still grow. Bond growth is bounded by both `maxbonddimension` and the algebraic index-space limit.
- Preserve `(solution, ranks, errors)` and absolute-error history semantics. No Rust-specific batched callback or termination-result API was added.
- Keep global-search vocabulary aligned with TensorCrossInterpolation.jl: `nsearchglobalpivot`, `maxnglobalpivot`, and `tolmarginglobalsearch`.
- Preserve PR #5 authorship by retaining its two commits in the implementation branch.

## Verification

- `julia --project=. -e 'using Pkg; Pkg.test()'`: 58/58 tests passed on Julia 1.12.5 with TensorCrossInterpolation 0.9.19.
- `julia +1.10 --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'`: 58/58 tests passed on Julia 1.10.11.
- Documenter doctests: 1/1 passed.
- Added value-based regressions for one-site evaluation, separated-peak recovery, binding bond-cap termination, input validation, and scaled/absolute tolerance behavior.

## Remaining risk

The guard calls the non-exported `TensorCrossInterpolation._floatingzone` helper. Compatibility is therefore pinned to TensorCrossInterpolation 0.9.5 or newer, where the required signature exists. A future public floating-zone API in TensorCrossInterpolation.jl should replace this internal call.
