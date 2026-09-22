# DCK version

This directory contains the construction-kit version of dma-is-back. The original Go sources are preserved at their original paths (revision `b47c293533938ec007216fbdc91d17e580b82fd0`), with small asset accessors so both versions use the same embedded resources.

Run the original with `go run ./cmd/dmaisback` and this version with `go run ./dck/cmd/dmaisback` from the repository root.

The choreography and assets remain in this repository. Reusable rendering and
effects come from the published `github.com/olivierh59500/democonstructionkit`
module pinned in `go.mod`. Go downloads the dependencies automatically, including
`github.com/olivierh59500/ym-player v1.0.0` for YM playback.

## Continuous cube effects

The DCK cube preserves the complete displayed vertex positions at every mode boundary, then removes the correction over 0.75 seconds using `geometry.Handoff`. Swing retains its preceding orientation. Tests exercise all five boundaries and verify that sampling does not advance animation.

The original command retains the historical behavior. DCK baseline captures can use `SetSmoothTransitions(false)` before the first update. The default is the corrected behavior.

See the [DCK effect configuration guide](../../../lib/democonstructionkit/docs/EFFECT_OPTIONS.md) for the shared API and examples.
