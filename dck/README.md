# DCK version

This directory contains the construction-kit version of dma-is-back. The original Go sources are preserved at their original paths (revision `b47c293533938ec007216fbdc91d17e580b82fd0`), with small asset accessors so both versions use the same embedded resources.

Run the original with `go run ./cmd/dmaisback` and this version with `go run ./dck/cmd/dmaisback` from the repository root.

The choreography and assets stay local; reusable rendering and effects live in `../../lib/democonstructionkit`. Second Reality retains its original ST3 music synchronization.
