# Native `precise()` experiment

This fork tests whether one source-text rewrite benefits from native code. It
does **not** change the production shader builder or claim an end-to-end build
speedup. The baseline is nxnative commit
`84e63860acf19f06bdc0133066739f1b755108e5`.

`dmsh_rewrite.precise()` replaces `fast::min(`, `fast::max(` and
`fast::clamp(` at a Unicode word boundary with `precise::...`. Its regular
expression starts with `\b`, preventing the literal-prefix search optimization
available when a pattern starts with `fast::`. This is a small, independently
testable candidate; its share of a real build has not been measured here.

The experiment compares:

- **Upstream:** the existing function imported directly from `dmsh_rewrite`.
- **Python candidate:** a regex with a literal prefix, followed by an equivalent
  boundary check in a callback. This avoids a native dependency but adds a
  Python callback for every candidate match.
- **C++ candidate:** a CPython extension using a literal search and one output
  allocation. It preserves Python's Unicode word-boundary semantics, including
  non-ASCII strings and embedded NULs. It holds the GIL and uses CPython's API;
  rebuild it for each Python version and platform. No binary is committed.

## Run

From the repository root, with CPython 3.10+ headers and a C++17 compiler
(Xcode Command Line Tools on macOS, or a C++ compiler and Python development
headers on Linux):

```sh
python3 scripts/bench_shader_precise.py --build --output /tmp/precise-results.json
python3 dm/gfx/experiments/native_precise/test_precise.py
```

`CXX` can select a compiler. For example, use `CXX=clang++` on a Mac if Python's
configured compiler is unavailable. This builds only the extension; Metal,
Ryujinx and game assets are not needed. Re-run `--build` after changing Python
or the C++ source.

To benchmark your own intermediate MSL sources, keep them outside the repository
and pass their directory:

```sh
python3 scripts/bench_shader_precise.py --build \
  --sources /path/to/local/shaders --limit 100 \
  --output /tmp/precise-local-results.json
```

This selects the first 100 sorted `.metal` files, not a random sample. The JSON
contains only measurements, content hashes and numeric case IDs; it does not
contain local shader text or paths. Use sources from **before** the `precise`
pass to measure representative replacement density: already rewritten output
can make the pass look cheaper than it was in the real build.

## Recorded result

`results_linux.json` records one Linux x86-64 run using CPython 3.12.14 and GCC
14.2, compiled with `-O3`. Seven timing samples per implementation and input
are interleaved in randomized order after warmup and independent batch-size
calibration. All outputs are compared with upstream before timing. The raw
JSON includes individual samples, iteration counts and source hashes.

For the four synthetic 220 KiB inputs (microseconds per call, sample median):

| Input | Upstream | Python candidate | C++ candidate | C++ speedup |
|---|---:|---:|---:|---:|
| No matches | 1,808.2 | 117.2 | 102.1 | 17.71x |
| Sparse matches | 1,810.0 | 193.4 | 111.9 | 16.18x |
| Dense matches | 2,860.9 | 3,188.2 | 255.4 | 11.20x |
| Nonmatching lookalikes | 2,007.2 | 1,247.1 | 190.1 | 10.56x |

The Python candidate is 9.36x faster on the sparse case, but **slower** than
upstream on the dense case. Changing the search strategy explains a substantial
part of the gain; this is not evidence that Python should be removed from the
whole pipeline.

There are 20 benchmark cases: generated inputs from 1 KiB to 1 MiB, plus the
repository's synthetic `test_dmsh_bit0_mask.reduced()` fixture and explicitly
labeled derivatives with inserted calls. None is a measured game corpus.
These timings come from a shared Linux environment, not Apple Silicon, and
cover a single function rather than the full rewrite or shader compilation.

## Validation and remaining work

Six differential tests pass, including 10,000 deterministic randomized cases,
Unicode boundaries, adjacency, long inputs, repeated calls and strings read
from the existing upstream Python test fixtures (4,309 strings in 40 files at
the baseline). The same suite passes with UndefinedBehaviorSanitizer. C++
compilation passes `-Wall -Wextra -Werror`; repository Python lint reports zero
new findings against its baseline. The thread test checks repeated use, not
parallel speedup.

Before considering production integration:

1. Profile a real cold shader build on Apple Silicon and measure this pass's
   fraction of total time, including extraction, translation and Metal.
2. Run both candidates on representative pre-rewrite MSL and compare complete
   outputs; rerun the microbenchmark on that same Mac and corpus.
3. If the end-to-end benefit justifies native build/distribution complexity,
   design integration and run the required shader-chain and game regressions.

No production chain files are modified, so this branch requires no chain-version
bump. Metal compilation, full game builds, macOS execution and gameplay remain
untested. The normal NX2Apple build does not use this experimental extension.
