# analysis/

Figures for the Alpamayo-on-Xavier pipeline, in the style a conference paper
would accept: serif type at print size, vector PDF, no colour-only encodings,
and every number traceable to the run that produced it.

```
analysis/
  alpamayo_figs/
    style.py    NeurIPS-width page geometry, palette, panel labels, saving
    facts.py    every measured number this analysis may plot, with provenance
    data.py     locating and loading golden/, refs/, results JSON
  fig01_why_golden.py
  fig02_export_sizes.py
  fig03_pieces_and_weightmap.py
  fig04_pipeline.tex        TikZ; build with build_latex.sh
  build_latex.sh
  fig05_latency.py          reads results/xavier-results/fp16_telemetry.json
  check_export_weights.py   proves prefill and decode carry the same bytes
  figures/      output, .pdf + .png
```

## Running

Figures that need the golden rollout must run where `golden/` lives. That is the
H100, and also this Mac — `~/alpamayo-payload/golden/` is part of the payload, so
`fig01` runs locally with no cluster access.

```bash
export ALPAMAYO_ROOT=/mnt/SHARED-SCRATCH/bthapama/alpamayo-work   # or pass --root
python analysis/fig01_why_golden.py
```

An explicit `--root` is never quietly replaced by a fallback: if the path has no
`golden/`, the script stops rather than plotting some other machine's data.

Needs only `numpy` and `matplotlib`. The backend is forced to `Agg`, so it runs
over SSH with no display.

To iterate on layout without the cluster:

```bash
python analysis/fig01_why_golden.py --synthetic
```

`--synthetic` fabricates arrays with roughly the right shape so the plotting code
can be exercised, and stamps **SYNTHETIC PREVIEW — not measured data** across the
figure. That stamp is not decoration: it is there so a preview can never be
pasted into a talk by accident. Never present a stamped figure.

## The two rules

**1. Numbers live in `facts.py`, not in figure scripts.** Anything a figure
quotes rather than computes goes there once, with a `source` naming the command
or log line it came from, and a `kind`:

| kind | meaning |
|---|---|
| `measured` | read off an instrument on the machine it describes |
| `derived` | arithmetic on measured values, with the arithmetic stated |
| `estimated` | a model — FLOP counts, roofline — **must be labelled in the figure** |
| `spec` | a datasheet constant |

If two figures disagree about a quantity, that is a bug in `facts.py`, not a
matter of taste.

**2. Compute from data where you can.** `fig01` reads every shape, size and
value range out of `golden/*.npz` at run time rather than quoting them, so the
figure cannot drift away from the file it describes. Doing this caught a real
error: the residual stream at layer 0 peaks at **15.4**, not the 7,517 that had
been repeated in earlier write-ups. The measured story is sharper anyway — 99.9%
of the tensor stays under 44 while the maximum reaches 26,240, so the overflow
comes from roughly one element in 2,400 and is invisible in any summary
statistic.

## fig01 — why the golden run is recorded, and what it stores

Two panels, one claim each.

**(a) Agreement with the golden run, stage by stage.** Linear axis, 0 to 1, so
it reads at a glance. The translated pipeline sits on 1.0 across all five
stages. The same pipeline carrying the fp16 RMSNorm overflow this project
shipped is *identical* through pixels and vision, then drops to 0 at prefill.
Both versions still emit a trajectory — that is the point. Only the per-stage
comparison says which stage is wrong, and the position of the drop is the
diagnosis.

**(b) What is stored, and the number that matters for each component.** Two
columns, one row per component, no grouping — a header band, a rule under it, a
hairline between rows and a rule at the foot, matching the tables in the
project's other write-ups. Every dimension and value is read out of
`golden/*.npz` at run time, so the table cannot drift from the file.

The right-hand cell carries whichever is informative: a shape where the shape is
the point (`3 006 × 4 096`), a magnitude where the magnitude is
(`|x| 15.4 → 26 240`), a physical quantity where that is what matters
(`57.1 m over 6.4 s`). No LaTeX toolchain is required — the maths is matplotlib
mathtext, so the script runs on a bare cluster node.

The script also prints the full inventory of the golden set — every key, shape,
dtype and max |x| — which is the table to have open when someone asks a question
the figure does not answer.

**A gap it surfaces:** the sampled reasoning is *not* recorded. `a1_golden.py`
prints the chain-of-causation trace but never saves it, so there is no golden
counterpart for the text the board now produces. The table says so in red.

## fig02 — what the export costs, per component

One panel, one grouped bar chart. Four components on x, two bars each: the size
in the original checkpoint and the size on disk after `torch.onnx.export`, with
the parameter count above each pair. Everything is measured from the files.

| component | parameters | checkpoint | exported |
|---|---:|---:|---:|
| vision | 576 388 336 | 1.153 GB | 1.157 GB |
| prefill | 7 583 810 560 | 15.168 GB | 15.820 GB |
| decode | *the same 7.584 B* | 0 | 15.169 GB |
| expert | 2 281 782 272 | 4.564 GB | 4.564 GB |
| **total** | **10.442 B** | **20.885 GB** | **36.710 GB** |

Decode has no bar on the left, and that is the figure's point. See the note on
tracing in `check_export_weights.py` below.

Two things the measurement turned up that were not obvious:

- `prefill.onnx` carries **651 MB of inline constants** — 36 copies of the
  3 006 × 3 006 fp16 causal mask, 18.07 MB each, one per layer
  (36 × 18 072 072 = 650 594 592 B, and the file is 651 349 507 B). It is also
  why each prefill piece is ~54 MB: three layers, three masks.
- Parameters from `bytes / 2` agree with the exact element count from the weight
  map to 0.004%, so that shortcut is safe for the two graphs the map does not
  cover.

## fig03 — cutting the graphs up, and reading the weights back out

**Left**, the twenty-eight pieces. Three horizontal bars, one block per piece,
every block drawn to its real weight size from `pieces.json`:

| graph | pieces | layers each | smallest | largest |
|---|---:|---:|---:|---:|
| prefill | 13 | 3 | 1.16 GB | 1.28 GB |
| decode | 10 | 4 | 1.28 GB | 1.54 GB |
| expert | 5 | 9 | 16 kB | 1.14 GB |

Myelin fuses a whole transformer and then demands a single allocation larger than
all its weights, so a 15 GB graph cannot be built on a 32 GB board. Cutting at
layer boundaries into ~1.2–1.5 GB pieces is what makes the build possible at all.
The expert's head piece is only a 16 kB norm — it has no `lm_head`, it emits a
velocity — so it is invisible at this scale and the bar says so.

**Right**, how decode gets those weights without TensorRT. The strip is
`prefill.onnx.data`, one slice per layer plus `lm_head`; below it, one layer
expanded into its eleven tensors at their real byte shares:

| tensor | shape | bytes | share |
|---|---|---:|---:|
| gate, up, down | 4096 × 12288 (×3) | 100 663 296 each | 26.1% each |
| q, o | 4096 × 4096 | 33 554 432 each | 8.7% each |
| k, v | 4096 × 1024 | 8 388 608 each | 2.2% each |
| norms (×4) | 4096 / 128 | 16 384 / 512 | ~0 |

385.9 MB per layer, × 36 + head = 15.17 GB. The board memory-maps the file and
builds the layer stack straight out of it — nothing exported, nothing copied.

## fig04 — the two pipelines

The only figure drawn in LaTeX rather than matplotlib, because it is a block
diagram and TikZ places boxes and elbow arrows better than anything else.

```bash
bash analysis/build_latex.sh fig04_pipeline     # or: bash analysis/build_latex.sh
```

Needs `pdflatex` and the `mathptmx` (Times) package; `ghostscript` is optional
and only produces the PNG preview beside the PDF. The PDF is the artefact to use
— it drops into a paper or a slide at any scale without resampling.

**Top**, what happens once, off the board: checkpoint → golden → export →
fp16-safe rewrite → split → fp32 proof → build → verify, as one chain.

**Bottom**, what happens every frame, on the board: images → preprocess →
vision → assemble → prefill → decode → expert → waypoints, with the shared KV
cache underneath — written by prefill, read and written by decode, read by the
expert.

The only lines crossing between the two panels are the two dashed arrows, and
they are the point of the figure: the fixtures and the weight map are produced
once by the translation and consumed by every frame.

## fig05 — where a frame's time goes

Six panels, every number read out of `fp16_telemetry.json` at run time.

```bash
python analysis/fig05_latency.py
python analysis/fig05_latency.py --json path/to/other.json
```

**(a)** End-to-end latency decomposed into the telemetry's *exclusive* wall
scopes, which sum to the measured frame wall exactly — nothing double counted,
nothing hidden. **(b)** the per-stage breakdown. **(c)** decode latency for every
generated token. **(d)** engine load against the bytes the process actually read.
**(e)** GPU utilisation with the clock and thermal state. **(f)** memory at the
six checkpoints of a frame.

| repeat | wall | load | compute | release | preprocess | tokens |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 146.8 s | 115.7 | 23.9 | 4.7 | 1.9 | 21 |
| 2 | 91.2 s | 61.9 | 22.4 | 4.6 | 1.8 | 18 |
| 3 | 91.7 s | 60.8 | 24.0 | 4.7 | 1.8 | 21 |

The three runs are `--repeat 3` over **one clip**, not three moments of driving:
the same 16 images and the same 3,006-token prompt each time. Two things differ.
Repeat 1 is process-cold and loads decode's weights too (36.8 GB read against
21.2 GB); and one CUDA generator is seeded once, its state advancing across
repeats, so the sampled reasoning differs — 21, 18 and 21 tokens, diverging at
token 13 into "encroaching into the lane", "on the right" and "encroaching on
the lane". Same decision, different wording, and that is the whole difference in
the decode bars.

Three things the exclusive decomposition shows that the stage totals hide:

- **Releasing engines costs 4.7 s a frame.** Of prefill's 10.8 s, only 8.37 s is
  execution; 2.35 s is releasing engines as it streams. The expert spends a
  further 1.68 s being torn down.
- **Loading is I/O, not compute.** Bytes read scale linearly with load time at
  **336 MB/s** — frame 1 reads 36.8 GB, frames 2 and 3 read 21.2 GB each,
  the difference being decode's weights that residency keeps.
- **The GPU is idle for most of the frame.** Mean utilisation 19%, 31%, 33%,
  with the clock pinned at 1377 MHz and temperatures between 48.5 and 59.5 °C —
  so the timings are not throttled, the board is simply waiting on the disk.

The script prints the caveats recorded in the telemetry itself, including that
bytes-read over load-time is achieved read throughput, not the drive's bandwidth.

## check_export_weights.py — are prefill and decode really the same weights?

Everything needed is local; no cluster. Three levels of evidence, cheapest first:

```bash
python analysis/check_export_weights.py --structure-only   # seconds
python analysis/check_export_weights.py --sample 65536     # 64 kB per tensor
python analysis/check_export_weights.py                    # every byte, ~30 GB
```

Result on the payload: 326 external tensors, identical role, shape, dtype, offset
and length; both `.onnx.data` files 15 168 200 704 bytes; the recorded ranges tile
**100.0000%** of each file with zero bytes unaccounted; and **0 tensors differ**
on a full byte comparison. The 72 remaining weights are the 128- and
4096-element norms, which `torch.onnx` keeps inline in the `.onnx` — 36 864 bytes
in total, compared as base64.

This is what licenses two load-bearing decisions: decode reading prefill's
weights file in PyTorch, and decode's own ten engines being built and then
discarded.

## Adding a figure

Copy `fig01_why_golden.py`'s skeleton: `--root`, `--outdir`, `--synthetic`, the
drawing split into small functions taking `(ax, ...)`, and a closing block that
prints both the data it read and the provenance of anything it asserted. That
printout is the figure's audit trail — keep it with the slide.

One figure should make one claim. If a second claim needs a second axis, it
probably needs a second figure.
