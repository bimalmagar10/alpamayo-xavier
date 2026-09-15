The results JSON now keeps the existing `precision`, `device`, `runs`, `stages`,
`reasoning_tokens`, and `waypoints` fields and adds schema version 2 telemetry.

From the Mac, send the code and prepared vocabulary:

```bash
cd ~/Downloads/all_projects/xavier-alpamayo
bash transfer/push_telemetry.sh 100.70.91.173
```

On Xavier, in the existing `alpamayo-jp5` environment:

```bash
python ~/alpamayo-xavier/xavier/run_alpamayo.py \
  --work /mnt/ssdhome/models/alpamayo \
  --precision fp16 --decode torch --residency auto \
  --images '/mnt/ssdhome/models/alpamayo/frames/*.png' \
  --repeat 3 \
  --json /mnt/ssdhome/models/alpamayo/results/fp16_telemetry.json
```

The existing inference command also works. No engine rebuild or new Jetson
Python dependency is required. `--vocab PATH` overrides the vocabulary location;
`--telemetry-interval 0` disables background sensor sampling (default: 0.5 seconds).

For different PhysicalAI clips or timestamps, use `--sample` with the matching
images and vehicle-history fixtures. See [preparing and running new samples](NEW_SAMPLES.md).

| JSON field | Meaning |
| --- | --- |
| `environment` | Python, loaded library versions/paths, installed package versions, CUDA build/toolkit/driver API, cuDNN, TensorRT, JetPack/L4T packages, CPU/GPU architecture, power mode and clocks |
| `environment.code.source_sha256` | Small source-file hashes, even on an rsync copy without Git metadata |
| `configuration.selected_stages` | Actual TensorRT/PyTorch backend, requested precision and partitions; PyTorch decode remains FP16 even if the TensorRT precision is INT8 |
| `configuration.tokenizer` | Vocabulary availability, path and SHA256 |
| `runs[].reasoning_tokens` | Count of generated IDs, including the trajectory-start stop token when present |
| `runs[].reasoning_token_ids` / `reasoning_text` | Exact generated IDs and their decoded sequence, retaining special tokens |
| `runs[].reasoning_token_pieces` | Text for each individual token; partial UTF-8 bytes may show replacement characters here while the complete text decodes correctly |
| `runs[].decode_steps` / `stop_reason` | Number of tokens processed into the KV cache, and whether generation stopped at the marker, token limit or was disabled |
| `runs[].frame_wall_ms` | Elapsed frame time including preprocessing, loading, inference, sampling, cleanup and progress logging; excludes setup, final result printing and JSON writes |
| `runs[].stages` / `total_ms` | Synchronized wall time of the original stage scopes; prefill engine loading is accounted under `engine load` |
| `runs[].stage_samples_ms` / `decode_latency` | Individual calls and per-token latency distribution |
| `runs[].telemetry.timing.records` | Nested stage, engine-piece load/execution/release, token preparation/sampling and expert flow-step measurements |
| `runs[].telemetry.timing.exclusive_wall_ms` | Non-overlapping wall-time contributions, which can be summed |
| `runs[].telemetry.timing.uninstrumented_wall_ms` | Remaining frame wall time, such as orchestration, progress printing and probe overhead |
| `runs[].telemetry.sensors` | Per-frame GPU load, clocks, temperatures and individual power rails when readable; stage scopes also contain sensor summaries |
| `runs[].telemetry.counters` | Process CPU time, faults, context switches, Linux process I/O, system swap events and PyTorch allocator retries/OOM counters |
| `runs[].telemetry.memory_checkpoints` | System shared RAM and PyTorch/device memory at frame boundaries, after prefill, during decode residency and with expert loaded |
| `runs[].telemetry.residency` | Stages held before/after the frame and any eviction/retry events |
| `summary` | First-frame and subsequent-frame distributions reported separately; p95 uses linear interpolation, not the maximum |
| `status` | `running`, `complete`, `failed`, or `interrupted`; complete frames are saved atomically after each repeat |

Timing interpretation changed in schema 2: the old runner subtracted CPU loading
time from a CUDA-event interval. Both terms now use the same wall clock.
`stages` includes host overhead and is not pure GPU compute. Use `frame_wall_ms`
for application latency. Nested inclusive records overlap; sum **exclusive**
wall contributions instead. CUDA intervals include stream idle time and host
launch gaps and must not be added across nested scopes.

For example, a large measured load contribution alongside process `read_bytes`
and major faults supports investigating model loading. It does not establish
that disk bandwidth alone is responsible: deserialization, allocation and page
cache state also contribute. Major faults are not automatically swap activity.
`pswpin`/`pswpout` are system-wide page counters, not per-process measurements.

PyTorch memory metrics exclude TensorRT's direct allocations; use the system
memory checkpoints alongside them. Xavier shares CPU/GPU physical memory, so
these values are not independent pools. Missing probes are `null` or explicitly
unavailable. JetPack is never inferred from a guessed L4T mapping. Sensor sampling
does not itself prove throttling, and power rails are never summed into a false
board total. Sensor definitions follow NVIDIA's
[Xavier power-management documentation](https://docs.nvidia.com/jetson/archives/r35.6.0/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonXavierNxSeriesAndJetsonAgxXavierSeries.html).

This telemetry measures time spent in pipeline operations. Establishing whether
an individual CUDA kernel is limited by memory throughput, occupancy or arithmetic
requires a separate Nsight profile. Instrumentation and synchronization have
some overhead; compare measurements with the same settings.

If the vocabulary is absent or invalid, inference still saves IDs and counts;
`reasoning_text` is `null` for nonempty output and its status explains why.
Copy the prepared vocabulary before the run to get text. Existing result files
that contain only counts cannot be used to reconstruct the missing token IDs.

CPU tests (the runner integration cases require CPU PyTorch):

```bash
python -m unittest discover -s tests -v
```

These tests exercise timing accounting, tokenizer handling, optional probes,
JSON checkpointing, and simulated runner control flow. They do not validate
CUDA/TensorRT performance or Alpamayo numerical accuracy on the actual board.
