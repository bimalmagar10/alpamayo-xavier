Run different PhysicalAI clips or timestamps by preparing one matched sample
folder for each inference. A sample contains 16 images, fused vehicle-history
tokens, positional inputs, the image grid and the initial vehicle speed.
Changing only `--images` would continue to use the original clip's history and
speed. `--repeat` repeats the same sample; it does not move through a video.

Update the repository on the H100 side and the Xavier before using these commands.
The new H100 script depends on the updated `h100/a3b_fixtures.py` and
`xavier/alpamayo_xavier/{sample_inputs,preprocess}.py`. Sending the updated source
directories together avoids version mismatches. No engine rebuild is needed for
samples that pass the image-grid and prompt-shape compatibility checks.

On the H100 side, activate your existing reference environment:

```bash
source /mnt/DISCL/work/bthapama/alpamayo-xavier/env.sh
source "$ALPAMAYO_ROOT/alpamayo/ar1_venv/bin/activate"

python "$ALPAMAYO_REPO/h100/a9_prepare_samples.py" \
  --clip 030c760c-ae38-49aa-9ad8-f5650a545d26 \
  --t0-us 6100000 7100000 8100000
```

This selects 6.1, 7.1 and 8.1 seconds in that clip and writes:

```text
$ALPAMAYO_ROOT/samples/
  030c760c-ae38-49aa-9ad8-f5650a545d26__6100000/
    frames/00_cam0_t0.png ... 15_cam3_t3.png
    fixtures/input_ids.npy
    fixtures/position_ids.npy
    fixtures/visual_mask.npy
    fixtures/image_grid_thw.npy
    fixtures/ego_history.npz
    fixtures/meta.json
  030c760c-ae38-49aa-9ad8-f5650a545d26__7100000/
  030c760c-ae38-49aa-9ad8-f5650a545d26__8100000/
```

Use `--out /another/samples/folder` to change the destination. The exporter refuses
to overwrite existing sample folders. Pass more than one ID after `--clip` to
prepare every clip/timestamp combination:

```bash
python "$ALPAMAYO_REPO/h100/a9_prepare_samples.py" \
  --clip FIRST_VALID_CLIP_ID SECOND_VALID_CLIP_ID \
  --t0-us 6100000 7100000
```

The IDs must be real IDs from your dataset. The loader needs sufficient past and
future coverage around each timestamp: it samples 16 ego-history steps and 64
future steps at 0.1 seconds. Cameras contribute four frames ending at `t0`, at
0.1-second spacing, in camera-index order `[0, 1, 2, 6]` (cross-left, front-wide,
cross-right, front-tele). The exporter preserves this order from
[NVIDIA's loader](https://github.com/NVlabs/alpamayo/blob/main/src/alpamayo_r1/load_physical_aiavdataset.py).

Preparation loads the reference model once in the existing environment, as the
fixture exporter already did, but does not run its VLM or export any ONNX graphs.
It still requires host memory for the model (roughly 22 GB of weights plus loading
overhead) and access to the gated PhysicalAI dataset. The existing authenticated
environment and data access route are reused. There is no need to rerun the full
golden-inference or engine-build workflow for every new timestamp.

Copy the new `samples/` directory through your normal cluster → Mac → Xavier
route. For example, on the Mac, after substituting your cluster login:

```bash
rsync -avh --partial --progress \
  YOUR_CLUSTER_LOGIN:/mnt/SHARED-SCRATCH/bthapama/alpamayo-work/samples/ \
  ~/alpamayo-payload/samples/

cd ~/Downloads/all_projects/xavier-alpamayo
bash transfer/push_telemetry.sh 100.70.91.173
rsync -avh --partial --progress ~/alpamayo-payload/samples/ \
  bimal@100.70.91.173:/mnt/ssdhome/models/alpamayo/samples/
```

On Xavier, in the existing `alpamayo-jp5` environment, run one new sample:

```bash
export ALPAMAYO_WORK=/mnt/ssdhome/models/alpamayo
python ~/alpamayo-xavier/xavier/run_alpamayo.py \
  --work "$ALPAMAYO_WORK" \
  --sample "$ALPAMAYO_WORK/samples/030c760c-ae38-49aa-9ad8-f5650a545d26__6100000" \
  --precision fp16 --decode torch --residency auto --repeat 3 \
  --json "$ALPAMAYO_WORK/results/clip_6100000.json"
```

Or run every prepared sample, writing a separate telemetry JSON for each:

```bash
export ALPAMAYO_WORK=/mnt/ssdhome/models/alpamayo
for sample_dir in "$ALPAMAYO_WORK"/samples/*; do
  [ -f "$sample_dir/fixtures/meta.json" ] || continue
  sample_id="${sample_dir##*/}"
  python ~/alpamayo-xavier/xavier/run_alpamayo.py \
    --work "$ALPAMAYO_WORK" --sample "$sample_dir" \
    --precision fp16 --decode torch --residency auto --repeat 1 \
    --json "$ALPAMAYO_WORK/results/$sample_id.json" || break
done
```

The loop starts a fresh process for each sample, so it reloads the decoder for
each sample's first frame. Use `--repeat 3` within each invocation when comparing
first-frame and subsequent-frame performance. Results record the chosen clip,
timestamp, image filenames, sample fixtures, reasoning text and telemetry.

Every sample reuses `WORK/fixtures/embed_tokens.fp16.npy`, the common vocabulary,
the existing engines and PyTorch decoder weights. Sample-specific `v0` is required;
the runner will not silently borrow the original golden clip's speed. Sample
metadata must match the original fixture dimensions, and its exact per-image grid
must match the grid baked into the vision engine, not merely the total token count.
The original grid is read from `WORK/fixtures/image_grid_thw.npy` or, for older
exports, `WORK/golden/inputs.npz`. Keep one of these on the Xavier.

Prepared PNGs and their fixtures form one sample. Do not replace just the PNGs
inside a prepared folder; prepare another clip/timestamp instead.
