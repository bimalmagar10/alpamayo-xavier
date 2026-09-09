# Moving the payload: cluster → Mac → USB → Ubuntu → Xavier

~37 GB, four hops, no direct route from the university cluster to the Jetson.

## Before anything: send vision first

Do **not** move 37 GB on faith. `vision.onnx` is 1.16 GB and exercises the entire
toolchain — transfer, engine build, numeric verification. If TensorRT 8.5 cannot
build a Qwen3-VL vision tower on sm_72, you want to discover that after moving
1 GB, not 37.

```bash
# hop 1-4 with just these three files
onnx/vision.onnx  onnx/vision.onnx.data  MANIFEST.sha256
# then on the Jetson
PRECISION=fp16 bash xavier/build_engines.sh          # builds whatever is present
python xavier/verify.py --precision fp16             # vision stage only
```

Only once that passes is the big transfer worth starting.

## Hop 1 — cluster → Mac (pull, don't push)

The cluster blocks outbound connections to your Jetson, but your Mac can reach the
cluster. So pull:

```bash
mkdir -p ~/alpamayo-payload && cd ~/alpamayo-payload
rsync -avh --partial --progress \
    <user>@<cluster>:/mnt/SHARED-SCRATCH/bthapama/alpamayo-work/{onnx,fixtures,golden,MANIFEST.sha256} .
bash /path/to/repo/transfer/verify.sh .
```

`--partial` matters: a 37 GB transfer over a university network will drop at least
once, and you do not want to restart. Re-running the same command resumes.

## Hop 2 — Mac → USB

**The stick must be exFAT.** FAT32 caps a single file at 4 GB and
`prefill.onnx.data` is ~14 GB; the copy fails partway with a misleading error.
exFAT is native read/write on macOS and on Linux kernels ≥ 5.4. Use a 64 GB stick
or larger.

```bash
diskutil list                                   # find the stick, e.g. /dev/disk4
diskutil eraseDisk ExFAT ALPAMAYO /dev/disk4    # DESTROYS the stick's contents

cp -Rv ~/alpamayo-payload/. /Volumes/ALPAMAYO/
bash /path/to/repo/transfer/verify.sh /Volumes/ALPAMAYO
diskutil eject /dev/disk4
```

Verify *before* ejecting — macOS buffers writes aggressively, and a checksum pass
forces every byte back off the stick.

## Hop 3 — USB → Ubuntu host

```bash
lsblk                                           # confirm it mounted
mkdir -p ~/alpamayo-payload && cd ~/alpamayo-payload
cp -rv /media/$USER/ALPAMAYO/. .
bash /path/to/repo/transfer/verify.sh .
```

If the stick does not mount, install `exfatprogs` (kernels ≥ 5.4 have the driver
built in; older ones also need `exfat-fuse`).

## Hop 4 — Ubuntu → Jetson

Same LAN, so rsync directly:

```bash
rsync -avh --partial --progress ~/alpamayo-payload/onnx/ \
    bimal@<xavier-ip>:/mnt/ssdhome/models/alpamayo/onnx/
rsync -avh --partial --progress \
    ~/alpamayo-payload/{fixtures,golden,MANIFEST.sha256} \
    bimal@<xavier-ip>:/mnt/ssdhome/models/alpamayo/
```

Then on the Jetson:

```bash
cd /mnt/ssdhome/models/alpamayo && bash ~/alpamayo-xavier/transfer/verify.sh .
df -h /mnt/ssdhome /            # confirm nothing landed on the 28 GB eMMC
```

## Jetson disk: build incrementally

Engines come out roughly the size of their graphs, so ONNX + engines together peak
near 70 GB against ~89 GB free — too tight once TensorRT wants build workspace.
Build one, confirm it loads, delete its ONNX, move to the next:

| Order | ONNX | Engine | Why this order |
|---|---:|---:|---|
| vision | 1.16 GB | ~1.2 GB | Cheapest way to find out TRT 8.5 can handle the graphs |
| expert | 4.60 GB | ~4.6 GB | Second smallest; exercises the fp32 island in `action_in_proj` |
| prefill | 14.34 GB | ~14 GB | The stage that dominates the latency budget |
| decode | 15.17 GB | ~15 GB | Largest; leave for last |

That keeps the peak near 45 GB.

## Verify at every hop

`verify.sh` works on macOS (`shasum`) and Linux (`sha256sum`). Four hops means four
chances to corrupt a byte, and a damaged `.data` file does not announce itself — it
surfaces much later as an incomprehensible TensorRT build error.
