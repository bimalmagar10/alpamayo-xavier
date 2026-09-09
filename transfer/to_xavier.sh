#!/usr/bin/env bash
# Push the repo and/or the ONNX payload from the Ubuntu host to the Jetson.
#
#   bash to_xavier.sh <xavier-host> repo      just the 58 KB code bundle (start here)
#   bash to_xavier.sh <xavier-host> vision    + vision.onnx and golden/  (~1.6 GB trial)
#   bash to_xavier.sh <xavier-host> all       + prefill, decode, expert, fixtures (~37 GB)
#
#   XAVIER_USER=bimal PAYLOAD=~/alpamayo-payload bash to_xavier.sh 192.168.1.50 vision
#
# Checks the Jetson has NVMe mounted and enough free space BEFORE moving anything,
# then verifies checksums on the far side.
set -euo pipefail

HOST="${1:?usage: to_xavier.sh <xavier-host> [repo|vision|all]}"
WHAT="${2:-repo}"
USER_="${XAVIER_USER:-bimal}"
PAYLOAD="${PAYLOAD:-$HOME/alpamayo-payload}"
REPO_TGZ="${REPO_TGZ:-$PAYLOAD/alpamayo-repo.tgz}"
REMOTE_WORK="/mnt/ssdhome/models/alpamayo"
REMOTE_REPO="/home/$USER_/alpamayo-xavier"
TARGET="$USER_@$HOST"

case "$WHAT" in
  repo)   NEED_GB=1  ;;
  vision) NEED_GB=4  ;;
  all)    NEED_GB=80 ;;   # payload + the engines built from it
  *) echo "unknown mode '$WHAT' (use: repo | vision | all)" >&2; exit 2 ;;
esac

say() { echo; echo "-- $* --"; }
# macOS ships `shasum`, Linux ships `sha256sum`. This script now runs from either.
if command -v sha256sum >/dev/null; then sha256_local() { sha256sum "$1" | cut -d' ' -f1; }
else                                     sha256_local() { shasum -a 256 "$1" | cut -d' ' -f1; }; fi
rsh() { ssh -o BatchMode=yes "$TARGET" "$@"; }

say "reachability"
rsh true || { echo "cannot ssh to $TARGET. Set up a key, or check the IP." >&2; exit 1; }
echo "ssh to $TARGET OK"

say "Jetson storage"
# A missing NVMe mount would silently redirect tens of GB onto the 28 GB eMMC.
SRC="$(rsh "findmnt -n -o SOURCE /mnt/ssdhome 2>/dev/null || true")"
case "$SRC" in
  /dev/nvme*) echo "NVMe OK: $SRC" ;;
  *) echo "ERROR: /mnt/ssdhome on the Jetson is not NVMe (got '${SRC:-nothing}')." >&2
     echo "Mount it before transferring." >&2; exit 1 ;;
esac
AVAIL_KB="$(rsh "df -Pk /mnt/ssdhome | awk 'NR==2{print \$4}'")"
echo "free on /mnt/ssdhome: $((AVAIL_KB / 1024 / 1024)) GiB"

if [ "$((AVAIL_KB / 1024 / 1024))" -lt "$NEED_GB" ]; then
    echo "ERROR: need ~${NEED_GB} GiB for '$WHAT', Jetson has less." >&2; exit 1
fi

say "repo bundle"
[ -f "$REPO_TGZ" ] || { echo "missing $REPO_TGZ" >&2; exit 1; }
rsh "mkdir -p '$REMOTE_REPO' '$REMOTE_WORK'/{onnx,engines,golden,fixtures,results,logs}"
rsync -avh --partial --progress "$REPO_TGZ" "$TARGET:$REMOTE_REPO/"
rsh "cd '$REMOTE_REPO' && tar -xzf $(basename "$REPO_TGZ") && ls"
echo "repo extracted to $REMOTE_REPO"

if [ "$WHAT" = "repo" ]; then
    say "done"
    cat <<MSG
Next, on the Jetson:
  cd $REMOTE_REPO && source ./env.sh
  bash xavier/setup_xavier.sh
  sudo nvpmodel -m 0 && sudo jetson_clocks
  python bench/xavier_hw_probe.py | tee results/hw_probe.txt
MSG
    exit 0
fi

say "payload: $WHAT"
cd "$PAYLOAD"
push() {  # push <local-relative-path> <remote-subdir>
    [ -e "$1" ] || { echo "  skip (absent): $1"; return; }
    rsync -avh --partial --progress "$1" "$TARGET:$REMOTE_WORK/$2"
}
push golden/ ""
if [ "$WHAT" = "vision" ]; then
    for f in onnx/vision.onnx*; do push "$f" onnx/; done
else
    push onnx/ ""
    push fixtures/ ""
fi
[ -f MANIFEST.sha256 ] && rsync -avh "MANIFEST.sha256" "$TARGET:$REMOTE_WORK/"

say "verifying on the Jetson"
# verify.sh tolerates a partial payload only for files listed in the manifest, so
# for the 'vision' trial just checksum what was actually sent.
if [ "$WHAT" = "all" ] && rsh "test -f '$REMOTE_WORK/MANIFEST.sha256'"; then
    rsh "cd '$REMOTE_WORK' && bash '$REMOTE_REPO/transfer/verify.sh' ."
else
    for f in onnx/vision.onnx*; do
        [ -e "$f" ] || continue
        L="$(sha256_local "$f")"
        Rm="$(rsh "sha256sum '$REMOTE_WORK/$f' | cut -d' ' -f1")"
        [ "$L" = "$Rm" ] && echo "  OK   $f" || { echo "  MISMATCH $f" >&2; exit 1; }
    done
fi

say "done"
cat <<MSG
On the Jetson:
  cd $REMOTE_REPO && source ./env.sh
  PRECISION=fp16 bash xavier/build_engines.sh
  python xavier/verify.py --precision fp16
MSG
