#!/usr/bin/env python3
"""Summarise a tegrastats log: is this benchmark run trustworthy?

    python xavier/tegrastats_summary.py results/bench_tegrastats.log

Three things invalidate a latency measurement on Jetson, and all three are only
visible here:
  * SWAP moved      -> the run paged to disk; the timings are not steady state
  * GR3D never high -> the GPU was idle or the workload never landed on it
  * temperature ramp with a rising tail -> thermal throttling mid-run
"""
import re
import sys

PATS = dict(
    ram=re.compile(r"RAM (\d+)/(\d+)MB"),
    swap=re.compile(r"SWAP (\d+)/(\d+)MB"),
    gr3d=re.compile(r"GR3D_FREQ (\d+)%"),
    emc=re.compile(r"EMC_FREQ (\d+)%"),
    gpu_c=re.compile(r"GPU@([\d.]+)C"),
    cpu_c=re.compile(r"CPU@([\d.]+)C"),
)


def main(path):
    rows = {k: [] for k in PATS}
    ram_total = swap_total = 0
    n = 0
    with open(path) as f:
        for line in f:
            n += 1
            for k, p in PATS.items():
                m = p.search(line)
                if not m:
                    continue
                rows[k].append(float(m.group(1)))
                if k == "ram":
                    ram_total = int(m.group(2))
                if k == "swap":
                    swap_total = int(m.group(2))
    if not n:
        sys.exit("empty log: %s" % path)

    def stat(k, unit="", scale=1.0):
        v = rows[k]
        if not v:
            return "  %-14s (not present)" % k
        return "  %-14s min %7.1f   mean %7.1f   max %7.1f %s" % (
            k, min(v) * scale, sum(v) / len(v) * scale, max(v) * scale, unit)

    print("%s  --  %d samples (%.0f s at 500 ms)\n" % (path, n, n * 0.5))
    print(stat("ram", "MB of %d" % ram_total))
    print(stat("swap", "MB of %d" % swap_total))
    print(stat("gr3d", "%  GPU utilisation"))
    print(stat("emc", "%  memory controller"))
    print(stat("gpu_c", "C"))
    print(stat("cpu_c", "C"))

    print("\nverdict")
    ok = True
    sw = rows["swap"]
    if sw and max(sw) - min(sw) > 1:
        print("  FAIL  swap moved by %.0f MB -- this run paged to disk, discard it"
              % (max(sw) - min(sw)))
        ok = False
    else:
        print("  ok    swap never moved")

    g = rows["gr3d"]
    if g and max(g) < 50:
        print("  WARN  GR3D peaked at only %.0f%% -- was the work actually on the GPU?" % max(g))
        ok = False
    elif g:
        busy = 100.0 * sum(1 for x in g if x > 50) / len(g)
        print("  ok    GR3D peaked at %.0f%%, above 50%% for %.0f%% of the run" % (max(g), busy))

    t = rows["gpu_c"]
    if len(t) > 20:
        head, tail = sum(t[:10]) / 10, sum(t[-10:]) / 10
        if tail - head > 8:
            print("  WARN  GPU rose %.1f C from start to end -- likely throttling; "
                  "compare p50 against p95" % (tail - head))
            ok = False
        else:
            print("  ok    GPU thermally stable (%.1f -> %.1f C)" % (head, tail))

    print("\n%s" % ("run looks trustworthy" if ok else "treat these timings with caution"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/bench_tegrastats.log")
