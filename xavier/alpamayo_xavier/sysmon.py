"""Read available Jetson clocks, load, temperatures and power rails from sysfs.

Reads use a background thread without sudo or a subprocess per sample. Missing
or unreadable nodes produce unavailable data. These measurements do not by
themselves identify thermal throttling or a causal performance bottleneck.
"""
from __future__ import print_function

import glob
import math
import os
import re
import threading
import time


def _channels():
    """Discover what this board exposes: (key, path, unit, scale to the unit)."""
    found = []

    def add(key, path, unit, scale=1.0):
        if os.path.exists(path):
            found.append((key, path, unit, scale))

    for load_path in ("/sys/devices/gpu.0/load", "/sys/devices/platform/17000000.gv11b/load",
                      "/sys/class/devfreq/17000000.gv11b/device/load"):
        if os.path.exists(load_path):
            add("gpu_load", load_path, "%", 0.1)  # per mille
            break
    for d in sorted(glob.glob("/sys/class/devfreq/*")):
        name = os.path.basename(d)
        tag = ("gpu" if any(x in name for x in ("gv11b", "ga10b", "gpu")) else
               "emc" if "emc" in name or "memory-controller" in name else name)
        add("%s_freq_mhz" % tag, os.path.join(d, "cur_freq"), "MHz", 1e-6)
    for c in sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq")):
        add("%s_freq_mhz" % c.split("/")[-3], c, "MHz", 1e-3)
    for z in sorted(glob.glob("/sys/devices/virtual/thermal/thermal_zone*")):
        kind = None
        try:
            kind = open(os.path.join(z, "type")).read().strip()
        except (OSError, IOError):
            pass
        add("temp_%s_c" % (kind or os.path.basename(z)), os.path.join(z, "temp"), "C", 1e-3)
    # Keep each rail separate. AGX Xavier has TWO INA3221 devices; rail1 alone
    # is not a unique name, and summing nested supply rails would double-count.
    for p in sorted(glob.glob("/sys/bus/i2c/drivers/ina3221*/*/iio:device*/in_power*_input")):
        device = os.path.basename(os.path.dirname(os.path.dirname(p)))
        add("power_%s_%s_w" % (device, os.path.basename(p)), p, "W", 1e-3)
    for p in sorted(glob.glob("/sys/bus/i2c/drivers/ina3221*/*/hwmon/hwmon*/curr*_input")):
        channel = re.search(r"curr(\d+)_input$", p).group(1)
        folder = os.path.dirname(p)
        volt = os.path.join(folder, "in%s_input" % channel)
        device = os.path.basename(os.path.dirname(os.path.dirname(folder)))
        try:
            with open(os.path.join(folder, "in%s_label" % channel)) as f:
                label = re.sub(r"[^a-zA-Z0-9_]+", "_", f.read().strip())
        except OSError:
            label = "rail" + channel
        if os.path.exists(volt):
            found.append(("power_%s_%s_w" % (device, label), (p, volt), "W", 1e-6))
    return found


def _value(path, scale):
    try:
        if isinstance(path, tuple):                 # current and voltage, multiplied
            with open(path[0]) as f:
                a = float(f.read().strip())
            with open(path[1]) as f:
                b = float(f.read().strip())
            value = a * b * scale
        else:
            with open(path) as f:
                value = float(f.read().strip()) * scale
        return value if math.isfinite(value) else None
    except (OSError, IOError, ValueError):
        return None


class Sampler(object):
    """Background sysfs sampling, with named windows so each run can be read alone."""

    def __init__(self, interval=0.5):
        if interval <= 0:
            raise ValueError("sampling interval must be positive")
        self.interval = interval
        self.channels = _channels()
        self.points, self.windows = [], []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    @property
    def available(self):
        return bool(self.channels)

    def start(self):
        if not self.available or self._thread:
            return self
        self._stop.clear()
        self._sample()
        self._thread = threading.Thread(target=self._loop)
        self._thread.daemon = True                  # never hold up interpreter exit
        self._thread.start()
        return self

    def _sample(self):
        values = [_value(p, s) for _, p, _, s in self.channels]
        with self._lock:
            self.points.append((time.perf_counter(), values))

    def _loop(self):
        while not self._stop.wait(self.interval):
            self._sample()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2 * self.interval + 1)
            self._thread = None

    class _Window(object):
        def __init__(self, owner, label):
            self.owner, self.label = owner, label

        def __enter__(self):
            self.t0 = time.perf_counter()
            return self

        def __exit__(self, *_):
            self.owner.windows.append((self.label, self.t0, time.perf_counter()))
            return False

    def window(self, label):
        """`with sampler.window("run 0"):` -- summarised separately at the end."""
        return Sampler._Window(self, label)

    def _summarise(self, lo=None, hi=None):
        with self._lock:
            points = [(ts, row) for ts, row in self.points
                      if (lo is None or ts >= lo) and (hi is None or ts <= hi)]
        out = {}
        for c, (key, _, unit, _s) in enumerate(self.channels):
            vals = [row[c] for _, row in points if row[c] is not None]
            if vals:
                out[key] = dict(unit=unit, samples=len(vals), min=min(vals), max=max(vals),
                                mean=sum(vals) / len(vals), first=vals[0], last=vals[-1])
        return dict(samples=len(points), channels=out)

    def report(self, lo=None, hi=None):
        summary = self._summarise(lo, hi)
        return dict(available=bool(summary["channels"]), interval_s=self.interval,
                    reason=None if summary["channels"] else "no readable sensor samples in this window",
                    sources={k: dict(path=list(p) if isinstance(p, tuple) else p, unit=u)
                             for k, p, u, _ in self.channels},
                    **summary)

    def verdict(self):
        return ["Temperature and clock samples alone do not prove thermal throttling. "
                "Power rails are reported separately and are not summed into board power."]


if __name__ == "__main__":
    import json
    s = Sampler(0.2).start()
    with s.window("demo"):
        time.sleep(1.2)
    s.stop()
    print(json.dumps(s.report(), indent=2))
    print(s.verdict())
