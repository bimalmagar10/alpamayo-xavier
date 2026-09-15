"""Wall-clock attribution and optional CUDA intervals, without profiler dependencies.

Exclusive wall times partition elapsed host time; nested scopes are never added
twice. CUDA intervals include stream idle/launch gaps and are not GPU busy time.
"""
import contextlib
import datetime
import json
import os
import resource
import tempfile
import time


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _numbers(path):
    out = {}
    try:
        with open(path) as f:
            for line in f:
                parts = line.replace(":", "").split()
                if len(parts) >= 2:
                    try:
                        out[parts[0]] = int(parts[1]) * (1024 if parts[-1] == "kB" else 1)
                    except ValueError:
                        pass
    except OSError:
        pass
    return out


def snapshot(torch=None):
    usage = resource.getrusage(resource.RUSAGE_SELF)
    mem = _numbers("/proc/meminfo")
    vm = _numbers("/proc/vmstat")
    status = _numbers("/proc/self/status")
    out = dict(monotonic_s=time.perf_counter(),
               process=dict(cpu_user_s=usage.ru_utime, cpu_system_s=usage.ru_stime,
                            minor_faults=usage.ru_minflt, major_faults=usage.ru_majflt,
                            voluntary_switches=usage.ru_nvcsw, involuntary_switches=usage.ru_nivcsw,
                            rss_bytes=status.get("VmRSS"), swap_bytes=status.get("VmSwap")),
               process_io=_numbers("/proc/self/io"),
               system_memory={k: mem.get(k) for k in
                              ("MemTotal", "MemAvailable", "MemFree", "Cached", "SwapTotal", "SwapFree")},
               system_vm={k: vm.get(k) for k in ("pswpin", "pswpout", "pgmajfault")})
    if torch is not None:
        try:
            free, total = torch.cuda.mem_get_info()
            stats = torch.cuda.memory_stats()
            out["cuda"] = dict(device_free_bytes=free, device_total_bytes=total,
                               torch_allocated_bytes=torch.cuda.memory_allocated(),
                               torch_reserved_bytes=torch.cuda.memory_reserved(),
                               torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                               torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                               allocation_retries=stats.get("num_alloc_retries"),
                               out_of_memory_events=stats.get("num_ooms"))
        except Exception as exc:
            out["cuda"] = dict(error=str(exc))
    return out


def counter_delta(before, after):
    out = {}
    counters = {"process": ("cpu_user_s", "cpu_system_s", "minor_faults", "major_faults",
                            "voluntary_switches", "involuntary_switches"),
                "process_io": ("read_bytes", "write_bytes", "rchar", "wchar"),
                "system_vm": ("pswpin", "pswpout", "pgmajfault"),
                "cuda": ("allocation_retries", "out_of_memory_events")}
    for group, keys in counters.items():
        a, b = before.get(group, {}), after.get(group, {})
        out[group] = {k: b[k] - a[k] for k in keys
                      if isinstance(a.get(k), (float, int)) and isinstance(b.get(k), (float, int))}
    return out


def distribution(values):
    values = sorted(values)
    def percentile(q):
        pos = (len(values) - 1) * q
        lo = int(pos)
        return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (pos - lo)
    return (dict(count=len(values), min_ms=values[0], mean_ms=sum(values) / len(values),
                 p50_ms=percentile(.5), p95_ms=percentile(.95), max_ms=values[-1])
            if values else dict(count=0))


class Trace(object):
    def __init__(self, torch=None, clock=time.perf_counter):
        self.torch, self.clock = torch, clock
        self.reset()

    def reset(self):
        self.origin = self.clock()
        self.records, self.stack = [], []
        self.stages = {}

    @contextlib.contextmanager
    def scope(self, kind, stage=None, cuda=False, sync=False, **detail):
        record = dict(kind=kind, stage=stage, detail=detail, depth=len(self.stack),
                      start_ms=(self.clock() - self.origin) * 1000, child_wall_ms=0.0)
        events = None
        if cuda and self.torch is not None:
            events = (self.torch.cuda.Event(enable_timing=True),
                      self.torch.cuda.Event(enable_timing=True))
            events[0].record()
        start = self.clock()
        self.stack.append(record)
        self.records.append(record)
        try:
            yield record
        except BaseException as exc:
            record["error"] = type(exc).__name__ + ": " + str(exc)
            raise
        finally:
            if events:
                events[1].record()
            if sync and self.torch is not None:
                self.torch.cuda.synchronize()
            record["wall_ms"] = (self.clock() - start) * 1000
            record["exclusive_wall_ms"] = max(0.0, record["wall_ms"] - record.pop("child_wall_ms"))
            if events:
                record["_events"] = events
            self.stack.pop()
            if self.stack:
                self.stack[-1]["child_wall_ms"] += record["wall_ms"]

    @contextlib.contextmanager
    def __call__(self, stage):
        first = len(self.records)
        with self.scope("stage", stage, cuda=stage != "engine load", sync=True) as rec:
            yield rec
        # Sequential prefill loads inside run(); use the SAME wall clock for both
        # terms instead of subtracting CPU time from a CUDA-event interval.
        nested_load = (sum(r["wall_ms"] for r in self.records[first + 1:]
                           if r["kind"] == "engine_load") if stage != "engine load" else 0.0)
        self.stages.setdefault(stage, []).append(max(0.0, rec["wall_ms"] - nested_load))
        if nested_load:
            self.stages.setdefault("engine load", []).append(nested_load)

    def observer(self, kind, stage, **detail):
        # Sequential execution synchronizes before close(), so teardown is not
        # incorrectly blamed for waiting on the preceding GPU computation.
        sequential = detail.get("sequential", False)
        return self.scope(kind, stage, cuda=kind == "engine_execute",
                          sync=kind == "engine_execute" and sequential, **detail)

    def total(self):
        return sum(sum(v) for v in self.stages.values())

    def report(self):
        total = self.total()
        print("\n%-26s %10s %8s %9s" % ("stage", "ms", "share", "calls"))
        for name, vals in self.stages.items():
            ms = sum(vals)
            print("%-26s %10.1f %7.1f%% %9d" % (name, ms, 100 * ms / total if total else 0, len(vals)))
        print("%-26s %10.1f" % ("STAGE TOTAL", total))
        return total

    def export(self, frame_wall_ms):
        exclusive = {}
        records = []
        for rec in self.records:
            row = {k: v for k, v in rec.items() if k != "_events"}
            if "_events" in rec:
                a, b = rec["_events"]
                row["cuda_interval_ms"] = a.elapsed_time(b)
            key = rec["kind"] + (":" + rec["stage"] if rec["stage"] else "")
            exclusive[key] = exclusive.get(key, 0.0) + rec["exclusive_wall_ms"]
            records.append(row)
        accounted = sum(exclusive.values())
        return dict(frame_wall_ms=frame_wall_ms, records=records,
                    exclusive_wall_ms=exclusive,
                    uninstrumented_wall_ms=max(0.0, frame_wall_ms - accounted),
                    interpretation="Exclusive wall scopes do not overlap. CUDA intervals include "
                    "stream idle/host launch gaps; they are not kernel busy time. Sampling and "
                    "synchronization add overhead. Frame wall time excludes final result printing "
                    "and JSON writing; progress logging inside the frame is included.")


def observations(timing, counters):
    """Measured contributors, not unsupported claims of a causal GPU bottleneck."""
    ranked = sorted(timing["exclusive_wall_ms"].items(), key=lambda p: p[1], reverse=True)
    notes = []
    if ranked:
        notes.append(dict(type="measured", message="Largest instrumented wall-time contributor",
                          component=ranked[0][0], wall_ms=ranked[0][1]))
    for group, key, message in (
            ("system_vm", "pswpin", "System-wide swap-ins occurred; other processes may contribute."),
            ("process", "major_faults", "Process major faults occurred; memory-mapped model reads can cause these."),
            ("cuda", "allocation_retries", "PyTorch allocator retried allocations during this frame.")):
        value = counters.get(group, {}).get(key, 0)
        if value:
            notes.append(dict(type="measured", message=message, counter=key, value=value))
    return notes


def summary(runs):
    return dict(first_frame=distribution([r["frame_wall_ms"] for r in runs[:1]]),
                subsequent_frames=distribution([r["frame_wall_ms"] for r in runs[1:]]),
                all_frames=distribution([r["frame_wall_ms"] for r in runs]),
                interpretation="First frame is process-cold, not necessarily disk-cache-cold. "
                "Subsequent frames can still reload stages. Small-sample percentiles are descriptive.")


def write_json(path, doc):
    """Checkpoint complete frames without leaving a half-written results file."""
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".alpamayo-results-", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
