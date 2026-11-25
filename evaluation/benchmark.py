"""Measure prepared audio, renderer and one short PortAudio session."""

import argparse
import ctypes
from pathlib import Path
import sys
from time import perf_counter, process_time
import tracemalloc

import numpy as np


def working_set_mib():
    if sys.platform != "win32":
        return None

    class Counters(ctypes.Structure):
        _fields_ = [("size", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in ("peak", "working", "quota_peak", "quota",
                                               "nonpaged_peak", "nonpaged", "page", "page_peak")
        ]

    stats = Counters()
    stats.size = ctypes.sizeof(stats)
    handle = ctypes.c_void_p(-1)
    if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(stats), stats.size):
        return round(stats.working / 2 ** 20, 1)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--device", action="store_true")
    parser.add_argument("--hours", type=float, default=2)
    args = parser.parse_args()
    sys.path.insert(0, str(args.root.resolve()))
    from engine import Engine
    from material import ROOT, Track, read_json

    config, rules = read_json("track.json"), read_json("rules.json")
    start = perf_counter()
    track = Track(config, rules) if args.legacy else Track(config, rules, ROOT / ".cache/track.npz")
    print(f"startup {perf_counter() - start:.3f} s, working set {working_set_mib()} MiB", flush=True)

    for seed in (42, 44, 117):
        engine = Engine(track, rules, seed)
        size = 8192
        audio = np.empty((size, 2), dtype=np.float32)
        timings = []
        peak = 0
        quiet_frames = 0
        worst_quiet = 0
        start = perf_counter()
        cpu = process_time()
        for i in range(0, track.frames(args.hours * 3600), size):
            begin = perf_counter()
            engine.render(audio)
            timings.append(perf_counter() - begin)
            maximum = float(np.max(np.abs(audio)))
            if not np.isfinite(maximum) or maximum >= .98:
                raise RuntimeError(f"Invalid output for seed {seed} at frame {i}: {maximum}")
            peak = max(peak, maximum)
            quiet_frames = quiet_frames + size if maximum < .001 else 0
            worst_quiet = max(worst_quiet, quiet_frames)
        elapsed = perf_counter() - start
        p99, maximum = np.percentile(timings, [99, 100]) * 1000
        print(f"seed {seed}: wall {elapsed:.2f} s, CPU {process_time() - cpu:.2f} s, "
              f"render p99/max {p99:.3f}/{maximum:.3f} ms, "
              f"peak {peak:.3f}, near-silence {worst_quiet / track.rate:.2f} s", flush=True)
        if worst_quiet > track.frames(2):
            raise RuntimeError("Unexpected extended silence")

    engine = Engine(track, rules, 44)
    tracemalloc.start()
    for index in range(4500):
        engine.render(audio)
        if index in (999, 4499):
            print(f"frame {index * size}: traced current/peak {tracemalloc.get_traced_memory()}", flush=True)
    tracemalloc.stop()
    if not args.device:
        return

    import sounddevice as sd

    engine = Engine(track, rules, 44)
    for _ in range(round(24 * track.rate / size)):
        engine.render(audio)
    timings = []
    underruns = 0

    def callback(out, frames, timing, flags):
        nonlocal underruns
        begin = perf_counter()
        engine.render(out)
        out *= .04
        timings.append(perf_counter() - begin)
        underruns += int(flags.output_underflow)

    cpu = process_time()
    with sd.OutputStream(samplerate=track.rate, channels=2, dtype="float32", blocksize=512,
                         latency="high", callback=callback) as stream:
        sd.sleep(12000)
        load = stream.cpu_load
    p99, maximum = np.percentile(timings, [99, 100]) * 1000
    print(f"device 12 s: {len(timings)} callbacks, p99/max {p99:.3f}/{maximum:.3f} ms, "
          f"{underruns} underruns, PortAudio load {load:.2%}, process CPU "
          f"{(process_time() - cpu) / 12:.2%}, working set {working_set_mib()} MiB")


if __name__ == "__main__":
    main()
