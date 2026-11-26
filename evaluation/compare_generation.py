"""Compare the audible arrangement features of three 30-minute renders per mode."""

from collections import Counter, deque
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import Engine
from material import ROOT, Track, read_json


def signature(engine):
    drums = engine.track.kit
    layers = [set(), set(), set(), set()]
    performers = {note[0] for notes in engine.lead_plan.values() for note in notes}
    if engine.generated is None:
        arrangement = engine.arrangement
        style = engine.styles[arrangement.style]
        rhythm = style.get("rhythm", {})
        layers[0].update(arrangement.bass_steps)
        layers[1].update(arrangement.tone_pattern)
        layers[1].update(arrangement.harmony_steps)
        layers[2].update(int(step) for step, hit in arrangement.kit_pattern.items() if hit[0] == "kick")
        layers[3].update(int(step) for step, hit in arrangement.kit_pattern.items() if hit[0] != "kick")
        if style["drum_gain"]:
            layers[3].update(engine.plan)
        harmony = rhythm.get("harmony", {}).get("instrument", "")
        if arrangement.tone_pattern:
            harmony = arrangement.style
        feel, treatment = "legacy", style["lead_mode"]
    else:
        texture, _, _ = engine.generated
        harmony = texture.harmony
        feel, treatment = texture.feel, texture.treatment
        for step, events in engine.generated_backing.items():
            for name, *_ in events:
                layer = (2 if name == "kick" else 3) if name in drums else (0 if name == texture.bass else 1)
                layers[layer].add(step)
    # Sixteenth-note jitter and gain changes do not affect the signature.
    accents = tuple(frozenset(step // 2 for step in layer) for layer in layers)
    lead = tuple(sorted(performers - {"lead_echo", "melody_shadow"}))
    return (bool(engine.mix_to), lead, harmony, accents), feel, treatment


def similar(left, right):
    if left[:3] != right[:3]:
        return False
    intersection = sum(len(a & b) for a, b in zip(left[3], right[3]))
    union = sum(len(a | b) for a, b in zip(left[3], right[3]))
    return not union or intersection / union >= .75


def measure(track, rules, seed, legacy, seconds=1800):
    engine = Engine(track, rules, seed, legacy=legacy)
    engine.render(np.empty((track.first_bar, 2), np.float32))
    block = np.empty((track.steps_per_bar * track.step, 2), np.float32)
    total = (track.frames(seconds) - track.first_bar) // len(block)
    history = deque(maxlen=64)
    occurrences, feels, treatments, runs = Counter(), Counter(), Counter(), []
    similar_bars = recorded = reprise_bars = 0
    peak = 0.
    start = perf_counter()
    elapsed_run = 0
    previous_texture = None
    for i in range(total):
        engine.render(block)
        assert np.isfinite(block).all()
        peak = max(peak, float(np.max(np.abs(block))))
        shape, feel, treatment = signature(engine)
        occurrences[shape] += 1
        feels[feel] += 1
        treatments[treatment] += 1
        recorded += shape[0]
        reprise_bars += bool(engine.generated and engine.procedural.reprise_source is not None)
        if any(similar(shape, older) for older in list(history)[:-7]):
            similar_bars += 1
        history.append(shape)
        texture = engine.generated[0] if engine.generated else engine.arrangement.style
        if i and texture != previous_texture:
            runs.append(elapsed_run)
            elapsed_run = 0
        elapsed_run += 1
        previous_texture = texture
    elapsed = perf_counter() - start
    runs.append(elapsed_run)
    assert peak <= .98
    print(f"{'legacy' if legacy else 'procedural':10} {seed:3}: {total} bars, "
          f"top={max(occurrences.values()) / total:.1%}, near={similar_bars / total:.1%}, "
          f"recording={recorded / total:.1%}, median texture={np.median(runs):g} bars, "
          f"reprise={reprise_bars} bars, peak={peak:.3f}, render={elapsed:.2f}s")
    print("  feels:", dict(feels), "  lead treatments:", dict(treatments))


if __name__ == "__main__":
    rules = read_json("rules.json")
    track = Track(read_json("track.json"), rules, ROOT / ".cache" / "track.npz")
    for seed in (17, 44, 108):
        for legacy in (True, False):
            measure(track, rules, seed, legacy)
