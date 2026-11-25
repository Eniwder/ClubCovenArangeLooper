"""Decode the supplied recording and prepare Club Coven's source material."""

import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
from accompaniment import analyse_melody, analyse_pitches, make_instruments, make_kit
from backing import make_backing
from cache import cache_key, read_cache, write_cache

ROOT = Path(__file__).resolve().parent
FFMPEG_CREATION_FLAGS = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8-sig"))


def decode(source, rate):
    try:
        raw = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-i", str(source),
             "-f", "f32le", "-ac", "2", "-ar", str(rate), "-"],
            capture_output=True, check=True, creationflags=FFMPEG_CREATION_FLAGS,
        ).stdout
    except FileNotFoundError as error:
        raise RuntimeError("FFmpeg が見つかりません。PATH を確認してください。") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(error.stderr.decode(errors="replace").strip()) from error
    return np.frombuffer(raw, dtype="<f4").reshape(-1, 2)


def integrated_loudness(audio, rate):
    """Measure the selected stereo PCM interval before applying fixed gain."""
    report = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-f", "f32le", "-ac", "2",
         "-ar", str(rate), "-i", "pipe:0", "-af", "ebur128=framelog=verbose",
         "-f", "null", "-"],
        input=memoryview(audio).cast("B"), capture_output=True,
        creationflags=FFMPEG_CREATION_FLAGS,
    )
    match = re.search(rb"Integrated loudness:\s+I:\s+(-?\d+(?:\.\d+)?)\s+LUFS",
                      report.stderr)
    if report.returncode or not match:
        raise RuntimeError("FFmpeg で原曲のラウドネスを測定できませんでした。")
    return float(match[1])


class Track:
    def __init__(self, config, rules, cache_path=None, rebuild=False):
        self.rate = config["sample_rate"]
        self.start = self.frames(config["loop_start"])
        self.end = self.frames(config["loop_end"])
        self.first_bar = self.frames(config["first_arranged_bar"])
        self.step = self.frames(60 / config["bpm"] / 4)
        self.steps_per_bar = config["beats_per_bar"] * 4
        self.sections = {self.frames(section["start"]): section for section in config["sections"]}
        self.kit = make_kit(self.rate)
        source_path = ROOT / config["source"]
        self.cache_hit = False
        if cache_path is not None:
            key = cache_key(source_path, config, rules)
            cached = None if rebuild else read_cache(cache_path, key)
            if cached is not None:
                try:
                    self.load_arrays(cached)
                    steps = (self.end - self.first_bar + self.step - 1) // self.step
                    if (self.audio.shape != (self.end, 2) or self.backing.shape != self.audio.shape
                            or self.notes.shape != (steps, 4) or self.melody.shape != (steps,)
                            or self.pitches.shape != ((steps + 3) // 4, 7)
                            or self.hits.keys() != config["hits"].keys()):
                        raise ValueError("キャッシュの形状が一致しません。")
                except (KeyError, ValueError):
                    pass
                else:
                    self.cache_hit = True
                    return
        self.prepare(source_path, config, rules)
        if cache_path is not None:
            write_cache(cache_path, key, self.cache_arrays())

    def load_arrays(self, arrays):
        for name in ("audio", "backing", "pitches", "melody", "notes"):
            setattr(self, name, arrays[name])
        self.hits = {key[4:]: value for key, value in arrays.items() if key.startswith("hit/")}
        self.instruments = {}
        for key, value in arrays.items():
            if key.startswith("instrument/"):
                _, instrument, midi = key.split("/")
                self.instruments.setdefault(instrument, {})[int(midi)] = value

    def cache_arrays(self):
        arrays = {key: getattr(self, key) for key in ("audio", "backing", "pitches", "melody", "notes")}
        arrays.update({f"hit/{name}": hit for name, hit in self.hits.items()})
        for name, notes in self.instruments.items():
            arrays.update({f"instrument/{name}/{midi}": note for midi, note in notes.items()})
        return arrays

    def prepare(self, source_path, config, rules):
        source = decode(source_path, self.rate)
        fade = self.frames(config["crossfade_ms"] / 1000)
        if not 2 <= fade <= self.start < self.end <= len(source):
            raise ValueError("音源の長さとループ設定が一致しません。")
        audio = source[:self.end].copy()
        blend = np.sin(np.linspace(0, np.pi / 2, fade, dtype=np.float32)) ** 2
        # Replace the material just before the wrap with the corresponding
        # incoming pre-roll, leaving the first traversal of loop_start intact.
        audio[self.end - fade:self.end] = (
            source[self.end - fade:self.end] * (1 - blend[:, None])
            + source[self.start - fade:self.start] * blend[:, None]
        )
        loudness = integrated_loudness(audio[self.start:self.end], self.rate)
        target_gain = 10 ** ((config["target_lufs"] - loudness) / 20)
        peak = float(np.max(np.abs(audio)))
        audio *= min(target_gain, (0.97 - rules["hit_peak"]) / max(peak, 1e-6))
        self.audio = audio
        self.hits = {
            name: self.make_hit(source, spec, rules["hit_peak"])
            for name, spec in config["hits"].items()
        }
        self.pitches = analyse_pitches(source, self)
        self.melody, self.notes = analyse_melody(source, self)
        self.backing = make_backing(audio, self, rules["backing"])
        self.backing[self.end - fade:self.end] = (
            self.backing[self.end - fade:self.end] * (1 - blend[:, None])
            + self.backing[self.start - fade:self.start] * blend[:, None]
        )
        playable = {name: spec for name, spec in rules["styles"].items()
                    if spec["pattern"] or "melody" in spec}
        self.instruments = make_instruments(self.rate, playable | rules["instruments"])

    def pitches_at(self, position):
        if position >= self.end:
            position = self.start + (position - self.end) % (self.end - self.start)
        return self.pitches[(position - self.first_bar) // (self.step * 4)]

    def melody_at(self, position):
        if position >= self.end:
            position = self.start + (position - self.end) % (self.end - self.start)
        return self.notes[(position - self.first_bar) // self.step]

    def phrase_confidence_at(self, position):
        width = self.steps_per_bar * 4
        start = (position - self.first_bar) // self.step
        phrase = self.notes[start:start + width]
        if len(phrase) < width:
            return float(sum(self.melody_at(position + step * self.step)[3]
                             for step in range(width)) / (100 * width))
        return float(phrase[:, 3].mean() / 100)

    def frames(self, seconds):
        return round(seconds * self.rate)

    def make_hit(self, source, spec, peak):
        start = self.frames(spec["time"])
        length = self.frames(spec["length_ms"] / 1000)
        pad = self.frames(0.020)
        excerpt = source[start - pad:start + length + pad]
        if len(excerpt) != length + 2 * pad or length > self.step:
            raise ValueError("打音の抽出範囲または長さが不正です。")
        spectrum = np.fft.rfft(excerpt, axis=0)
        freq = np.fft.rfftfreq(len(excerpt), 1 / self.rate)
        low = np.clip((freq - spec["low_hz"]) / 900, 0, 1)
        high = np.clip((spec["high_hz"] - freq) / 1200, 0, 1)
        hit = np.fft.irfft(spectrum * (low * high)[:, None], n=len(excerpt), axis=0)
        hit = hit[pad:pad + length]
        t = np.linspace(0, 1, length)
        envelope = np.minimum(t / 0.06, 1) * (1 - t) ** 1.8
        hit *= envelope[:, None]
        hit *= peak / max(float(np.max(np.abs(hit))), 1e-6)
        return hit.astype(np.float32)
