"""Decode the supplied recording and prepare Club Coven's source material."""

import json
from pathlib import Path
import subprocess

import numpy as np
from accompaniment import analyse_pitches, make_instruments

ROOT = Path(__file__).resolve().parent


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8-sig"))


def decode(source, rate):
    try:
        raw = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-i", str(source),
             "-f", "f32le", "-ac", "2", "-ar", str(rate), "-"],
            capture_output=True, check=True,
        ).stdout
    except FileNotFoundError as error:
        raise RuntimeError("FFmpeg が見つかりません。PATH を確認してください。") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(error.stderr.decode(errors="replace").strip()) from error
    return np.frombuffer(raw, dtype="<f4").reshape(-1, 2)


class Track:
    def __init__(self, config, rules):
        self.rate = config["sample_rate"]
        self.start = self.frames(config["loop_start"])
        self.end = self.frames(config["loop_end"])
        self.first_bar = self.frames(config["first_arranged_bar"])
        self.step = self.frames(60 / config["bpm"] / 4)
        self.steps_per_bar = config["beats_per_bar"] * 4

        source = decode(ROOT / config["source"], self.rate)
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
        peak = float(np.max(np.abs(audio)))
        audio *= min(1.0, (0.97 - rules["hit_peak"]) / max(peak, 1e-6))
        self.audio = audio
        self.hits = {
            name: self.make_hit(source, spec, rules["hit_peak"])
            for name, spec in config["hits"].items()
        }
        self.pitches = analyse_pitches(source, self)
        playable = {name: spec for name, spec in rules["styles"].items()
                    if spec["pattern"] or "melody" in spec}
        self.instruments = make_instruments(self.rate, playable | rules["instruments"])

    def pitches_at(self, position):
        if position >= self.end:
            position = self.start + (position - self.end) % (self.end - self.start)
        return self.pitches[(position - self.first_bar) // (self.step * 4)]

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
