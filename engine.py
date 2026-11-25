"""Render the original recording and generated accents in bounded blocks."""

from collections import deque

import numpy as np

from arrangement import Arrangement


class Engine:
    def __init__(self, track, rules, seed, intensity=1.0, controls=None, include_lead=True):
        self.track = track
        self.seed = seed
        self.intensity = intensity
        self.controls = controls if controls is not None else {}
        self.control_log = deque(maxlen=256)
        self.controls_truncated = False
        self.history = deque(maxlen=48)
        self.elapsed = 0
        self.snapshot = None
        self.arrangement = Arrangement(rules, seed)
        self.source_pos = 0
        self.until_step = track.first_bar
        self.step = 0
        self.plan = {}
        self.hit = None
        self.hit_pos = 0
        self.hit_gain = 0
        self.styles = rules["styles"]
        self.voices = []
        self.lead_plan = {}
        self.include_lead = include_lead
        self.mix_from = self.mix_to = 1.0
        self.mix_length = self.mix_position = track.step * 8
        self.trim_from = self.trim_to = 1.0
        self.trim_length = self.trim_position = self.mix_length
        self.trims = {name: 10 ** (style.get("trim_db", 0) / 20)
                      for name, style in self.styles.items()}
        self.release = np.linspace(1, 0, track.frames(.035), dtype=np.float32)

    def mix_source(self, block, source_slice):
        gain = self.mix_to
        if self.mix_position >= self.mix_length:
            if gain == 0:
                block.fill(0)
                return
            np.copyto(block, self.track.audio[source_slice])
            if gain != 1:
                block *= gain
            return
        progress = np.minimum(
            (self.mix_position + np.arange(len(block))) / self.mix_length, 1
        )[:, None]
        block[:] = self.track.audio[source_slice] * (self.mix_from + (gain - self.mix_from) * progress)
        self.mix_position += len(block)

    def play_note(self, instrument, pitch_class, gain):
        notes = self.track.instruments[instrument]
        midi = next(midi for midi in notes if midi % 12 == pitch_class)
        self.voices.append((notes[midi], 0, gain, False))

    def schedule_step(self):
        track = self.track
        style = self.arrangement.style
        config = self.styles[style]
        tone = self.arrangement.tone_pattern.get(self.step)
        if tone:
            pitch, gain = tone
            self.play_note(style, track.pitches_at(self.source_pos)[pitch], gain)
        if self.include_lead:
            for instrument, midi, gain, duration, delay in self.lead_plan.get(self.step, ()):
                note = track.instruments[instrument][midi]
                length = duration * track.step + len(self.release)
                self.voices.append((note[:length], -track.frames(delay / 1000), gain, len(note) > length))
        rhythm = config.get("rhythm")
        reconstructed = config["lead_mode"] in ("replace", "trade", "omit")
        hits = track.kit if reconstructed else track.hits
        if rhythm:
            pitches = track.pitches_at(self.source_pos)
            if self.step in self.arrangement.bass_steps:
                bass = rhythm.get("bass_instrument", "replacement_bass")
                self.play_note(bass, pitches[0], .9)
            harmony = rhythm.get("harmony")
            if harmony and self.step in harmony["steps"]:
                gain = harmony["gain"]
                harmonic = pitches[5] if pitches[6] >= 35 else pitches[0]
                self.play_note(harmony["instrument"], harmonic, gain)
                if pitches[6] >= 56 and pitches[4] != harmonic:
                    self.play_note(harmony["instrument"], pitches[4], gain * .63)
            drum = rhythm["kit"].get(str(self.step))
            if drum:
                hit, gain = drum
                self.voices.append((hits[hit], 0, gain, False))
        event = self.plan.get(self.step)
        if event and config["drum_gain"]:
            name, gain = event
            self.hit_gain = gain * config["drum_gain"]
            self.hit = hits[name]
            self.hit_pos = 0

    def next_bar(self):
        track = self.track
        bar = self.arrangement.bar
        if bar in self.controls:
            self.intensity = self.controls[bar]
        intensity = self.intensity
        if bar == 0 or intensity != self.arrangement.intensity:
            self.controls_truncated |= len(self.control_log) == self.control_log.maxlen
            self.control_log.append((bar, intensity))
        previous = self.arrangement.style
        section = track.sections.get(self.source_pos, {})
        contour = [track.melody_at(self.source_pos + step * track.step)
                   for step in range(track.steps_per_bar)]
        confidence = track.phrase_confidence_at(self.source_pos)
        self.plan = self.arrangement.next_bar(section, intensity, confidence)
        name = self.arrangement.style
        style = self.styles[name]
        melody = style.get("melody")
        if melody:
            self.lead_plan = self.arrangement.lead_events(contour, melody, previous != name)
        else:
            self.lead_plan = {}
        mode = style["lead_mode"]
        target = style["original_gain"]
        if target != self.mix_to:
            self.mix_from = self.mix_to
            self.mix_to = target
            self.mix_position = 0
            self.mix_length = track.step * (2 if target == 0 else 8)
        trim = self.trims[name]
        if trim != self.trim_to:
            self.trim_from, self.trim_to = self.trim_to, trim
            self.trim_position = 0
        parts = [label for label, enabled in (
            ("原曲", target),
            ("旋律", self.lead_plan),
            ("伴奏", self.arrangement.tone_pattern or style.get("rhythm", {}).get("harmony")),
            ("ベース", self.arrangement.bass_steps),
            ("打音", style.get("rhythm", {}).get("kit") or (self.plan and style["drum_gain"])),
        ) if enabled]
        labels = {"original": "原曲演奏", "reinforce": "旋律補強", "replace": "旋律置換",
                  "omit": "旋律省略", "trade": "旋律交代"}
        self.snapshot = (bar, name, self.arrangement.energy, intensity,
                         f'{labels[mode]} · {self.arrangement.arc} · {" / ".join(parts)}')
        if previous != name or bar == 0:
            self.history.append((self.elapsed, self.source_pos, name))

    def render(self, output):
        track = self.track
        offset = 0
        while offset < len(output):
            if self.until_step == 0:
                if self.step == 0:
                    self.next_bar()
                self.schedule_step()
                self.step = (self.step + 1) % track.steps_per_bar
                self.until_step = track.step

            count = min(len(output) - offset, track.end - self.source_pos, self.until_step)
            block = output[offset:offset + count]
            self.mix_source(block, slice(self.source_pos, self.source_pos + count))
            if self.hit is not None:
                length = min(count, len(self.hit) - self.hit_pos)
                block[:length] += self.hit[self.hit_pos:self.hit_pos + length] * self.hit_gain
                self.hit_pos += length
                if self.hit_pos == len(self.hit):
                    self.hit = None
            active = []
            for note, position, gain, release in self.voices:
                delay = max(0, -position)
                if delay >= count:
                    active.append((note, position + count, gain, release))
                    continue
                position += delay
                length = min(count - delay, len(note) - position)
                samples = note[position:position + length] * gain
                release_at = len(note) - len(self.release)
                if release and position + length > release_at:
                    start = max(0, release_at - position)
                    fade = position + start - release_at
                    samples[start:] *= self.release[fade:fade + length - start, None]
                block[delay:delay + length] += samples
                if position + length < len(note):
                    active.append((note, position + length, gain, release))
            self.voices = active
            if self.trim_position < self.trim_length:
                progress = np.minimum(
                    (self.trim_position + np.arange(count)) / self.trim_length, 1
                )
                block *= (self.trim_from + (self.trim_to - self.trim_from) * progress)[:, None]
                self.trim_position += count
            elif self.trim_to != 1:
                block *= self.trim_to
            # Keep coincident lead, bass and drum attacks below full scale.
            loud = np.abs(block) > .88
            if loud.any():
                peaks = block[loud]
                excess = np.abs(peaks) - .88
                block[loud] = np.copysign(.88 + .10 * excess / (.10 + excess), peaks)
            self.source_pos += count
            if self.source_pos == track.end:
                self.source_pos = track.start
            self.until_step -= count
            offset += count
            self.elapsed += count
