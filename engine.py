"""Render the original recording and generated accents in bounded blocks."""

from collections import deque

import numpy as np

from arrangement import Arrangement
from procedural import Procedural


class Engine:
    def __init__(self, track, rules, seed, intensity=1.0, controls=None, include_lead=True, legacy=False):
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
        self.procedural = None if legacy else Procedural(seed, rules["phrase_bars"])
        self.generated = None
        self.generated_backing = {}
        self.texture_started = False
        self.wait_for_source_fade = False
        bar_frames = track.step * track.steps_per_bar
        loop_frames = track.end - track.start
        loop_bars = loop_frames // bar_frames
        first_loop_bar = (track.end - track.first_bar + bar_frames - 1) // bar_frames
        first_loop_time = track.first_bar + first_loop_bar * bar_frames
        extra_loops = max(0, round((track.arrangement_length - first_loop_time) / loop_frames))
        self.next_style_bar = first_loop_bar + extra_loops * loop_bars
        self.style_period = max(1, round(track.arrangement_length / loop_frames)) * loop_bars
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

    def queue_note(self, note, gain, duration, delay):
        length = duration * self.track.step + len(self.release) if duration else len(note)
        self.voices.append((note[:length], -self.track.frames(delay / 1000), gain, len(note) > length))

    def schedule_lead(self):
        if self.include_lead and not (self.wait_for_source_fade and self.mix_position < self.mix_length):
            for instrument, midi, gain, duration, delay in self.lead_plan.get(self.step, ()):
                self.queue_note(self.track.instruments[instrument][midi],
                                gain * self.texture_entry_gain(), duration, delay)

    def texture_entry_gain(self):
        return min(1., .45 + self.step * .07) if self.texture_started else 1.

    def schedule_step(self):
        track = self.track
        if self.generated is not None:
            self.schedule_lead()
            for instrument, pitch, gain, duration, delay in self.generated_backing.get(self.step, ()):
                if instrument in track.kit:
                    note = track.kit[instrument]
                else:
                    notes = track.instruments[instrument]
                    midi = next(midi for midi in notes if midi % 12 == pitch)
                    note = notes[midi]
                self.queue_note(note, gain * self.texture_entry_gain(), duration, delay)
            return
        style = self.arrangement.style
        config = self.styles[style]
        tone = self.arrangement.tone_pattern.get(self.step)
        if tone:
            pitch, gain = tone
            self.play_note(style, track.pitches_at(self.source_pos)[pitch], gain)
        self.schedule_lead()
        rhythm = config.get("rhythm")
        reconstructed = config["lead_mode"] in ("replace", "trade", "omit")
        hits = track.kit if reconstructed else track.hits
        if rhythm:
            pitches = track.pitches_at(self.source_pos)
            if self.step in self.arrangement.bass_steps:
                bass = rhythm.get("bass_instrument", "replacement_bass")
                self.play_note(bass, pitches[0], .9)
            harmony = rhythm.get("harmony")
            if harmony and self.step in self.arrangement.harmony_steps:
                gain = harmony["gain"]
                harmonic = pitches[5] if pitches[6] >= 35 else pitches[0]
                self.play_note(harmony["instrument"], harmonic, gain)
                if pitches[6] >= 56 and pitches[4] != harmonic:
                    self.play_note(harmony["instrument"], pitches[4], gain * .63)
            drum = self.arrangement.kit_pattern.get(str(self.step))
            if drum:
                hit, gain = drum
                self.voices.append((hits[hit], 0, gain, False))
        event = self.plan.get(self.step)
        if event and config["drum_gain"]:
            name, gain = event
            self.hit_gain = gain * config["drum_gain"]
            self.hit = hits[name]
            self.hit_pos = 0

    def accompaniment_position(self, texture, position):
        track = self.track
        if texture.treatment != "interlude":
            return self.source_pos
        bar_frames = track.steps_per_bar * track.step
        if position == 0:
            self.procedural.reprise_source = None
            alternate = track.reprises.get(self.source_pos)
            last_beat = bar_frames * self.procedural.phrase_bars - 4 * track.step
            if alternate is not None and self.mix_to == 0 and self.source_pos + last_beat < track.end:
                for offset in (0, last_beat):
                    expected = track.pitches_at(self.source_pos + offset)
                    actual = track.pitches_at(alternate + offset)
                    if expected[0] != actual[0] or tuple(expected[3:6]) != tuple(actual[3:6]):
                        break
                else:
                    self.procedural.reprise_source = alternate
        alternate = self.procedural.reprise_source
        return self.source_pos if alternate is None else alternate + position * bar_frames

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
        change_style = bar >= self.next_style_bar and not section.get("protected")
        if change_style:
            self.next_style_bar += self.style_period
        self.plan = self.arrangement.next_bar(section, intensity, confidence, change_style)
        name = self.arrangement.style
        style = self.styles[name]
        melody = style.get("melody")
        if melody:
            self.lead_plan = self.arrangement.lead_events(contour, melody, previous != name)
        else:
            self.lead_plan = {}
        mode = style["lead_mode"]
        target = style["original_gain"]
        was_generated = self.generated is not None
        self.generated = None
        self.texture_started = False
        if self.procedural is not None:
            change_texture = change_style or (bar == 0 and name != "original")
            self.generated = self.procedural.next_bar(style, change_texture, intensity,
                                                      self.arrangement.energy, confidence)
            self.texture_started = self.generated is not None and change_texture
        if self.generated is not None:
            texture, position, phrase = self.generated
            accompaniment = self.accompaniment_position(texture, position)
            pitches = [track.pitches_at(accompaniment + beat * 4 * track.step) for beat in range(4)]
            self.generated_backing = self.procedural.backing(texture, position, phrase, pitches, intensity)
            self.lead_plan = self.procedural.lead(texture, position, phrase, contour, pitches,
                                                  section.get("protected"))
            mode = "original" if texture.source_gain else "replace" if texture.lead else "omit"
            target = texture.source_gain
        if target != self.mix_to:
            self.wait_for_source_fade = self.generated is not None and target == 0
            self.mix_from = self.mix_to
            self.mix_to = target
            self.mix_position = 0
            self.mix_length = track.step * (8 if was_generated or self.generated is not None or target else 2)
        trim = 1.0 if self.generated is not None else self.trims[name]
        if trim != self.trim_to:
            self.trim_from, self.trim_to = self.trim_to, trim
            self.trim_position = 0
        parts = [label for label, enabled in (
            ("原曲", target),
            ("旋律", self.lead_plan),
            ("伴奏", self.arrangement.tone_pattern or self.arrangement.harmony_steps),
            ("ベース", self.arrangement.bass_steps),
            ("打音", self.arrangement.kit_pattern or (self.plan and style["drum_gain"])),
        ) if enabled]
        labels = {"original": "原曲演奏", "reinforce": "旋律補強", "replace": "旋律置換",
                  "omit": "旋律省略", "trade": "旋律交代"}
        if self.generated is not None:
            parts = [texture.feel, self.procedural.previous_lead or "—", texture.treatment,
                     texture.harmony]
            if accompaniment != self.source_pos:
                parts.append("回帰")
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
