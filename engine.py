"""Render the original recording and generated accents in bounded blocks."""

import numpy as np

from arrangement import Arrangement


class Engine:
    def __init__(self, track, rules, seed):
        self.track = track
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
        self.bar_start = 0
        self.last_lead = None
        self.replacement = None
        self.mix_from = self.mix_to = 1.0
        self.mix_length = self.mix_position = track.step * 8

    def play_note(self, instrument, pitch_class, gain, lead_center=None):
        notes = self.track.instruments[instrument]
        matching = [midi for midi in notes if midi % 12 == pitch_class]
        if lead_center is None:
            midi = matching[0]
        else:
            reference = lead_center if self.last_lead is None else self.last_lead
            midi = min(matching, key=lambda note: abs(note - reference)
                       + .45 * abs(note - lead_center))
            self.last_lead = midi
        self.voices.append((notes[midi], 0, gain))

    def schedule_step(self):
        track = self.track
        style = self.arrangement.style
        config = self.styles[style]
        tone = self.arrangement.tone_pattern.get(self.step)
        if tone:
            pitch, gain = tone
            self.play_note(style, track.pitches_at(self.source_pos)[pitch], gain)
        if self.replacement:
            event = self.arrangement.melody_pattern.get(self.step)
            if event:
                beat, pitch, gain = event
                pitch_class = track.pitches_at(self.bar_start + beat * track.step * 4)[pitch]
                if self.step % 4 == 0:
                    bass, upper, _ = track.pitches_at(self.source_pos)
                    if (pitch_class - bass) % 12 in (1, 6, 11):
                        pitch_class = upper
                self.play_note(style, pitch_class, gain, self.replacement["center"])
            if self.step in self.replacement["bass_steps"]:
                self.play_note("replacement_bass", track.pitches_at(self.source_pos)[0], .9)
            drum = self.replacement["kit"].get(str(self.step))
            if drum:
                hit, gain = drum
                self.voices.append((track.hits[hit], 0, gain))
        event = self.plan.get(self.step)
        if event and config["drum_gain"]:
            name, gain = event
            self.hit_gain = gain * config["drum_gain"]
            self.hit = track.hits[name]
            self.hit_pos = 0

    def render(self, output):
        track = self.track
        offset = 0
        while offset < len(output):
            if self.until_step == 0:
                if self.step == 0:
                    self.plan = self.arrangement.next_bar()
                    style = self.styles[self.arrangement.style]
                    self.bar_start = self.source_pos
                    replacement = style.get("melody")
                    if replacement is not self.replacement:
                        self.last_lead = None
                    self.replacement = replacement
                    target = style["original_gain"]
                    if target != self.mix_to:
                        self.mix_from = self.mix_to
                        self.mix_to = target
                        self.mix_position = 0
                self.schedule_step()
                self.step = (self.step + 1) % track.steps_per_bar
                self.until_step = track.step

            count = min(len(output) - offset, track.end - self.source_pos, self.until_step)
            block = output[offset:offset + count]
            np.copyto(block, track.audio[self.source_pos:self.source_pos + count])
            if self.mix_position < self.mix_length:
                progress = np.minimum(
                    (self.mix_position + np.arange(count)) / self.mix_length, 1
                )
                block *= (self.mix_from + (self.mix_to - self.mix_from) * progress)[:, None]
                self.mix_position += count
            elif self.mix_to != 1:
                block *= self.mix_to
            if self.hit is not None:
                length = min(count, len(self.hit) - self.hit_pos)
                block[:length] += self.hit[self.hit_pos:self.hit_pos + length] * self.hit_gain
                self.hit_pos += length
                if self.hit_pos == len(self.hit):
                    self.hit = None
            active = []
            for note, position, gain in self.voices:
                length = min(count, len(note) - position)
                block[:length] += note[position:position + length] * gain
                if position + length < len(note):
                    active.append((note, position + length, gain))
            self.voices = active
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
