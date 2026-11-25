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
        self.replacement = None
        self.lead_plan = {}
        self.mix_from = self.mix_to = (1.0, False)
        self.mix_length = self.mix_position = track.step * 8

    def mix_source(self, block, source_slice):
        gain, filtered = self.mix_to
        dry = self.track.audio[source_slice]
        backing = self.track.backing[source_slice]
        target = backing if filtered else dry
        if self.mix_position >= self.mix_length:
            if gain == 0:
                block.fill(0)
                return
            np.copyto(block, target)
            if gain != 1:
                block *= gain
            return
        previous_gain, previous_filtered = self.mix_from
        previous = backing if previous_filtered else dry
        progress = np.minimum(
            (self.mix_position + np.arange(len(block))) / self.mix_length, 1
        )[:, None]
        block[:] = previous * (previous_gain * (1 - progress)) + target * (gain * progress)
        self.mix_position += len(block)

    def play_note(self, instrument, pitch_class, gain):
        notes = self.track.instruments[instrument]
        midi = next(midi for midi in notes if midi % 12 == pitch_class)
        self.voices.append((notes[midi], 0, gain))

    def schedule_step(self):
        track = self.track
        style = self.arrangement.style
        config = self.styles[style]
        tone = self.arrangement.tone_pattern.get(self.step)
        if tone:
            pitch, gain = tone
            self.play_note(style, track.pitches_at(self.source_pos)[pitch], gain)
        for instrument, midi, gain in self.lead_plan.get(self.step, ()):
            self.voices.append((track.instruments[instrument][midi], 0, gain))
        rhythm = config.get("rhythm")
        if rhythm:
            if self.step in self.arrangement.bass_steps:
                bass = rhythm.get("bass_instrument", "replacement_bass")
                self.play_note(bass, track.pitches_at(self.source_pos)[0], .9)
            drum = rhythm["kit"].get(str(self.step))
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
                    self.replacement = style.get("melody")
                    if self.replacement:
                        contour = [track.melody_at(self.source_pos + step * track.step)
                                   for step in range(track.steps_per_bar)]
                        self.lead_plan = self.arrangement.lead_events(contour, self.replacement)
                    else:
                        self.lead_plan = {}
                    target = (style["original_gain"], bool(self.replacement))
                    if target != self.mix_to:
                        self.mix_from = self.mix_to
                        self.mix_to = target
                        self.mix_position = 0
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
