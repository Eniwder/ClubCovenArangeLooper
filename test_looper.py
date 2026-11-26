import copy
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import tracemalloc
import unittest
from unittest.mock import patch
import wave

import numpy as np

from arrangement import Arrangement
from engine import Engine
from material import ROOT, Track, read_json
from play import preview
from procedural import Procedural
from settings import validate


class ClubCovenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = read_json("track.json")
        cls.rules = read_json("rules.json")
        validate(cls.config, cls.rules)
        cls.track = Track(cls.config, cls.rules, ROOT / ".cache" / "track.npz")

    def render(self, seed, size, seconds, intensity=1, controls=None):
        track = self.track
        engine = Engine(track, self.rules, seed, intensity, controls)
        block = np.empty((size, 2), np.float32)
        digest = sha256()
        for start in range(0, track.frames(seconds), size):
            chunk = block[:min(size, track.frames(seconds) - start)]
            engine.render(chunk)
            self.assertTrue(np.isfinite(chunk).all())
            self.assertLess(float(np.max(np.abs(chunk))), .98)
            digest.update(chunk)
        return digest.digest(), engine

    def test_seed_block_size_and_control_replay(self):
        inputs = {0: .45, 7: 1, 33: 0, 61: .75}
        digest, engine = self.render(44, 4096, 210, controls=inputs)
        self.assertEqual(dict(engine.control_log), inputs)
        self.assertEqual(digest, self.render(44, 507, 210, controls=dict(engine.control_log))[0])
        self.assertNotEqual(digest, self.render(43, 4096, 210, controls=inputs)[0])

    def test_loop_waveform_continuity_and_zero_intensity(self):
        track = self.track
        for audio in (track.audio, track.backing):
            np.testing.assert_array_equal(audio[track.end - 1], audio[track.start - 1])
        engine = Engine(track, self.rules, 44, 0)
        engine.source_pos = track.end - 32
        engine.until_step = 0
        block = np.empty((64, 2), np.float32)
        engine.render(block)
        np.testing.assert_array_equal(block[:32], track.audio[-32:])
        np.testing.assert_array_equal(block[32:], track.audio[track.start:track.start + 32])
        self.assertFalse(engine.voices)

    def test_style_transitions_and_intensity_bounds(self):
        sections = ({}, {"energy": .5, "protected": True}, {}, {"energy": .65})
        for seed in range(6):
            for intensity in (0, .20, .55, 1):
                a = Arrangement(self.rules, seed)
                seen = set()
                for bar in range(200):
                    section = sections[bar % 4]
                    previous = a.style
                    plan = a.next_bar(section, intensity)
                    seen.add(a.style)
                    self.assertGreaterEqual(intensity, self.rules["styles"][a.style]["min_intensity"])
                    self.assertTrue(all(0 <= step < 16 for step in plan))
                    if section.get("protected"):
                        self.assertEqual(previous, a.style)
                    if a.style != previous and a.style != "original":
                        self.assertIn(a.style, self.rules["styles"][previous]["next"])
                if not intensity:
                    self.assertEqual(seen, {"original"})
                else:
                    self.assertGreater(len(seen), 1)

    def test_engine_changes_style_at_song_length_boundaries(self):
        track = self.track
        engine = Engine(track, self.rules, 44)
        bar_frames = track.step * track.steps_per_bar
        texture_changes = []
        previous_texture = None
        for bar in range(281):
            engine.elapsed = track.first_bar + bar * bar_frames
            engine.source_pos = engine.elapsed
            if engine.source_pos >= track.end:
                engine.source_pos = track.start + (engine.source_pos - track.end) % (track.end - track.start)
            engine.next_bar()
            texture = engine.generated[0] if engine.generated else None
            if texture is not previous_texture:
                texture_changes.append(engine.elapsed / track.rate)
                previous_texture = texture
        self.assertEqual([elapsed / track.rate for elapsed, _, _ in engine.history],
                         [16.375, 168.375, 372.375, 576.375])
        self.assertEqual(texture_changes, [168.375, 372.375, 576.375])
        self.assertTrue(all(position == track.frames(32.375)
                            for _, position, _ in list(engine.history)[1:]))

    def test_phrase_layer_choices_combine_independently(self):
        arrangement = Arrangement(self.rules, 44)
        arrangement.style = "melody_sync"
        patterns = set()
        for bar in range(256):
            arrangement.next_bar(change_style=False)
            if bar % self.rules["phrase_bars"] == 1:
                patterns.add((tuple(arrangement.bass_steps), tuple(arrangement.harmony_steps),
                              tuple(arrangement.kit_pattern)))
        self.assertEqual(len(patterns), 8)

    def test_procedural_motifs_repeat_and_rhythms_replace_the_recording(self):
        generator = Procedural(44, 4)
        style = self.rules["styles"]["melody_keys"]
        pitches = [[0, 4, 7, 0, 4, 7, 80]] * 4
        feels = set()
        for _ in range(100):
            texture = generator.candidate(style, 1, .55, .7)
            feels.add(texture.feel)
            self.assertEqual(texture.source_gain, 0)
            first = generator.backing(texture, 0, 0, pitches, 1)
            self.assertEqual(first, generator.backing(texture, 1, 0, pitches, 1))
            if texture.feel == "ambient":
                self.assertTrue(all(instrument not in self.track.kit
                                    for events in first.values() for instrument, *_ in events))
        self.assertEqual(len(feels), 8)
        for _ in range(50):
            generator.select(style, 1, .55, .7)
        self.assertEqual(len(generator.history), 32)

    def test_procedural_reprise_requires_muted_source_and_matching_harmony(self):
        track = self.track
        engine = Engine(track, self.rules, 44)
        engine.source_pos = track.frames(32.375)
        style = self.rules["styles"]["melody_keys"]
        generator = engine.procedural
        for _ in range(100):
            texture = generator.candidate(style, 1, .4, .6)
            if texture.treatment == "interlude":
                break
        else:
            self.fail("no interlude generated")
        self.assertEqual(engine.accompaniment_position(texture, 0), engine.source_pos)
        engine.mix_to = 0
        self.assertEqual(engine.accompaniment_position(texture, 0), track.frames(16.375))
        self.assertEqual(engine.accompaniment_position(texture, 1), track.frames(18.375))
        with patch.object(track, "pitches_at", side_effect=lambda position: (position, 0, 0, 0, 0, 0, 0)):
            self.assertEqual(engine.accompaniment_position(texture, 0), engine.source_pos)

    def test_note_rest_repeated_attack_and_cross_bar_hold(self):
        notes = self.track.notes
        self.assertTrue(np.any(notes[:, 0] == -1))
        self.assertTrue(np.any((notes[1:, 0] == notes[:-1, 0]) & (notes[1:, 1] == 1)))
        a = Arrangement(self.rules, 44)
        a.style = "melody_keys"
        a.intensity = 1
        spec = self.rules["styles"][a.style]["melody"]
        bar = [(-1, 0, 0, 0)] * 15 + [(67, 1, 4, 92)]
        note = a.lead_events(bar, spec)[15][0]
        self.assertEqual((note[1], note[3]), (67, 4))
        held = [(67, 0, 3, 92), (67, 0, 2, 92), (67, 0, 1, 92), (67, 1, 1, 92)]
        self.assertEqual(set(a.lead_events(held, spec)), {3})
        engine = Engine(self.track, self.rules, 44)
        engine.arrangement.style = a.style
        engine.arrangement.style_bars = 1
        engine.step = 15
        engine.until_step = 0
        engine.lead_plan = a.lead_events(bar, spec)
        engine.source_pos = self.track.first_bar + 15 * self.track.step
        engine.render(np.empty((self.track.step + 1, 2), np.float32))
        self.assertTrue(any(position > self.track.step // 2 for _, position, _, _ in engine.voices))

    def test_bad_settings(self):
        for path, value in (("hit_peak", float("nan")), ("phrase_bars", 0)):
            rules = copy.deepcopy(self.rules)
            rules[path] = value
            with self.assertRaises(ValueError):
                validate(self.config, rules)
        config = dict(self.config, loop_end=self.config["loop_end"] + .1)
        with self.assertRaises(ValueError):
            validate(config, self.rules)
        rules = copy.deepcopy(self.rules)
        rules["styles"]["original"]["next"] = {"missing": 1}
        with self.assertRaises(ValueError):
            validate(self.config, rules)

    def test_same_style_renews_and_recent_history_is_bounded(self):
        rules = copy.deepcopy(self.rules)
        rules["styles"]["light"]["next"] = {"light": 1}
        a = Arrangement(rules, 44)
        a.style = "light"
        a.style_bars = rules["phrase_bars"]
        a.next_bar({"energy": .5}, 1)
        self.assertEqual(a.style, "light")
        self.assertEqual(a.style_bars, 1)
        for _ in range(250):
            a.next_bar({"energy": .5}, 1)
        self.assertEqual(len(a.recent_styles), 6)
        self.assertLessEqual(len(a.recent_arcs), 3)

    def test_reconstruction_has_no_source_audio_and_can_isolate_backing(self):
        t = self.track

        def play(lead, legacy):
            engine = Engine(t, self.rules, 43, include_lead=lead, legacy=legacy)
            engine.arrangement.style = "melody_keys"
            engine.source_pos = t.frames(48.375)
            engine.until_step = 0
            engine.mix_from = engine.mix_to = 0
            out = np.empty((t.frames(6), 2), np.float32)
            for start in range(0, len(out), 8192):
                engine.render(out[start:start + 8192])
            self.assertEqual(engine.mix_to, 0)
            return out

        for legacy in (False, True):
            with self.subTest(legacy=legacy):
                with_lead = play(True, legacy)
                backing = play(False, legacy)
                self.assertGreater(float(np.max(np.abs(with_lead - backing))), .05)
                self.assertGreater(float(np.max(np.abs(backing))), .05)
                with (patch.object(t, "audio", np.zeros_like(t.audio)),
                      patch.object(t, "backing", np.zeros_like(t.backing)),
                      patch.object(t, "hits", {})):
                    np.testing.assert_array_equal(with_lead, play(True, legacy))
                    np.testing.assert_array_equal(backing, play(False, legacy))

    def test_trade_changes_performer_at_two_bar_boundaries(self):
        a = Arrangement(self.rules, 44)
        a.style = "groove_return"
        contour = [(67, 1, 4, 88)]
        spec = self.rules["styles"][a.style]["melody"]
        for age, expected in enumerate(("melody_keys", "melody_keys", "melody_legato", "melody_legato"), 1):
            a.style_bars = age
            plan = a.lead_events(contour, spec)
            self.assertEqual(plan[0][0][0], expected)

    def test_cache_reuse_invalidation_and_manual_rebuild(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "track.npz"
            first = Track(self.config, self.rules, path)
            self.assertFalse(first.cache_hit)
            with patch("material.decode", side_effect=AssertionError("cache should avoid decoding")):
                hit = Track(self.config, self.rules, path)
            np.testing.assert_array_equal(first.notes, hit.notes)
            np.testing.assert_array_equal(first.backing[:100], hit.backing[:100])
            self.assertTrue(hit.cache_hit)
            del first, hit
            with patch.object(Track, "prepare", side_effect=RuntimeError("rebuild ran")):
                for config, rebuild in ((self.config, True), (dict(self.config, target_lufs=-18), False)):
                    with self.assertRaisesRegex(RuntimeError, "rebuild ran"):
                        Track(config, self.rules, path, rebuild)
                path.write_bytes(b"broken")
                with self.assertRaisesRegex(RuntimeError, "rebuild ran"):
                    Track(self.config, self.rules, path)

    def test_wav_and_bounded_state(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "test.wav"
            preview(str(path), 2, Engine(self.track, self.rules, 44))
            with wave.open(str(path), "rb") as audio:
                self.assertEqual(audio.getnframes(), self.track.frames(2))
                self.assertEqual(audio.getnchannels(), 2)
                self.assertEqual(audio.getframerate(), self.track.rate)
        engine = Engine(self.track, self.rules, 44)
        buffer = np.empty((8192, 2), np.float32)
        tracemalloc.start()
        try:
            for i in range(4000):
                engine.render(buffer)
                if i == 999:
                    initial = tracemalloc.get_traced_memory()[0]
            final = tracemalloc.get_traced_memory()[0]
        finally:
            tracemalloc.stop()
        self.assertLess(final - initial, 128 * 1024)
        self.assertLessEqual(len(engine.history), 48)
        self.assertLess(len(engine.voices), 40)


if __name__ == "__main__":
    unittest.main()
