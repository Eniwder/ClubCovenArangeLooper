"""Generate section textures and recurring rhythmic motifs on the source grid."""

from collections import deque
from dataclasses import dataclass
import random


# Kick and backbeat anchors leave the bass, harmony and percussion rhythms open.
FEELS = {
    "four": ((0, 4, 8, 12), (4, 12), .82, 0),
    "half": ((0, 10), (8,), .38, 0),
    "double": ((0, 6, 8, 14), (4, 12), .87, 0),
    "offbeat": ((0, 8), (4, 12), .64, 0),
    "broken": ((0, 7, 10), (4, 12), .72, 0),
    "shuffle": ((0, 8), (4, 12), .56, 28),
    "percussion": ((), (), .60, 0),
    "ambient": ((), (), .16, 0),
    "recording": ((), (), .3, 0),
}
LEADS = ("melody_keys", "melody_sync", "melody_legato", "melody_response")


@dataclass(frozen=True)
class Texture:
    feel: str
    lead: str
    treatment: str
    bass: str
    harmony: str
    voicing: str
    motifs: tuple
    source_gain: float

    def similarity(self, other):
        score = (3 * (self.feel == other.feel) + 2 * (self.lead == other.lead)
                 + (self.treatment == other.treatment) + (self.bass == other.bass)
                 + (self.harmony == other.harmony) + (self.voicing == other.voicing)
                 + 3 * (bool(self.source_gain) == bool(other.source_gain)))
        for left, right in zip(self.motifs, other.motifs):
            if left or right:
                # Compare metrical position and density before exact sixteenth placement.
                a, b = set(left), set(right)
                beats_a, beats_b = {step % 4 for step in a}, {step % 4 for step in b}
                score += .65 * len(a & b) / len(a | b) + .35 * len(beats_a & beats_b) / len(beats_a | beats_b)
                score += .5 * (1 - abs(len(a) - len(b)) / max(len(a), len(b)))
        return score


class Procedural:
    def __init__(self, seed, phrase_bars):
        self.random = random.Random(seed ^ 0x639F)
        self.history = deque(maxlen=32)
        self.texture = None
        self.age = 0
        self.phrase_bars = phrase_bars
        self.direction = None
        self.previous_lead = None
        self.reprise_source = None

    def rhythm(self, count, weights, anchors=(), avoid=(), gap=2):
        steps = list(anchors)
        available = [step for step in range(16) if step not in avoid]
        while len(steps) < count:
            choices = [step for step in available if all(abs(step - other) >= gap for other in steps)]
            if not choices:
                break
            steps.append(self.random.choices(choices, [weights[step % 4] for step in choices])[0])
        return tuple(sorted(steps))

    def candidate(self, style, intensity, energy, confidence):
        rng = self.random
        mode = style["lead_mode"]
        keep = mode in ("original", "reinforce") and (intensity < .50 or rng.random() < .64)
        if keep:
            harmony = rng.choice(("soft_keys", "soft_harmony"))
            motif = self.rhythm(rng.choice((2, 3, 4)), (3, .5, 2, .8), gap=3)
            return Texture("recording", "", "source", "", harmony, "single",
                           ((), motif, (), ()), max(.76, style["original_gain"]))
        names = list(FEELS)[:-1]
        target = .64 * energy + .36 * style["energy"]
        weights = [max(.06, 1 - 1.8 * abs(FEELS[name][2] - target)) for name in names]
        feel = rng.choices(names, weights)[0]
        kicks, backbeat, drive, _ = FEELS[feel]
        treatment = rng.choices(
            ("theme", "sparse", "answer", "interlude"),
            (2.5 + confidence, 1.7, 1 + confidence, 2.5 if mode == "omit" else .6),
        )[0]
        favored = {"piano": "melody_keys", "strings": "melody_legato",
                   "synth": "melody_sync", "bell": "melody_response"}.get(style["role"])
        lead = rng.choices(LEADS, [2 if name == favored else 1 for name in LEADS])[0] if treatment != "interlude" else ""
        bass = rng.choice(("replacement_bass", "groove_bass"))
        harmony = rng.choice(("soft_keys", "soft_harmony", "muted_reed"))
        if feel == "ambient":
            harmony = "soft_harmony"
        sync = (2, .9, 3, 1.3) if feel in ("offbeat", "broken", "shuffle") else (4, .5, 1.5, .7)
        bass_steps = self.rhythm(1 + round(4 * drive * intensity), sync, (0,), gap=3)
        chords = self.rhythm(1 + round(3 * drive), sync, avoid=bass_steps, gap=3)
        percussion = () if feel == "ambient" else self.rhythm(
            2 + round(6 * drive), (.9, 1, 4, 1), avoid=kicks + backbeat, gap=2)
        return Texture(feel, lead, treatment, bass, harmony, rng.choice(("single", "broken", "dyad")),
                       (bass_steps, chords, kicks, percussion), 0.)

    def select(self, style, intensity, energy, confidence):
        candidates = [self.candidate(style, intensity, energy, confidence) for _ in range(7)]
        def weight(candidate):
            recent = [candidate.similarity(previous) for previous in self.history]
            if not recent:
                return 1.
            return 1 / (1 + .10 * max(recent[-5:]) ** 2 + .075 * sum(v >= 10 for v in recent))
        self.texture = self.random.choices(candidates, [weight(item) for item in candidates])[0]
        self.history.append(self.texture)
        self.age = 0
        self.reprise_source = None

    def next_bar(self, style_name, style, section, intensity, energy, confidence):
        if intensity < .30:
            self.texture = None
            self.direction = None
            self.previous_lead = None
            return None
        direction = (style_name, intensity < .50)
        if (self.direction != direction or
                (not section.get("protected") and self.age >= 2 * self.phrase_bars
                 and ((section and self.random.random() < .72) or self.age >= 4 * self.phrase_bars))):
            self.select(style, intensity, energy, confidence)
            self.direction = direction
        texture = self.texture
        position = self.age % self.phrase_bars
        phrase = self.age // self.phrase_bars
        self.age += 1
        return texture, position, phrase

    def backing(self, texture, position, phrase, pitches, intensity):
        events = {}
        _, backbeat, drive, swing = FEELS[texture.feel]
        bass, chords, kicks, percussion = texture.motifs
        ending = position == self.phrase_bars - 1
        if ending and bass and bass[-1] <= 11 and drive >= .4:
            bass += (14,)
        if position >= 2 and chords and chords[-1] < 14 and phrase % 2:
            chords = (*chords[:-1], chords[-1] + 1)

        def add(step, instrument, pitch, gain, length):
            delay = swing if step % 4 == 2 else 0
            events.setdefault(step, []).append((instrument, int(pitch), gain, length, delay))

        for step in bass:
            root, _, _, _, _, fifth, confidence = pitches[step // 4]
            pitch = fifth if step == 14 and confidence >= 65 else root
            add(step, texture.bass, pitch, (.87 if step % 4 == 0 else .64) * intensity,
                2 if drive > .65 else 5)
        for index, step in enumerate(chords):
            root, _, _, chord_root, third, fifth, confidence = pitches[step // 4]
            pitch = root
            if confidence >= 48 and not texture.source_gain:
                pitch = (chord_root, fifth, third)[index % 3] if texture.voicing == "broken" else fifth
            gain = (.26 if texture.source_gain else .65) * intensity
            length = 5 if texture.harmony == "soft_harmony" else 2
            add(step, texture.harmony, pitch, gain, length)
            if confidence >= 65 and texture.voicing == "dyad" and third != pitch:
                add(step, texture.harmony, third, gain * .53, length)
        for step in kicks:
            add(step, "kick", 0, (1.1 if step == 0 else .82) * intensity, 0)
        for step in backbeat:
            add(step, "fill", 0, .88 * intensity, 0)
        for step in percussion:
            hit = "open" if step % 4 == 2 and drive > .6 else "closed"
            add(step, hit, 0, (.68 if step % 2 == 0 else .43) * intensity, 0)
        if ending and percussion and drive > .5 and 15 not in percussion:
            add(15, "fill", 0, .53 * intensity, 0)
        return events

    def lead(self, texture, position, phrase, contour, pitches, protected=False):
        performer = texture.lead
        answer = texture.treatment == "answer" and position >= 2 and not protected
        if answer:
            performer = "melody_response" if performer != "melody_response" else "melody_keys"
        entering = performer != self.previous_lead
        self.previous_lead = performer
        if not performer or (texture.treatment == "sparse" and position == 2 and not protected):
            return {}
        kicks, backbeat, _, swing = FEELS[texture.feel]
        events = {}
        for step, (midi, onset, length, confidence) in enumerate(contour):
            if midi < 0 or not (onset or (entering and step == 0)):
                continue
            if length == 1 and confidence < 36 and step % 4:
                continue
            if not protected and texture.treatment in ("sparse", "answer") and step % 4 and length < 3:
                continue
            strong = step in kicks or step in backbeat or (step % 4 == 2 and texture.feel == "offbeat")
            accent = 1.07 if strong or step == 0 else .77
            delay = swing if step % 4 == 2 else 0
            if performer == "melody_keys":
                delay += 5 if step % 4 == 0 else 12
            elif performer == "melody_response":
                length = max(3, length)
            elif performer == "melody_legato":
                length += 1
            if texture.treatment == "sparse":
                length = max(3, length)
            gain = accent * (.65 + .003 * confidence) * self.random.uniform(.91, 1)
            events[step] = [(performer, int(midi), gain, int(length), delay)]
            # Repeat a stable note as a quiet answer only where the source rests.
            dst = step + 4
            if (answer and length >= 3 and confidence >= 65 and dst < len(contour)
                    and contour[dst][0] < 0 and phrase % 2 == 0
                    and pitches[dst // 4][6] >= 48 and midi % 12 in pitches[dst // 4][3:6]):
                events[dst] = [("lead_echo", int(midi), gain * .43, 2, 0)]
        return events
