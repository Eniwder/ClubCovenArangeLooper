"""Stateful, reproducible rhythmic arrangement at bar boundaries."""

from collections import deque
import random


class Arrangement:
    def __init__(self, rules, seed):
        self.rules = rules
        self.random = random.Random(seed)
        self.state = "original"
        self.age = 0
        self.bar = 0
        self.motif = []
        self.style = "original"
        self.style_random = random.Random(seed ^ 0x2764)
        self.tone_pattern = {}
        self.bass_steps = ()
        self.harmony_steps = ()
        self.kit_pattern = {}
        self.layer_random = random.Random(seed ^ 0x796B)
        self.layer_variation = 0
        self.intensity = 1.0
        self.energy = .35
        self.energy_goal = .35
        self.energy_bars = 0
        self.arc_wait = 0
        self.arc_random = random.Random(seed ^ 0x4BC1)
        self.style_bars = 0
        self.recent_styles = deque(["original"], maxlen=6)
        self.recent_roles = deque(["recording"], maxlen=6)
        self.arc = "rest"
        self.recent_arcs = deque(["rest"], maxlen=3)

    def update_energy(self, section):
        arc = self.rules["energy"]
        if self.energy_bars == 0 and (section or not self.bar or self.arc_wait == 2):
            states = arc["states"]
            current = states[self.arc]
            if self.bar:
                choices = current["next"]
                weights = [weight * (.68 ** self.recent_arcs.count(name))
                           * (1 - .40 * abs(states[name]["target"] - section.get("energy", .5)))
                           for name, weight in choices.items()]
                self.arc = self.arc_random.choices(list(choices), weights)[0]
                self.recent_arcs.append(self.arc)
                current = states[self.arc]
            self.energy_bars = self.arc_random.randint(*current["bars"])
            self.energy_goal = current["target"]
            self.arc_wait = 0
        self.energy += (self.energy_goal - self.energy) * arc["smoothing"]
        if self.energy_bars:
            self.energy_bars -= 1
        else:
            self.arc_wait += 1

    def choose_style(self, section, confidence=.6, change_style=None):
        styles = self.rules["styles"]
        current = styles[self.style]
        desired = current["phrases"] * self.rules["phrase_bars"]
        if not self.intensity:
            if self.style == "original":
                return
            new_style = "original"
        elif current["min_intensity"] > self.intensity:
            new_style = "original"
        elif change_style is False:
            return
        elif change_style is None and (section.get("protected") or self.style_bars < desired
              or (not section and self.style_bars < desired + 3)):
            return
        else:
            choices = {}
            favored = self.rules["energy"]["states"][self.arc]["favor"]
            for name, weight in current["next"].items():
                spec = styles[name]
                if spec["min_intensity"] > self.intensity or weight <= 0:
                    continue
                mode = spec["lead_mode"]
                role = spec["role"]
                weight *= (1.05 - abs(spec["energy"] - self.energy))
                weight *= 1.05 - .60 * abs(spec["energy"] - section.get("energy", .5))
                weight *= .64 ** self.recent_styles.count(name) * .84 ** self.recent_roles.count(role)
                if mode in favored:
                    weight *= 1.6
                if mode in ("replace", "trade"):
                    weight *= .65 + .80 * confidence
                    if list(self.recent_roles)[-2:] == ["recording"] * 2:
                        weight *= 1.7
                choices[name] = weight
            if change_style and len(choices) > 1:
                choices.pop(self.style, None)
            new_style = self.style_random.choices(list(choices), list(choices.values()))[0] if choices else "original"
        self.style = new_style
        self.style_bars = 0
        self.recent_styles.append(new_style)
        self.recent_roles.append(styles[new_style]["role"])

    def lead_events(self, contour, spec, entering=False):
        """Keep lead notes in source order and vary their attack or voicing."""
        events = {}
        style = self.rules["styles"][self.style]
        performer = spec.get("performer", self.style)
        exchange = spec.get("exchange")
        if exchange and (self.style_bars - 1) % 4 >= 2:
            performer = exchange
        instrument = self.rules["instruments"].get(performer, style)
        expression = instrument.get("expression", "neutral")
        for step, (midi, onset, duration, confidence) in enumerate(contour):
            if midi < 0 or not (onset or (step == 0 and entering)):
                continue
            if (spec["legato"] and duration == 1 and 0 < step < len(contour) - 1
                    and contour[step - 1][0] >= 0
                    and contour[step - 1][0] == contour[step + 1][0]):
                continue
            if duration == 1 and step % 4 and confidence < (.46 if expression == "bell" else .28) * 100:
                continue
            if expression == "bell" and step % 4 and duration == 1:
                continue
            dst = step
            if (step in spec["delay_steps"] and step + 1 < len(contour) and duration > 1
                    and self.style_random.random() < .8 * self.intensity):
                dst += 1
                duration -= 1
            gain = self.style_random.uniform(.82, 1) * (1 if step % 4 == 0 else .84)
            gain *= .76 + .24 * confidence / 100
            delay = 0
            if expression == "piano":
                delay = 5 if step % 4 == 0 else 13
                gain *= 1.08 if step in (0, 8) else .93
            elif expression == "bell":
                duration = max(5, duration)
                delay = 12
            elif expression == "strings":
                duration += 1
                delay = 0
            events[dst] = [(performer, int(midi), gain, int(duration), delay)]
            if spec["shadow"] and step % 4 == 0:
                events[dst].append(("melody_shadow", int(midi) - 12, gain * .48, int(duration), delay))
            if spec["response"] and duration >= 4 and dst + 3 < len(contour):
                events[dst + 3] = [("lead_echo", int(midi), gain * .64, int(duration) - 3, 0)]
        return events

    def next_bar(self, section=None, intensity=1.0, confidence=.6, change_style=None):
        self.intensity = intensity
        self.update_energy(section or {})
        self.choose_style(section or {}, confidence, change_style)
        self.style_bars += 1
        phrase_bars = self.rules["phrase_bars"]
        position = self.bar % phrase_bars
        states = self.rules["states"]
        if position == 0:
            current = states[self.state]
            if self.age >= current["min_phrases"]:
                choices = dict(current["next"])
                if self.age >= current["max_phrases"]:
                    choices.pop(self.state, None)
                new_state = self.random.choices(list(choices), list(choices.values()))[0]
                if new_state != self.state:
                    self.state = new_state
                    self.age = 0
            self.age += 1
        self.bar += 1
        style = self.rules["styles"][self.style]
        pattern = style["pattern"]
        rhythm = style.get("rhythm", {})
        variants = rhythm.get("bass_variants")
        if position == 0 or self.style_bars == 1:
            self.layer_variation = self.layer_random.getrandbits(3)
        self.bass_steps = (
            variants[position % len(variants)] if variants else rhythm.get("bass_steps", ())
        )
        self.harmony_steps = rhythm.get("harmony", {}).get("steps", ())
        self.kit_pattern = rhythm.get("kit", {})
        if position % 2 and self.layer_variation & 1:
            self.bass_steps = self.bass_steps[::2]
        if position % 2 and self.layer_variation & 2:
            self.harmony_steps = self.harmony_steps[::2]
        if position % 2 and self.layer_variation & 4:
            self.kit_pattern = {step: hit for step, hit in self.kit_pattern.items()
                                if hit[0] not in ("closed", "open")}
        self.tone_pattern = {
            step: (tone, gain * self.style_random.uniform(.88, 1.0))
            for step, tone, gain in pattern
            if position % 2 == 0 or gain >= .65 or self.style_random.random() < .36 + .48 * self.energy
        }

        current = states[self.state]
        count = current["steps"]
        if not count:
            return {}
        fill = self.state == "lift" and position == phrase_bars - 1
        pool = self.rules["offbeats"]
        if self.state in ("groove", "lift"):
            pool = pool + self.rules["syncopations"]
        if position == 0:
            self.motif = sorted(self.random.sample(pool, count))
        if not self.intensity:
            return {}
        steps = self.motif.copy()
        if position % 2:
            alternatives = [step for step in pool if step not in steps]
            if alternatives:
                steps[self.random.randrange(count)] = self.random.choice(alternatives)
        if fill:
            steps = sorted(set(steps[: max(0, count - 2)] + self.rules["fill_steps"]))
        velocity = current["velocity"] * (.60 + .40 * self.energy) * self.intensity
        jitter = self.rules["velocity_jitter"]
        plan = {}
        for step in sorted(steps):
            hit = "fill" if fill and step in self.rules["fill_steps"] else (
                "open" if step % 4 == 2 and position % 2 == 1 else "closed"
            )
            gain = min(1, velocity * self.random.uniform(1 - jitter, 1 + jitter))
            plan[step] = hit, gain
        return plan
