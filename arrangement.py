"""Stateful, reproducible rhythmic arrangement at bar boundaries."""

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
        self.intensity = 1.0
        self.energy = .35
        self.energy_goal = .35
        self.energy_bars = 0
        self.arc_random = random.Random(seed ^ 0x4BC1)
        self.style_bars = 0

    def update_energy(self):
        arc = self.rules["energy"]
        if self.energy_bars == 0:
            self.energy_bars = self.arc_random.randint(*arc["bars"])
            self.energy_goal = self.arc_random.uniform(*arc["target"])
        self.energy += (self.energy_goal - self.energy) * arc["smoothing"]
        self.energy_bars -= 1

    def choose_style(self, section):
        styles = self.rules["styles"]
        current = styles[self.style]
        desired = current["phrases"] * self.rules["phrase_bars"]
        if not self.intensity:
            new_style = "original"
        elif current["min_intensity"] > self.intensity:
            new_style = "original"
        elif (section.get("protected") or self.style_bars < desired
              or (not section and self.style_bars < desired + 3)):
            return
        else:
            choices = {
                name: weight * (1.05 - abs(spec["energy"] - self.energy))
                * (1.05 - .60 * abs(spec["energy"] - section.get("energy", .5)))
                for name, weight in current["next"].items()
                if (spec := styles[name])["min_intensity"] <= self.intensity and weight > 0
            }
            new_style = self.style_random.choices(list(choices), list(choices.values()))[0] if choices else "original"
        if new_style != self.style:
            self.style = new_style
            self.style_bars = 0

    def lead_events(self, contour, spec, entering=False):
        """Keep lead notes in source order and vary their attack or voicing."""
        events = {}
        for step, (midi, onset, duration) in enumerate(contour):
            if midi < 0 or not (onset or (step == 0 and entering)):
                continue
            if (spec["legato"] and duration == 1 and 0 < step < len(contour) - 1
                    and contour[step - 1][0] >= 0
                    and contour[step - 1][0] == contour[step + 1][0]):
                continue
            dst = step
            if (step in spec["delay_steps"] and step + 1 < len(contour) and duration > 1
                    and self.style_random.random() < .8 * self.intensity):
                dst += 1
                duration -= 1
            gain = self.style_random.uniform(.82, 1) * (1 if step % 4 == 0 else .84)
            events[dst] = [(self.style, int(midi), gain, int(duration))]
            if spec["shadow"] and step % 4 == 0:
                events[dst].append(("melody_shadow", int(midi) - 12, gain * .48, int(duration)))
            if spec["response"] and duration >= 4 and dst + 3 < len(contour):
                events[dst + 3] = [("lead_echo", int(midi), gain * .64, int(duration) - 3)]
        return events

    def next_bar(self, section=None, intensity=1.0):
        self.intensity = intensity
        self.update_energy()
        self.choose_style(section or {})
        maximum = self.rules["styles"][self.style]["phrases"] * self.rules["phrase_bars"] + 3
        self.style_bars = min(self.style_bars + 1, maximum)
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
        self.bass_steps = (
            variants[position % len(variants)] if variants else rhythm.get("bass_steps", ())
        )
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
