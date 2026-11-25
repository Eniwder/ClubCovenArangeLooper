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
        self.style_age = 0
        self.style_random = random.Random(seed ^ 0x2764)
        self.tone_pattern = {}
        self.melody_pattern = {}
        self.melody_rotation = 0

    def next_bar(self):
        phrase_bars = self.rules["phrase_bars"]
        position = self.bar % phrase_bars
        states = self.rules["states"]
        if position == 0:
            styles = self.rules["styles"]
            if self.style_age >= styles[self.style]["phrases"]:
                choices = styles[self.style]["next"]
                self.style = self.style_random.choices(list(choices), list(choices.values()))[0]
                self.style_age = 0
            self.style_age += 1
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
        pattern = self.rules["styles"][self.style]["pattern"]
        self.tone_pattern = {
            step: (tone, gain * self.style_random.uniform(.88, 1.0))
            for step, tone, gain in pattern
            if position % 2 == 0 or gain >= .65 or self.style_random.random() < .60
        }
        melody = self.rules["styles"][self.style].get("melody")
        self.melody_pattern = {}
        if melody:
            if position == 0:
                self.melody_rotation = self.style_random.randrange(4)
            score = melody["answer"] if position % 2 else melody["call"]
            self.melody_pattern = {
                step: ((beat + self.melody_rotation) % 4, tone,
                       gain * self.style_random.uniform(.86, 1))
                for step, beat, tone, gain in score
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
        steps = self.motif.copy()
        if position % 2:
            alternatives = [step for step in pool if step not in steps]
            if alternatives:
                steps[self.random.randrange(count)] = self.random.choice(alternatives)
        if fill:
            steps = sorted(set(steps[: max(0, count - 2)] + self.rules["fill_steps"]))
        velocity = current["velocity"]
        jitter = self.rules["velocity_jitter"]
        plan = {}
        for step in sorted(steps):
            hit = "fill" if fill and step in self.rules["fill_steps"] else (
                "open" if step % 4 == 2 and position % 2 == 1 else "closed"
            )
            gain = min(1, velocity * self.random.uniform(1 - jitter, 1 + jitter))
            plan[step] = hit, gain
        return plan
