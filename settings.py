"""Validate the editable Club Coven configuration before opening an audio stream."""

import math


def validate(config, rules):
    def number(value, low, high, name, integer=False):
        if (type(value) not in (int, float) or not math.isfinite(value)
                or not low <= value <= high or (integer and type(value) is not int)):
            raise ValueError(f"{name} は {low}〜{high} の{'整数' if integer else '数値'}にしてください。")

    def steps(values, name):
        for value in values:
            number(value, 0, 15, name, True)

    def weights(mapping, options, name):
        if not mapping or mapping.keys() - options or not any(mapping.values()):
            raise ValueError(f"{name} の遷移先または重みが不正です。")
        for value in mapping.values():
            number(value, 0, 1000, name)

    def instrument(spec, name):
        number(spec["register"], 24, 100, name, True)
        number(spec.get("register_span", 12), 12, 48, name, True)
        number(spec["length"], .1, 4, name)
        number(spec["attack"], .001, spec["length"], name)
        for key in ("decay", "brightness_decay"):
            number(spec[key], 0, 100, name)
        for key in ("stretch", "level"):
            number(spec[key], 0, 1, name)
        if len(spec["pan"]) != 2 or not 1 <= len(spec["partials"]) <= 12:
            raise ValueError(f"{name} の音色設定が不正です。")
        for value in spec["pan"] + spec["partials"]:
            number(value, 0, 2, name)
        for key in ("unison", "echo"):
            number(spec.get(key, 0), 0, .5, name)

    number(config["target_lufs"], -35, -8, "target_lufs")
    number(config["sample_rate"], 32000, 96000, "sample_rate", True)
    number(config["bpm"], 60, 180, "bpm")
    number(config["crossfade_ms"], 1, 500, "crossfade_ms")
    for key in ("loop_start", "loop_end", "first_arranged_bar"):
        number(config[key], 1, 360, key)
    if config["beats_per_bar"] != 4 or not config["first_arranged_bar"] < config["loop_start"] < config["loop_end"]:
        raise ValueError("拍子またはループ位置が不正です。")
    step = round(config["sample_rate"] * 15 / config["bpm"])
    loop = round(config["loop_end"] * config["sample_rate"]) - round(config["loop_start"] * config["sample_rate"])
    if loop % (step * 16):
        raise ValueError("ループの長さは小節単位で設定してください。")
    for item in config["sections"]:
        number(item["start"], config["first_arranged_bar"], config["loop_end"], "section.start")
        if (round(item["start"] * config["sample_rate"])
                - round(config["first_arranged_bar"] * config["sample_rate"])) % (step * 16):
            raise ValueError("section.start は小節の先頭に合わせてください。")
        number(item.get("energy", .5), 0, 1, "section.energy")
    for name, spec in config["hits"].items():
        number(spec["time"], .025, 360, name)
        number(spec["length_ms"], 5, 15000 / config["bpm"], name)
        number(spec["low_hz"], 0, config["sample_rate"] / 2, name)
        number(spec["high_hz"], spec["low_hz"] + 1, config["sample_rate"] / 2, name)

    number(rules["phrase_bars"], 2, 16, "phrase_bars", True)
    number(rules["hit_peak"], .01, .5, "hit_peak")
    number(rules["velocity_jitter"], 0, 1, "velocity_jitter")
    for key in ("harmonic_reduction", "mid_reduction"):
        number(rules["backing"][key], 0, 1, key)
    for key in ("offbeats", "syncopations", "fill_steps"):
        steps(rules[key], key)
    energy = rules["energy"]
    number(energy["smoothing"], .01, 1, "energy.smoothing")
    low, high = energy["bars"]
    number(low, 1, 100, "energy.bars", True)
    number(high, low, 100, "energy.bars", True)
    low, high = energy["target"]
    number(low, 0, 1, "energy.target")
    number(high, low, 1, "energy.target")
    styles = rules["styles"]
    if styles["original"]["min_intensity"] != 0 or styles["original"]["original_gain"] != 1:
        raise ValueError("original の強度とゲインは 0 と 1 に設定してください。")
    for name, spec in styles.items():
        number(spec["phrases"], 1, 12, name, True)
        number(spec.get("trim_db", 0), -12, 12, f"{name}.trim_db")
        for key in ("min_intensity", "energy", "original_gain", "drum_gain"):
            number(spec[key], 0, 1, f"{name}.{key}")
        weights(spec["next"], styles.keys(), name)
        for position, tone, gain in spec["pattern"]:
            steps([position], name)
            number(tone, 0, 2, name, True)
            number(gain, 0, 1, name)
        if spec["pattern"] or spec.get("melody"):
            instrument(spec, name)
        if spec.get("melody"):
            steps(spec["melody"]["delay_steps"], name)
            if not spec["register"] <= 60 or spec["register"] + spec.get("register_span", 12) <= 88:
                raise ValueError(f"{name} の音域は MIDI 60〜88 を含めてください。")
        rhythm = spec.get("rhythm", {})
        steps(rhythm.get("bass_steps", ()), name)
        for variant in rhythm.get("bass_variants", ()):
            steps(variant, name)
        if rhythm and rhythm.get("bass_instrument", "replacement_bass") not in rules["instruments"]:
            raise ValueError(f"{name} のベース音色がありません。")
        for position, (hit, gain) in rhythm.get("kit", {}).items():
            steps([int(position)], name)
            if hit not in config["hits"]:
                raise ValueError(f"打音 {hit} がありません。")
            number(gain, 0, 2, name)
    if (styles["original"]["pattern"] or styles["original"].get("melody")
            or styles["original"].get("rhythm") or styles["original"].get("trim_db", 0)):
        raise ValueError("original に追加演奏を設定できません。")
    for name, spec in rules["instruments"].items():
        instrument(spec, name)
    for name, start, end in (("melody_shadow", 48, 76), ("lead_echo", 60, 88)):
        instrument_spec = rules["instruments"][name]
        if not instrument_spec["register"] <= start or instrument_spec["register"] + instrument_spec.get("register_span", 12) <= end:
            raise ValueError(f"{name} の音域は MIDI {start}〜{end} を含めてください。")
    states = rules["states"]
    for name, spec in states.items():
        count = len(rules["offbeats"]) + (len(rules["syncopations"]) if name in ("groove", "lift") else 0)
        number(spec["steps"], 0, count, name, True)
        number(spec["velocity"], 0, 1, name)
        number(spec["min_phrases"], 1, 20, name, True)
        number(spec["max_phrases"], spec["min_phrases"], 20, name, True)
        weights(spec["next"], states.keys(), name)
        if not any(weight > 0 for other, weight in spec["next"].items() if other != name):
            raise ValueError(f"{name} には別の状態への遷移が必要です。")
