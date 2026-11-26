"""Pitch candidates from the recording and short synthesized instruments."""

import numpy as np


def analyse_pitches(source, track):
    beat = track.step * 4
    notes = np.arange(35, 81)
    frequencies = 440 * 2 ** ((notes - 69) / 12)
    bins = np.rint(frequencies * 65536 / track.rate).astype(int)
    window = np.hanning(beat)
    pitches = []
    # Rank triads from the observed chroma with a small prior for the bass.
    triads = [(root, (root + third) % 12, (root + 7) % 12)
              for root in range(12) for third in (3, 4)]
    for start in range(track.first_bar, track.end, beat):
        mono = source[start:start + beat].mean(axis=1)
        spectrum = np.abs(np.fft.rfft(mono * window, n=65536))
        power = spectrum[bins]
        bass = int(notes[:22][np.argmax(power[:22])] % 12)
        chroma = np.bincount(notes[12:] % 12, weights=power[12:], minlength=12)
        order = np.argsort(chroma)[::-1]
        upper = next((int(pc) for pc in order if (pc - bass) % 12 not in (0, 1, 6, 11)
                      and chroma[pc] >= chroma[order[0]] * 0.48), bass)
        ranked = sorted((sum(chroma[pc] for pc in pcs) + .18 * chroma.max() * (bass in pcs), pcs)
                        for pcs in triads)
        score, chord = ranked[-1]
        fifth = chord[2] if bass in chord and chroma[chord[2]] >= .35 * chroma.max() else upper
        certainty = min(1, 5 * (score - ranked[-2][0]) / max(score, 1e-6))
        pitches.append((bass, upper, fifth, chord[0], chord[1], chord[2], round(100 * certainty)))
    return np.asarray(pitches, dtype=np.int16)


def analyse_melody(source, track):
    """Track a prominent upper voice on the sixteenth-note grid."""
    notes = np.arange(60, 89)
    hz = 440 * 2 ** ((notes - 69) / 12)
    size = 16384
    bins = np.rint(hz * size / track.rate).astype(int)
    window = np.hanning(track.step)
    scores = []
    strengths = []
    for start in range(track.first_bar, track.end, track.step):
        mono = source[start:start + track.step].mean(axis=1)
        spectrum = np.abs(np.fft.rfft(mono * window, n=size))
        fundamental = np.maximum.reduce([spectrum[bins + offset] for offset in (-1, 0, 1)])
        harmonic = spectrum[bins * 2] + .45 * spectrum[bins * 3]
        sub = spectrum[bins // 2]
        salience = (fundamental + .32 * harmonic) / np.sqrt(1 + sub / (fundamental + 1e-4))
        strengths.append(fundamental)
        scores.append(np.log(salience / max(salience.max(), 1e-6) + .012))

    # Prefer a continuous melodic line over isolated loud partials.
    distance = np.abs(notes[:, None] - notes[None, :])
    transition = -.09 * distance - .18 * (distance > 0) - .36 * (distance > 7)
    previous = scores[0]
    parents = []
    for frame in scores[1:]:
        candidates = previous[:, None] + transition
        parents.append(candidates.argmax(axis=0))
        previous = frame + candidates.max(axis=0)
    index = int(previous.argmax())
    melody = [int(notes[index])]
    for parent in reversed(parents):
        index = parent[index]
        melody.append(int(notes[index]))
    melody = np.array(melody[::-1], dtype=np.int16)
    strength = np.asarray(strengths)[np.arange(len(melody)), melody - notes[0]]
    active = strength >= max(float(np.percentile(strength, 45)) * .21, 1e-5)
    grid_scores = np.asarray(scores)
    picked = grid_scores[np.arange(len(melody)), melody - notes[0]]
    grid_scores[np.arange(len(melody)), melody - notes[0]] = -np.inf
    margin = picked - grid_scores.max(axis=1)
    persistence = np.r_[False, melody[1:] == melody[:-1]] | np.r_[melody[:-1] == melody[1:], False]
    certainty = np.clip(.48 + .65 * margin + .17 * persistence, 0, 1)
    events = np.zeros((len(melody), 4), dtype=np.int16)
    events[:, 0] = np.where(active, melody, -1)
    events[:, 3] = np.where(active, np.rint(100 * certainty), 0)
    for step in range(len(melody) - 1, -1, -1):
        if not active[step]:
            continue
        onset = (step == 0 or not active[step - 1] or melody[step] != melody[step - 1]
                 or strength[step] > 2.2 * strength[step - 1])
        events[step, 1] = onset
        sustain = (step + 1 < len(melody) and events[step + 1, 0] == melody[step]
                   and not events[step + 1, 1])
        events[step, 2] = min(32, int(events[step + 1, 2]) + 1) if sustain else 1
    return melody, events


def make_instruments(rate, styles):
    instruments = {}
    for name, spec in styles.items():
        t = np.arange(round(rate * spec['length'])) / rate
        shape = np.minimum(t / spec['attack'], 1) * np.clip((t[-1] - t) / .09, 0, 1)
        notes = {}
        for midi in range(spec['register'], spec['register'] + spec.get('register_span', 12)):
            phase = 2 * np.pi * (440 * 2 ** ((midi - 69) / 12)) * t
            if spec.get("vibrato"):
                phase += spec["vibrato"] * np.sin(2 * np.pi * 5.1 * t) * np.clip((t - .08) / .24, 0, 1)
            if spec.get("fm_index"):
                phase += spec["fm_index"] * np.sin(2 * phase) * np.exp(-7 * t)
            signal = sum(
                amplitude * np.sin(phase * (harmonic + spec['stretch'] * harmonic ** 2))
                * np.exp(-t * (spec['decay'] + harmonic * spec['brightness_decay']))
                for harmonic, amplitude in enumerate(spec['partials'], 1)
            ) * shape
            if spec.get("unison"):
                detuned = np.sin(phase * 1.0018) + .32 * np.sin(phase * 2.004)
                signal += spec["unison"] * detuned * shape * np.exp(-t * spec["decay"])
            signal *= spec['level'] / max(np.max(np.abs(signal)), 1e-6)
            stereo = (signal[:, None] * spec['pan']).astype(np.float32)
            if spec.get("echo"):
                delay = round(rate * .093)
                tail = stereo[:, ::-1] * spec["echo"]
                stereo = np.pad(stereo, ((0, delay), (0, 0)))
                stereo[delay:] += tail
            notes[midi] = stereo
        instruments[name] = notes
    return instruments


def make_kit(rate):
    """Short drum voices without pitched material from the recording."""
    kit = {}
    random = np.random.default_rng(714)
    for name, seconds, level in (("kick", .115, .23), ("fill", .105, .14),
                                 ("closed", .045, .075), ("open", .110, .085)):
        t = np.arange(round(rate * seconds)) / rate
        noise = random.standard_normal(len(t))
        bright = np.diff(noise, prepend=0)
        if name == "kick":
            phase = 2 * np.pi * (43 * t + 52 * .022 * (1 - np.exp(-t / .022)))
            wave = np.sin(phase) * np.exp(-t / .035) + .05 * bright * np.exp(-t / .006)
        elif name == "fill":
            wave = (.72 * noise + .20 * np.sin(2 * np.pi * 177 * t)) * np.exp(-t / .027)
        else:
            wave = bright * np.exp(-t / (seconds / 4))
        wave *= np.clip(t / .002, 0, 1) * np.clip((t[-1] - t) / .018, 0, 1)
        kit[name] = (wave[:, None] * (level / np.max(np.abs(wave))) * (1, .93)).astype(np.float32)
    return kit
