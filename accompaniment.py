"""Pitch candidates from the recording and short synthesized instruments."""

import numpy as np


def analyse_pitches(source, track):
    beat = track.step * 4
    notes = np.arange(35, 81)
    frequencies = 440 * 2 ** ((notes - 69) / 12)
    bins = np.rint(frequencies * 65536 / track.rate).astype(int)
    window = np.hanning(beat)
    pitches = []
    for start in range(track.first_bar, track.end, beat):
        mono = source[start:start + beat].mean(axis=1)
        spectrum = np.abs(np.fft.rfft(mono * window, n=65536))
        power = spectrum[bins]
        bass = int(notes[:22][np.argmax(power[:22])] % 12)
        chroma = np.bincount(notes[12:] % 12, weights=power[12:], minlength=12)
        order = np.argsort(chroma)[::-1]
        upper = next((int(pc) for pc in order if (pc - bass) % 12 not in (0, 1, 6, 11)
                      and chroma[pc] >= chroma[order[0]] * 0.48), bass)
        pitches.append((bass, upper))
    return pitches


def analyse_melody(source, track):
    """Track a prominent upper voice on the sixteenth-note grid."""
    notes = np.arange(60, 89)
    hz = 440 * 2 ** ((notes - 69) / 12)
    size = 16384
    bins = np.rint(hz * size / track.rate).astype(int)
    window = np.hanning(track.step)
    scores = []
    for start in range(track.first_bar, track.end, track.step):
        mono = source[start:start + track.step].mean(axis=1)
        spectrum = np.abs(np.fft.rfft(mono * window, n=size))
        fundamental = np.maximum.reduce([spectrum[bins + offset] for offset in (-1, 0, 1)])
        harmonic = spectrum[bins * 2] + .45 * spectrum[bins * 3]
        sub = spectrum[bins // 2]
        salience = (fundamental + .32 * harmonic) / np.sqrt(1 + sub / (fundamental + 1e-4))
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
    return melody[::-1]


def make_instruments(rate, styles):
    instruments = {}
    for name, spec in styles.items():
        t = np.arange(round(rate * spec['length'])) / rate
        shape = np.minimum(t / spec['attack'], 1) * np.clip((t[-1] - t) / .09, 0, 1)
        notes = {}
        for midi in range(spec['register'], spec['register'] + spec.get('register_span', 12)):
            phase = 2 * np.pi * (440 * 2 ** ((midi - 69) / 12)) * t
            signal = sum(
                amplitude * np.sin(phase * (harmonic + spec['stretch'] * harmonic ** 2))
                * np.exp(-t * (spec['decay'] + harmonic * spec['brightness_decay']))
                for harmonic, amplitude in enumerate(spec['partials'], 1)
            ) * shape
            signal *= spec['level'] / max(np.max(np.abs(signal)), 1e-6)
            notes[midi] = (signal[:, None] * spec['pan']).astype(np.float32)
        instruments[name] = notes
    return instruments
