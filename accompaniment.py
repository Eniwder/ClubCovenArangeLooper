"""Pitch candidates from the recording and short synthesized instruments."""

import numpy as np


def analyse_pitches(source, track):
    beat = track.step * 4
    notes = np.arange(35, 89)
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
        # Discount upper partials of lower notes before choosing a lead candidate.
        # A stereo mix still yields a candidate, rather than a clean transcription.
        lead_bins = bins[25:]
        partial = spectrum[np.rint(lead_bins / 2).astype(int)]
        salience = power[25:] / (1 + partial / max(np.median(power), 1e-6))
        lead = int(notes[25:][np.argmax(salience)] % 12)
        pitches.append((bass, upper, lead))
    return pitches


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
