"""Prepare a backing mix with the tracked lead and other tonal peaks reduced."""

import numpy as np


def make_backing(audio, track, settings):
    size = 4096
    hop = size // 4
    window = np.sin(np.pi * (np.arange(size) + .5) / size).astype(np.float32)
    frequencies = np.fft.rfftfreq(size, 1 / track.rate)
    midrange = np.clip((frequencies - 170) / 150, 0, 1)
    midrange *= np.clip((6500 - frequencies) / 3300, 0, 1)

    # Cache smooth notches at the fundamental and upper harmonics of each
    # tracked note. Apply a shared mask to preserve the stereo image.
    notches = {}
    for midi in set(track.melody):
        fundamental = 440 * 2 ** ((midi - 69) / 12)
        centers = fundamental * np.arange(1, 11)
        width = 16 + .021 * centers
        distance = (frequencies[:, None] - centers) / width
        notches[midi] = np.exp(-.5 * distance ** 2).max(axis=1)

    # Only prepare the part that can enter a melody arrangement. Four
    # overlapping sine windows have a constant summed squared gain of two.
    start = track.first_bar
    lead_in = size
    padded = np.pad(audio[start - lead_in:], ((0, size), (0, 0)))
    backing = np.zeros_like(padded)
    previous = np.zeros(size // 2 + 1)
    for offset in range(0, len(padded) - size + 1, hop):
        frame = padded[offset:offset + size]
        spectrum = np.fft.rfft(frame * window[:, None], axis=0)
        magnitude = np.abs(spectrum).max(axis=1)
        index = (offset + size // 2 - lead_in) // track.step
        midi = track.melody[min(max(index, 0), len(track.melody) - 1)]
        # Narrow persistent peaks catch melodic partials that escaped the
        # monophonic tracker. Protect fast rises from excessive attenuation.
        skirt = np.pad(magnitude, (4, 4), mode="edge")
        floor = (skirt[:-8] + skirt[8:]) * .5
        tonal = np.clip(1 - 2 * floor / np.maximum(magnitude, 1e-6), 0, 1)
        attack = np.clip(1 - 1.8 * previous / np.maximum(magnitude, 1e-6), 0, 1)
        previous = magnitude
        reduction = np.maximum(settings["harmonic_reduction"] * notches[midi],
                               settings["mid_reduction"] * tonal)
        gain = 1 - midrange * reduction * (1 - .40 * attack)
        wave = np.fft.irfft(spectrum * gain[:, None], n=size, axis=0)
        backing[offset:offset + size] += wave * (.5 * window[:, None])

    result = audio.copy()
    result[start:] = backing[lead_in:lead_in + len(audio) - start]
    return result
