import argparse
from pathlib import Path
import sys
import wave

import numpy as np

from engine import Engine
from material import ROOT, Track, read_json


def preview(path, seconds, engine):
    block = np.empty((4096, 2), dtype=np.float32)
    remaining = engine.track.frames(seconds)
    fade = engine.track.frames(0.03)
    with wave.open(path, "wb") as output:
        output.setparams((2, 2, engine.track.rate, 0, "NONE", ""))
        while remaining:
            chunk = block[:min(remaining, len(block))]
            engine.render(chunk)
            if remaining - len(chunk) < fade:
                distance = remaining - 1 - np.arange(len(chunk))
                chunk *= np.minimum(distance / fade, 1)[:, None]
            output.writeframesraw(np.rint(chunk * 32767).astype("<i2").tobytes())
            remaining -= len(chunk)


def run_gui(track, rules, initial_seed):
    import tkinter as tk
    from tkinter import ttk
    import sounddevice as sd

    root = tk.Tk()
    root.title("Club Coven")
    root.resizable(False, False)
    body = ttk.Frame(root, padding=16)
    body.grid(sticky="nsew")
    seed = tk.StringVar(value=str(initial_seed))
    status = tk.StringVar(value="停止中")
    volume = 0.7
    stream = None
    stopping = False
    current_gain = 0.0
    underrun = False

    def update_volume(value):
        nonlocal volume
        volume = float(value) / 100

    def close_stream():
        nonlocal stream
        if stream is not None:
            stream.close()
            stream = None

    def play():
        nonlocal stream, stopping, current_gain, underrun
        if stream is not None:
            return
        try:
            engine = Engine(track, rules, int(seed.get()))
        except ValueError:
            status.set("シードは整数で入力してください。")
            return
        stopping = False
        current_gain = 0.0
        underrun = False

        def callback(out, frames, timing, flags):
            nonlocal current_gain, underrun
            underrun |= bool(flags.output_underflow)
            engine.render(out)
            target = 0.0 if stopping else volume
            if target != current_gain:
                gain = np.linspace(current_gain, target, frames, dtype=np.float32)
                out *= gain[:, None]
                current_gain = target
            else:
                out *= current_gain
            if stopping:
                raise sd.CallbackStop

        try:
            stream = sd.OutputStream(samplerate=track.rate, channels=2, dtype="float32",
                                     latency="high", callback=callback)
            stream.start()
        except sd.PortAudioError as error:
            close_stream()
            status.set(str(error))
            return
        play_button.state(["disabled"])
        stop_button.state(["!disabled"])
        seed_entry.state(["disabled"])
        status.set("再生中")

    def stop():
        nonlocal stopping
        if stream is not None:
            stopping = True
            stop_button.state(["disabled"])

    def poll():
        if stream is not None:
            if not stream.active:
                close_stream()
                play_button.state(["!disabled"])
                stop_button.state(["disabled"])
                seed_entry.state(["!disabled"])
                status.set("停止中" if stopping else "音声出力が停止しました。")
            elif underrun and not stopping:
                status.set("再生中（音声出力の遅延を検出）")
        root.after(100, poll)

    play_button = ttk.Button(body, text="再生", command=play)
    play_button.grid(row=0, column=0, sticky="ew", padx=(0, 8))
    stop_button = ttk.Button(body, text="停止", command=stop, state="disabled")
    stop_button.grid(row=0, column=1, sticky="ew")
    ttk.Label(body, text="音量").grid(row=1, column=0, sticky="w", pady=(12, 0))
    slider = ttk.Scale(body, from_=0, to=100, command=update_volume)
    slider.set(70)
    slider.grid(row=1, column=1, sticky="ew", pady=(12, 0))
    ttk.Label(body, text="乱数シード").grid(row=2, column=0, sticky="w", pady=(8, 0))
    seed_entry = ttk.Entry(body, textvariable=seed, width=16)
    seed_entry.grid(row=2, column=1, sticky="ew", pady=(8, 0))
    ttk.Label(body, textvariable=status, wraplength=270).grid(
        row=3, column=0, columnspan=2, sticky="w", pady=(12, 0)
    )

    def quit_app():
        close_stream()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", quit_app)
    root.after(100, poll)
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Club Coven endless arrangement")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview", metavar="WAV", help="render a WAV instead of opening the UI")
    parser.add_argument("--seconds", type=int, default=115)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    try:
        rules = read_json("rules.json")
        config = read_json("track.json")
        if args.preview and Path(args.preview).resolve() == (ROOT / config["source"]).resolve():
            parser.error("試聴 WAV には原曲とは別の出力先を指定してください。")
        track = Track(config, rules)
        if args.preview:
            preview(args.preview, args.seconds, Engine(track, rules, args.seed))
        else:
            run_gui(track, rules, args.seed)
    except (OSError, RuntimeError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
