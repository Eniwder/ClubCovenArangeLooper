import argparse
import json
from pathlib import Path
from random import getrandbits
import sys
import wave

import numpy as np

from engine import Engine
from material import ROOT, Track, read_json
from settings import validate


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


def run_gui(track, rules, initial_seed, initial_intensity=1.0, controls=None):
    import tkinter as tk
    from tkinter import filedialog, ttk
    import sounddevice as sd

    root = tk.Tk()
    root.title("Club Coven")
    root.resizable(False, False)
    body = ttk.Frame(root, padding=16)
    body.grid(sticky="nsew")
    seed = tk.StringVar(value=str(initial_seed))
    status = tk.StringVar(value="停止中")
    details = tk.StringVar()
    recent = tk.StringVar()
    intensity_label = tk.StringVar()
    intensity = initial_intensity
    engine = None
    volume = 0.7
    stream = None
    stopping = False
    current_gain = 0.0
    underrun = False
    tray = None

    def update_volume(value):
        nonlocal volume
        volume = float(value) / 100

    def update_intensity(value):
        nonlocal intensity
        intensity = float(value) / 100
        intensity_label.set(f"編曲強度 {intensity:.0%}")
        if engine is not None:
            engine.intensity = intensity

    def close_stream():
        nonlocal stream
        if stream is not None:
            stream.close()
            stream = None

    def play():
        nonlocal stream, stopping, current_gain, underrun, engine
        if stream is not None:
            return
        try:
            engine = Engine(track, rules, int(seed.get()), intensity, controls)
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
        random_seed_button.state(["disabled"])
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
                random_seed_button.state(["!disabled"])
                status.set("停止中" if stopping else "音声出力が停止しました。")
            elif underrun and not stopping:
                status.set("再生中（音声出力の遅延を検出）")
            if engine.snapshot:
                bar, style, energy, applied, parts = engine.snapshot
                details.set(
                    f"{style}  |  エネルギー {energy:.0%}  |  適用強度 {applied:.0%}\n"
                    f"経過 {engine.elapsed / track.rate:.1f} 秒  |  原曲位置 {engine.source_pos / track.rate:.1f} 秒\n"
                    f"小節 {bar + 1}  |  シード {engine.seed}  |  {parts}"
                )
                history = engine.history.copy()
                recent.set(" → ".join(f"{position / track.rate:.0f}s {name}"
                                      for position, _, name in list(history)[-4:]))
                export_button.state(["disabled"] if engine.controls_truncated else ["!disabled"])
        root.after(100, poll)

    def export_controls():
        if engine is None or engine.controls_truncated:
            return
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if path:
            try:
                if Path(path).suffix.lower() != ".json" or Path(path).resolve() in (
                    ROOT / "track.json", ROOT / "rules.json",
                ):
                    raise ValueError("操作記録は設定とは別の JSON ファイルに保存してください。")
                Path(path).write_text(json.dumps({
                    "seed": engine.seed,
                    "changes": list(engine.control_log.copy()),
                    "transitions": [
                        {"elapsed_seconds": frame / track.rate,
                         "source_seconds": source / track.rate, "style": name}
                        for frame, source, name in engine.history.copy()
                    ],
                }, indent=2), encoding="utf-8")
            except (OSError, ValueError) as error:
                status.set(str(error))

    play_button = ttk.Button(body, text="再生", command=play)
    play_button.grid(row=0, column=0, sticky="ew", padx=(0, 8))
    stop_button = ttk.Button(body, text="停止", command=stop, state="disabled")
    stop_button.grid(row=0, column=1, sticky="ew")
    ttk.Label(body, text="音量").grid(row=1, column=0, sticky="w", pady=(12, 0))
    slider = ttk.Scale(body, from_=0, to=100, command=update_volume)
    slider.set(70)
    slider.grid(row=1, column=1, sticky="ew", pady=(12, 0))
    ttk.Label(body, text="乱数シード").grid(row=2, column=0, sticky="w", pady=(8, 0))
    seed_controls = ttk.Frame(body)
    seed_controls.grid(row=2, column=1, sticky="ew", pady=(8, 0))
    seed_controls.columnconfigure(0, weight=1)
    seed_entry = ttk.Entry(seed_controls, textvariable=seed, width=16)
    seed_entry.grid(row=0, column=0, sticky="ew")
    random_seed_button = ttk.Button(
        seed_controls, text="ランダム", command=lambda: seed.set(str(getrandbits(32)))
    )
    random_seed_button.grid(row=0, column=1, padx=(8, 0))
    ttk.Label(body, textvariable=status, wraplength=270).grid(
        row=3, column=0, columnspan=2, sticky="w", pady=(12, 0)
    )
    ttk.Label(body, textvariable=intensity_label).grid(row=4, column=0, sticky="w", pady=(8, 0))
    strength = ttk.Scale(body, from_=0, to=100, command=update_intensity)
    strength.set(initial_intensity * 100)
    update_intensity(initial_intensity * 100)
    strength.grid(row=4, column=1, sticky="ew")
    ttk.Label(body, textvariable=details, wraplength=510).grid(row=5, columnspan=2, sticky="w")
    ttk.Label(body, textvariable=recent, wraplength=510).grid(row=6, columnspan=2, sticky="w")
    export_button = ttk.Button(body, text="強度操作を保存", command=export_controls, state="disabled")
    export_button.grid(row=7, columnspan=2, sticky="w", pady=(8, 0))

    def quit_app():
        try:
            close_stream()
        finally:
            root.destroy()

    if sys.platform == "win32":
        from tray import WindowTray

        tray = WindowTray(root, quit_app)

    root.protocol("WM_DELETE_WINDOW", quit_app)
    root.after(100, poll)
    try:
        root.mainloop()
    finally:
        if tray is not None:
            tray.close()


def main():
    parser = argparse.ArgumentParser(description="Club Coven endless arrangement")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview", metavar="WAV", help="render a WAV instead of opening the UI")
    parser.add_argument("--seconds", type=int, default=115)
    parser.add_argument("--intensity", type=float, default=100, help="arrangement intensity, 0–100")
    parser.add_argument("--controls", type=Path, help="replay a saved intensity operation log")
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()
    if args.seconds <= 0 or not 0 <= args.intensity <= 100:
        parser.error("--seconds must be positive and --intensity must be in 0–100")
    try:
        rules = read_json("rules.json")
        config = read_json("track.json")
        validate(config, rules)
        if args.preview and (Path(args.preview).suffix.lower() != ".wav"
                             or Path(args.preview).resolve() == (ROOT / config["source"]).resolve()):
            parser.error("試聴先には原曲とは別の WAV ファイルを指定してください。")
        controls = {}
        if args.controls:
            log = json.loads(args.controls.read_text(encoding="utf-8-sig"))
            args.seed = log["seed"]
            for bar, intensity in log["changes"]:
                if type(bar) is not int or bar < 0 or not 0 <= intensity <= 1:
                    raise ValueError("強度操作の記録が不正です。")
                controls[bar] = intensity
            if type(args.seed) is not int or 0 not in controls:
                raise ValueError("強度操作の記録にはシードと最初の小節が必要です。")
        track = Track(config, rules, ROOT / ".cache" / "track.npz", args.rebuild_cache)
        if args.preview:
            preview(args.preview, args.seconds, Engine(track, rules, args.seed, args.intensity / 100, controls))
        else:
            run_gui(track, rules, args.seed, args.intensity / 100, controls)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        if sys.stderr is None and not args.preview:
            from tkinter import messagebox

            messagebox.showerror("Club Coven", str(error))
        else:
            print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
