"""Short branch themes, synthesised from the original music bed.

``music_bed`` (in the episode renderer) is a sine pad plus a soft bell over
an A-minor-ish progression, with no samples. ``config/branch_themes.json``
records one small change of key, bar length, waveform or bell for each
branch. ``scripts/render_branch_stingers.py`` writes the files the video
page plays. A future episode render should read the same JSON and pass
those knobs into ``music_bed``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

from firstbook.library import VIDEOS_CATALOG_PATH

THEMES_PATH = VIDEOS_CATALOG_PATH.parent / "branch_themes.json"
STINGER_DIR = VIDEOS_CATALOG_PATH.parent.parent / "static" / "audio" / "branches"


def load_themes(path: Path | None = None) -> dict:
    data = json.loads(Path(path or THEMES_PATH).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("branches"), dict):
        raise ValueError("branch_themes.json needs a 'branches' object.")
    return data


def _osc(kind: str, phase: np.ndarray) -> np.ndarray:
    if kind == "triangle":
        return (2.0 / np.pi) * np.arcsin(np.sin(phase))
    if kind == "soft-saw":
        # A few harmonics of a saw, still pure synthesis. Quieter than a raw saw.
        sig = np.zeros_like(phase, dtype=np.float64)
        for k in range(1, 7):
            sig += np.sin(k * phase) / k
        return sig * (2.0 / np.pi)
    return np.sin(phase)


def synthesize(themes: dict, branch_id: str, *, duration_s: float | None = None) -> np.ndarray:
    """One mono theme. ``duration_s`` overrides the JSON length (tests use a short one)."""
    base = themes.get("base") or {}
    theme = themes["branches"][branch_id]
    sr = int(themes.get("sample_rate") or 22050)
    duration = float(duration_s if duration_s is not None else theme.get("duration_s") or 8.0)
    n = max(1, int(duration * sr))
    tt = np.arange(n) / sr
    rng = np.random.default_rng(int(theme.get("seed", base.get("seed", 1))))
    a4 = float(base.get("a4", 440.0))
    transpose = int(theme.get("transpose", 0))
    third_lift = int(theme.get("third_lift", 0))

    def hz(semi: float) -> float:
        return a4 * 2 ** ((semi + transpose) / 12)

    chords = [list(chord) for chord in base["chords"]]
    if third_lift:
        for chord in chords:
            if len(chord) > 2:
                chord[2] = chord[2] + third_lift
    bar = float(theme.get("bar_s", base.get("bar_s", 4.0)))
    detune = float(theme.get("detune_hz", base.get("detune_hz", 0.12)))
    root_gain = float(theme.get("pad_root_gain", base.get("pad_root_gain", 0.35)))
    voice_gain = float(theme.get("pad_voice_gain", base.get("pad_voice_gain", 0.22)))
    harmonic = float(theme.get("harmonic_gain", base.get("harmonic_gain", 0.05)))
    bell_gain = float(theme.get("bell_gain", base.get("bell_gain", 0.16)))
    bell_decay = float(theme.get("bell_decay", base.get("bell_decay", 3.2)))
    kind = str(theme.get("waveform", base.get("waveform", "sine")))
    degrees = list(base.get("bell_degrees") or [2, 3, 4, 3])

    pad = np.zeros(n, np.float64)
    for ci in range(int(duration / bar) + 2):
        notes = chords[ci % len(chords)]
        t0 = ci * bar - 0.8
        start, end = max(0, int(t0 * sr)), min(n, int((t0 + bar + 1.6) * sr))
        if start >= end:
            continue
        lt = tt[start:end] - t0
        env = np.sin(np.pi * np.clip(lt / (bar + 1.6), 0, 1)) ** 1.5
        sig = np.zeros(end - start, np.float64)
        for k, semi in enumerate(notes):
            freq = hz(semi)
            gain = root_gain if k == 0 else voice_gain
            for det in (-detune, detune):
                phase = 2 * np.pi * (freq + det) * lt + rng.uniform(0, 6)
                sig += _osc(kind, phase) * gain
            sig += harmonic * np.sin(2 * np.pi * 2 * freq * lt)
        pad[start:end] += sig * env

    bell = np.zeros(n, np.float64)
    step = bar / 8
    jumps = max(1, int(duration / step))
    for j in range(jumps):
        notes = chords[int(j * step / bar) % len(chords)]
        if j % 2 and rng.random() < 0.5:
            continue
        degree = degrees[j % len(degrees)]
        if degree >= len(notes):
            degree = len(notes) - 1
        freq = hz(notes[degree] + 12)
        start = int(j * step * sr)
        end = min(n, start + int(1.6 * sr))
        if start >= end:
            continue
        lt = tt[start:end] - j * step
        # The bell stays a sine, so every branch still has the same soft attack.
        bell[start:end] += np.sin(2 * np.pi * freq * lt) * np.exp(-lt * bell_decay) * bell_gain

    peak = float(np.max(np.abs(pad))) + 1e-9
    mixed = pad / peak * 0.6 + bell
    fade = np.clip(tt / 0.35, 0, 1) * np.clip((duration - tt) / 1.1, 0, 1)
    level = float(theme.get("level", 0.9))
    return (mixed * fade * level).astype(np.float32)


def _write_wav(path: Path, data: np.ndarray, sr: int) -> None:
    pcm = (np.clip(data, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(pcm.tobytes())


def _encode(wav: Path, dest: Path, codec_args: list[str]) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is not on PATH, so the stinger cannot be encoded.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(wav), *codec_args, str(dest)],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0 or not dest.is_file():
        tail = (proc.stderr or proc.stdout or "ffmpeg failed").strip().splitlines()[-1:]
        raise RuntimeError(tail[0] if tail else "ffmpeg failed")


def render_stingers(dest: Path | None = None, *, branch_id: str | None = None, themes: dict | None = None) -> list[Path]:
    """Write one ogg and one mp3 per branch. Returns the paths."""
    themes = themes or load_themes()
    dest = Path(dest or STINGER_DIR)
    dest.mkdir(parents=True, exist_ok=True)
    ids = [branch_id] if branch_id else list(themes["branches"])
    written = []
    sr = int(themes.get("sample_rate") or 22050)
    for ident in ids:
        audio = synthesize(themes, ident)
        wav = dest / f"{ident}.wav"
        ogg = dest / f"{ident}.ogg"
        mp3 = dest / f"{ident}.mp3"
        _write_wav(wav, audio, sr)
        try:
            _encode(wav, ogg, ["-c:a", "libvorbis", "-b:a", "64k", "-ac", "1"])
            _encode(wav, mp3, ["-c:a", "libmp3lame", "-b:a", "64k", "-ac", "1"])
        finally:
            wav.unlink(missing_ok=True)
        written.extend([ogg, mp3])
    return written
