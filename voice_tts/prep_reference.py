"""
prep_reference.py - one-shot audio prep for XTTS-v2 reference clips.

Pipeline:
  raw .wav (any format) -> downmix mono -> resample 22050 -> HPF 80Hz ->
  spectral denoise -> trim silence -> loudness normalize -> best 15s window
  -> mike.wav

Usage:
  python prep_reference.py /voices/raw/mike_raw.wav /voices/mike.wav
"""
import sys
import os
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import noisereduce as nr
import pyloudnorm as pyln
from scipy.signal import butter, sosfilt

TARGET_SR = 22050
TARGET_WINDOW_S = 15.0
HPF_HZ = 80
SILENCE_DB = -40.0
MAX_GAP_S = 0.5
COLLAPSED_GAP_S = 0.2
PEAK_DBFS = -3.0
LUFS_TARGET = -18.0
DENOISE_PROP = 0.85


def _dbfs(x: np.ndarray) -> float:
    v = float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0
    return 20.0 * np.log10(v + 1e-9)


def _peak_dbfs(x: np.ndarray) -> float:
    p = float(np.max(np.abs(x))) if x.size else 0.0
    return 20.0 * np.log10(p + 1e-9)


def _snr_estimate(x: np.ndarray, sr: int) -> float:
    """Quick stationary-noise SNR estimate from 20ms frames."""
    n = int(sr * 0.02)
    if n <= 0 or x.size < n:
        return 0.0
    frames = np.array([np.sqrt(np.mean(x[i:i + n] ** 2)) for i in range(0, len(x) - n, n)])
    if frames.size == 0:
        return 0.0
    nf = float(np.percentile(frames, 10))
    sf_ = float(np.percentile(frames, 90))
    return 20.0 * np.log10((sf_ + 1e-9) / (nf + 1e-9))


def metrics(x: np.ndarray, sr: int, label: str) -> dict:
    return {
        "label": label,
        "sr": int(sr),
        "duration_s": round(len(x) / sr, 3),
        "channels": 1 if x.ndim == 1 else x.shape[1],
        "rms_dbfs": round(_dbfs(x.flatten() if x.ndim > 1 else x), 2),
        "peak_dbfs": round(_peak_dbfs(x.flatten() if x.ndim > 1 else x), 2),
        "snr_db": round(_snr_estimate(x.flatten() if x.ndim > 1 else x, sr), 2),
    }


def downmix_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x.astype(np.float32, copy=False)
    return x.mean(axis=1).astype(np.float32, copy=False)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    return librosa.resample(x.astype(np.float32, copy=False), orig_sr=sr_in, target_sr=sr_out)


def highpass(x: np.ndarray, sr: int, cutoff_hz: int = HPF_HZ) -> np.ndarray:
    sos = butter(4, cutoff_hz / (sr / 2.0), btype="highpass", output="sos")
    return sosfilt(sos, x).astype(np.float32, copy=False)


def denoise(x: np.ndarray, sr: int) -> np.ndarray:
    return nr.reduce_noise(y=x, sr=sr, stationary=True, prop_decrease=DENOISE_PROP).astype(np.float32, copy=False)


def trim_silence(x: np.ndarray, sr: int, top_db: float = -SILENCE_DB) -> np.ndarray:
    trimmed, _ = librosa.effects.trim(x, top_db=top_db)
    return trimmed


def collapse_internal_gaps(x: np.ndarray, sr: int, max_gap_s: float = MAX_GAP_S, collapsed_s: float = COLLAPSED_GAP_S) -> np.ndarray:
    """Collapse internal silence runs longer than max_gap_s down to collapsed_s."""
    intervals = librosa.effects.split(x, top_db=-SILENCE_DB)
    if len(intervals) <= 1:
        return x
    out_parts: list[np.ndarray] = []
    gap_samples = int(collapsed_s * sr)
    max_gap_samples = int(max_gap_s * sr)
    prev_end = None
    for start, end in intervals:
        if prev_end is not None:
            gap = start - prev_end
            if gap > max_gap_samples:
                out_parts.append(np.zeros(gap_samples, dtype=np.float32))
            else:
                out_parts.append(x[prev_end:start])
        out_parts.append(x[start:end])
        prev_end = end
    return np.concatenate(out_parts).astype(np.float32, copy=False)


def best_window(x: np.ndarray, sr: int, window_s: float = TARGET_WINDOW_S) -> np.ndarray:
    """Pick the contiguous window_s seconds with the highest sustained RMS."""
    n_win = int(window_s * sr)
    if len(x) <= n_win:
        return x
    frame = int(sr * 0.05)
    hop = frame
    energy = np.array([np.sqrt(np.mean(x[i:i + frame] ** 2)) for i in range(0, len(x) - frame, hop)])
    frames_per_window = n_win // hop
    if frames_per_window <= 0 or energy.size < frames_per_window:
        return x[:n_win]
    cumsum = np.cumsum(np.insert(energy, 0, 0.0))
    window_energies = cumsum[frames_per_window:] - cumsum[:-frames_per_window]
    best_idx = int(np.argmax(window_energies))
    start_sample = best_idx * hop
    return x[start_sample:start_sample + n_win]


def normalize_loudness(x: np.ndarray, sr: int, lufs_target: float = LUFS_TARGET, peak_dbfs: float = PEAK_DBFS) -> np.ndarray:
    meter = pyln.Meter(sr)
    try:
        lufs = meter.integrated_loudness(x)
    except Exception:
        lufs = lufs_target
    if np.isfinite(lufs):
        x = pyln.normalize.loudness(x, lufs, lufs_target)
    peak = float(np.max(np.abs(x))) + 1e-9
    target_peak = 10 ** (peak_dbfs / 20.0)
    if peak > target_peak:
        x = x * (target_peak / peak)
    return x.astype(np.float32, copy=False)


def prep(in_path: str, out_path: str, debug_dir: str | None = None) -> dict:
    raw, sr_in = sf.read(in_path, always_2d=False)
    raw = raw.astype(np.float32, copy=False)
    before = metrics(raw, sr_in, "raw")

    debug = Path(debug_dir) if debug_dir else None
    if debug:
        debug.mkdir(parents=True, exist_ok=True)

    x = downmix_mono(raw)
    if debug: sf.write(debug / "01_mono.wav", x, sr_in)

    x = resample(x, sr_in, TARGET_SR)
    sr = TARGET_SR
    if debug: sf.write(debug / "02_resampled.wav", x, sr)

    x = highpass(x, sr)
    if debug: sf.write(debug / "03_hpf.wav", x, sr)

    x = denoise(x, sr)
    if debug: sf.write(debug / "04_denoised.wav", x, sr)

    x = trim_silence(x, sr)
    if debug: sf.write(debug / "05_trimmed.wav", x, sr)

    x = collapse_internal_gaps(x, sr)
    if debug: sf.write(debug / "06_collapsed.wav", x, sr)

    x = best_window(x, sr)
    if debug: sf.write(debug / "07_window.wav", x, sr)

    x = normalize_loudness(x, sr)
    if debug: sf.write(debug / "08_normalized.wav", x, sr)

    after = metrics(x, sr, "prepared")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, x, sr, subtype="PCM_16")

    report = {
        "input": in_path,
        "output": out_path,
        "before": before,
        "after": after,
        "params": {
            "target_sr": TARGET_SR,
            "target_window_s": TARGET_WINDOW_S,
            "hpf_hz": HPF_HZ,
            "silence_db": SILENCE_DB,
            "denoise_prop": DENOISE_PROP,
            "peak_dbfs": PEAK_DBFS,
            "lufs_target": LUFS_TARGET,
        },
    }
    return report


def validate(path: str) -> tuple[bool, str]:
    if not Path(path).exists():
        return False, f"missing: {path}"
    try:
        info = sf.info(path)
    except Exception as exc:
        return False, f"unreadable: {exc}"
    if info.duration < 10:
        return False, f"too short: {info.duration:.1f}s (need >= 10s)"
    if info.duration > 60:
        return False, f"too long: {info.duration:.1f}s (need <= 60s for prep)"
    if info.samplerate < 16000:
        return False, f"sample rate too low: {info.samplerate} (need >= 16000)"
    return True, "ok"


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: prep_reference.py <input.wav> <output.wav> [debug_dir]", file=sys.stderr)
        sys.exit(2)
    in_p, out_p = sys.argv[1], sys.argv[2]
    dbg = sys.argv[3] if len(sys.argv) > 3 else None

    ok, msg = validate(in_p)
    if not ok:
        print(json.dumps({"status": "INVALID_INPUT", "reason": msg}), file=sys.stderr)
        sys.exit(1)

    t0 = time.time()
    report = prep(in_p, out_p, dbg)
    report["elapsed_s"] = round(time.time() - t0, 2)
    report["status"] = "OK"
    print(json.dumps(report, indent=2))
