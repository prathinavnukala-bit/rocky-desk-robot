"""
Heuristic vocal-tone reading -- NOT a trained speech-emotion classifier.
There isn't a small, well-established offline SER model the way FER+ exists
for faces, so this estimates a rough "vibe" from prosody (loudness, pitch
variation, speaking rate) using simple signal processing. Treat the output
as a rough signal to react to, not a confident diagnosis.
"""
import wave

import numpy as np


def _read_wav(path):
    with wave.open(path, "rb") as wf:
        raw = wf.readframes(wf.getnframes())
        rate = wf.getframerate()
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    return samples, rate


def _rms_energy(samples):
    if len(samples) == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples ** 2)))


def _pitch_stats(samples, rate, frame_ms=40, silence_thresh=200):
    """Rough per-frame pitch via autocorrelation, restricted to a human voice range."""
    frame_len = int(rate * frame_ms / 1000)
    if frame_len <= 0:
        return 0.0, 0.0

    min_lag = int(rate / 400)  # 400 Hz upper bound
    max_lag = int(rate / 80)   # 80 Hz lower bound

    pitches = []
    for start in range(0, len(samples) - frame_len, frame_len):
        frame = samples[start:start + frame_len]
        if np.max(np.abs(frame)) < silence_thresh:
            continue

        frame = frame - np.mean(frame)
        corr = np.correlate(frame, frame, mode="full")
        corr = corr[len(corr) // 2:]

        if max_lag >= len(corr):
            continue
        segment = corr[min_lag:max_lag]
        if len(segment) == 0:
            continue

        peak = int(np.argmax(segment)) + min_lag
        if corr[peak] <= 0:
            continue
        pitches.append(rate / peak)

    if len(pitches) < 2:
        return 0.0, (pitches[0] if pitches else 0.0)
    return float(np.std(pitches)), float(np.mean(pitches))


def analyze_mood(wav_path, transcript_duration_s=None, word_count=None):
    samples, rate = _read_wav(wav_path)
    energy = _rms_energy(samples)
    pitch_std, pitch_mean = _pitch_stats(samples, rate)

    speaking_rate = None
    if transcript_duration_s and word_count and transcript_duration_s > 0:
        speaking_rate = word_count / transcript_duration_s

    # Raw loudness (RMS) is heavily dependent on mic gain and distance from the
    # mic -- not calibrated against this hardware, so it's reported but not used
    # for classification. Pitch variability relative to mean pitch (coefficient
    # of variation) is unit-normalized and a more hardware-independent signal.
    pitch_cv = (pitch_std / pitch_mean) if pitch_mean > 0 else 0.0

    if pitch_mean == 0:
        vibe = "unclear (too quiet or too short to read)"
    elif pitch_cv > 0.30 or (speaking_rate and speaking_rate > 3.0):
        vibe = "animated / expressive"
    elif pitch_cv < 0.12:
        vibe = "flat / monotone"
    else:
        vibe = "calm / neutral"

    return {
        "vibe": vibe,
        "pitch_mean_hz": round(pitch_mean, 1),
        "pitch_variability": round(pitch_cv, 2),
        "speaking_rate_wps": round(speaking_rate, 2) if speaking_rate else None,
        "loudness_raw": round(energy, 1),  # hardware-dependent, informational only
    }
