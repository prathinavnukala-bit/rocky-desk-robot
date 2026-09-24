"""Speech-to-text server. Runs inside .stt-venv (mlx-whisper on the Mac's GPU):
same Whisper "small" weights as before, ~0.6 s instead of ~1.4 s per utterance.
brain.py talks to it through stt.py and falls back to the in-process
faster-whisper model if this server is not running.

Protocol: one JSON line in {"path": wav} or {"ping": 1}; one JSON line out."""
import json
import socketserver
import sys
import wave

import numpy as np
import mlx_whisper

MODEL = "mlx-community/whisper-small-mlx"
HOST, PORT = "127.0.0.1", 8765

# Whisper's initial_prompt biases vocabulary/spelling toward these words when
# it's unsure, without forcing them -- confirmed live that "servo(s)" was
# being misheard as "server(s)" repeatedly (a Prathinav's real robot project
# got mis-transcribed and its memory forked in two before this was caught,
# see memory_store.py's upsert_project docstring). Whisper's own vocabulary
# just doesn't lean toward "servo" without a hint.
VOCAB_HINT = ("Rocky, servo, servos, PCA9685, ESP32, quadruped, wake word, "
              "enclosure, firmware, OTA, watchdog, mDNS, Wi-Fi.")


def load_wav(path):
    with wave.open(path) as w:
        raw = w.readframes(w.getnframes())
        rate, width, ch = w.getframerate(), w.getsampwidth(), w.getnchannels()
    if width != 2 or rate != 16000:
        raise ValueError(f"expected 16 kHz 16-bit audio, got {rate} Hz / {width * 8}-bit")
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return audio.reshape(-1, ch).mean(axis=1) if ch > 1 else audio


def transcribe(path):
    audio = load_wav(path)
    r = mlx_whisper.transcribe(
        audio, path_or_hf_repo=MODEL, language="en", condition_on_previous_text=False,
        initial_prompt=VOCAB_HINT,
        fp16=True, no_speech_threshold=0.6, logprob_threshold=-1.0, compression_ratio_threshold=2.4)
    segs = r.get("segments", [])
    # Same defence the VAD filter gave the old engine: drop segments Whisper itself flags as non-speech.
    segs = [s for s in segs if not (s["no_speech_prob"] > 0.6 and s["avg_logprob"] < -1.0)]
    if not segs:
        return {"text": "", "avg_logprob": -5.0, "no_speech_prob": 1.0}
    return {"text": " ".join(s["text"].strip() for s in segs).strip(),
            "avg_logprob": float(np.mean([s["avg_logprob"] for s in segs])),
            "no_speech_prob": float(max(s["no_speech_prob"] for s in segs))}


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            req = json.loads(self.rfile.readline())
            out = {"ok": True} if req.get("ping") else transcribe(req["path"])
        except Exception as e:  # report, never die
            out = {"error": str(e)}
        self.wfile.write((json.dumps(out) + "\n").encode())


class Server(socketserver.TCPServer):
    allow_reuse_address = True


if __name__ == "__main__":
    import os
    import tempfile
    warm = os.path.join(tempfile.gettempdir(), "rocky_stt_warm.wav")
    with wave.open(warm, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(b"\x00\x00" * 16000)
    transcribe(warm)
    print("STT READY", flush=True)
    with Server((HOST, PORT), Handler) as srv:
        srv.serve_forever()
