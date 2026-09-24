"""Client for stt_server.py (mlx-whisper). Stdlib only, so brain.py can use it
without the heavy .stt-venv packages. Any failure returns None and the caller
falls back to the in-process model."""
import json
import os
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
VENV_PY = os.path.join(HERE, ".stt-venv", "bin", "python")
HOST, PORT = "127.0.0.1", 8765

# Kept in sync with stt_server.py's VOCAB_HINT -- used by brain.py's fallback
# (faster-whisper) path when the mlx server isn't available.
VOCAB_HINT = ("Rocky, servo, servos, PCA9685, ESP32, quadruped, wake word, "
              "enclosure, firmware, OTA, watchdog, mDNS, Wi-Fi.")
LOG = "/tmp/rocky_stt.log"
_proc = None


def _ask(payload, timeout):
    with socket.create_connection((HOST, PORT), timeout=2) as s:
        s.settimeout(timeout)
        s.sendall((json.dumps(payload) + "\n").encode())
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
    return json.loads(buf)


def is_up():
    try:
        return _ask({"ping": 1}, 3).get("ok", False)
    except Exception:
        return False


def start(wait_s=90):
    """Starts the server if needed and waits until it answers. Returns True if ready."""
    global _proc
    if is_up():
        return True
    if not os.path.exists(VENV_PY):
        return False
    if _proc is None or _proc.poll() is not None:
        _proc = subprocess.Popen([VENV_PY, "-u", os.path.join(HERE, "stt_server.py")],
                                 stdout=open(LOG, "a"), stderr=subprocess.STDOUT, cwd=HERE)
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if _proc.poll() is not None:
            return False
        if is_up():
            return True
        time.sleep(0.5)
    return False


def transcribe(wav_path, timeout=20):
    """-> (text, avg_logprob, no_speech_prob), or None if the server can't do it."""
    try:
        r = _ask({"path": os.path.abspath(wav_path)}, timeout)
        if "error" in r:
            print(f"STT server error: {r['error']}")
            return None
        return r["text"], r["avg_logprob"], r["no_speech_prob"]
    except Exception:
        # Server died? Restart it in the background for next time.
        if _proc is not None and _proc.poll() is not None:
            print("STT server exited; restarting it for the next turn.")
            try:
                start(wait_s=0)
            except Exception:
                pass
        return None


def stop():
    global _proc
    if _proc is not None and _proc.poll() is None:
        _proc.terminate()
    _proc = None
