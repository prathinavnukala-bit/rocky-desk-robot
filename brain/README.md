# Rocky brain

Mac-side "mind" for Rocky: speech-to-text (mlx-whisper), local LLM chat (Ollama), vision and face
recognition, long-term memory (SQLite), text-to-speech (Piper), and the persistent WiFi link to the
robot body (`esp_link.py`, `body.py`, `listen.py`). Run with `python3 brain.py`; `diagnose.py --deep`
checks every subsystem.

## Not included (download / generate yourself)
- `models/` - ONNX models: YuNet face detector, SFace recognizer, YOLOX-s, FER+ emotion
- `voices/en_GB-alan-medium.onnx` - Piper voice (the `.onnx.json` config is included)
- Personal data (`rocky_memory.db`, `memory.json`, `faces.json`) is created at runtime and gitignored.
