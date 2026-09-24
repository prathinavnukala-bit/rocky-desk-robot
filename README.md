# Rocky - desk robot

A 3D-printed quadruped companion robot that hears "Hey Rocky", talks, sees, remembers, and moves.
It is split into a small body and a Mac-side brain that talk over WiFi.

| Folder | What it is |
|---|---|
| `firmware/` | ESP32-S3 body: 8 servos (PCA9685), OLED face, mic/speaker, on-device "Hey Rocky" wake word, 22 poses/tricks, WiFi control link, OTA updates, enclosure CAD |
| `brain/` | Mac-side Python: speech-to-text, local LLM chat, vision and face recognition, long-term memory, text-to-speech, robot link |
| `servo-bench/` | Standalone servo-calibration web page, reusable for any PCA9685/ESP32 project |

## Getting started
1. **Firmware** - set `WIFI_SSID`, `WIFI_PASSWORD`, `OTA_PASSWORD` in `firmware/src/main.cpp` (and the matching
   `TOKEN` in `firmware/tools/ota_push.py`), then `cd firmware && pio run -e esp32-s3-devkitc-1 -t upload`.
2. **Brain** - see `brain/README.md` for the model files to download, then `cd brain && python3 brain.py`.
3. **Calibrating servos** - open `http://<board-ip>:8080/control`, or use `servo-bench/index.html`.

Personal data (memory database, face data, recordings) is never committed - see `brain/.gitignore`.
