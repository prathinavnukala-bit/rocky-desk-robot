# Rocky camera

Firmware for Rocky's camera: an AI-Thinker ESP32-CAM with an OV3660 sensor on its own board. It serves JPEG pictures to the brain over Wi-Fi and is found as `rocky-cam.local`.

Reliability: a timer-driven watchdog reboots the board if the main loop stops making progress (the camera driver can hang after long uptime or on marginal power), and a periodic self-capture catches a wedged sensor before anyone asks for a picture.

## Before building
Edit the placeholders:
- `WIFI_SSID`, `WIFI_PASSWORD`, `OTA_TOKEN` in `src/main.cpp`
- `TOKEN` in `tools/ota_push.py` (must match `OTA_TOKEN`)

## Build / flash
- USB (first time): `pio run -e esp32cam -t upload`
- Over Wi-Fi afterwards: `pio run -e ota -t upload`
