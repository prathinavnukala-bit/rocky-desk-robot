# Rocky firmware

ESP32-S3 body firmware for Rocky, a 3D-printed quadruped companion robot: 8 servos on a PCA9685,
SSD1306 OLED face, I2S mic/speaker, on-device "Hey Rocky" wake word (TFLite Micro), WiFi control
link (port 8266), HTTP status/OTA/servo-control page (port 8080), and 22 poses/tricks.

## Before building
Edit three placeholders in `src/main.cpp` and `tools/ota_push.py`:
- `WIFI_SSID`, `WIFI_PASSWORD`
- `OTA_PASSWORD` (must match `TOKEN` in `tools/ota_push.py`)

## Build / flash
- USB: `pio run -e esp32-s3-devkitc-1 -t upload`
- Over WiFi (once running): `pio run -e ota -t upload`

Servo angles are hand-calibrated for this build (see the pose functions in `main.cpp`); recalibrate
with the phone control page at `http://<board-ip>:8080/control`.
`enclosure/` holds the OLED face-panel CAD.
