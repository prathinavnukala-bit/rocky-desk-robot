# Servo Bench

Single-file live servo-calibration panel for any PCA9685/ESP32 project. It only needs one endpoint on
the device: `GET /servo?ch=<n>&angle=<deg>` returning HTTP 200. Open `index.html` locally and enter the
board's IP, or use the page's "Copy as PROGMEM snippet" button to serve it from the device itself.
