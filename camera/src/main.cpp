#include <Arduino.h>
#include "esp_camera.h"
#include <WiFi.h>
#include <ESPmDNS.h>
#include <WebServer.h>
#include <Update.h>
#include <esp_timer.h>
#include <esp_system.h>

// =====================================================
// ROCKY CAMERA (AI-Thinker ESP32-CAM + OV3660)
//
// Reliability design, each piece answering a failure actually seen:
//  * esp_camera_fb_get() can hang FOREVER if the sensor loses sync (seen after
//    long uptime, and again on marginal power). A timer-driven watchdog that
//    does not depend on the stock task watchdog reboots the board if the loop
//    stops making progress, so it recovers by itself instead of needing a
//    power-cycle by hand.
//  * A periodic self-check captures a frame on its own, so a wedged sensor is
//    caught (and the board rebooted) before anyone asks for a picture.
//  * The board is found as rocky-cam.local (mDNS), whatever DHCP address the
//    router hands out (it has moved five times).
//  * Firmware can be pushed over Wi-Fi (HTTP, port 8080) -- no USB needed once
//    the enclosure is closed.
//  * WiFi TX power is reduced from 19.5 to 15 dBm to lower the current spikes
//    that browned this board out on a USB hub.
// =====================================================

const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
#define HOSTNAME "rocky-cam"
#define OTA_TOKEN "change-me"   // LAN-only; keep in sync with tools/ota_push.py

// AI-THINKER ESP32-CAM pins
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

WiFiServer server(80);
WebServer otaServer(8080);

// ---------------------------------------------------------------- watchdog
#define WATCHDOG_MS 20000        // a normal capture+send is well under 3s
volatile unsigned long lastProgressMs = 0;
void feedWatchdog() { lastProgressMs = millis(); }

static void watchdogCheck(void *) {
  if (lastProgressMs != 0 && millis() - lastProgressMs > WATCHDOG_MS) {
    Serial.println("Watchdog: loop stalled (camera hang?) -- rebooting.");
    esp_restart();
  }
}

void startWatchdog() {
  feedWatchdog();
  esp_timer_create_args_t args = {};
  args.callback = &watchdogCheck;
  args.name = "camwdt";
  esp_timer_handle_t timer;
  esp_timer_create(&args, &timer);
  esp_timer_start_periodic(timer, 1000000);
}

// ---------------------------------------------------------------- health
unsigned long lastSelfCheckMs = 0;
int captureFailures = 0;
unsigned long lastCaptureMs = 0;
unsigned long capturesServed = 0;

// Grabs and returns one frame to prove the sensor still delivers. If the
// sensor is wedged this call never returns, and the watchdog reboots us.
bool cameraSelfCheck() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb || fb->len < 1000) {
    if (fb) esp_camera_fb_return(fb);
    captureFailures++;
    Serial.printf("Camera self-check FAILED (%d in a row)\n", captureFailures);
    if (captureFailures >= 3) {
      Serial.println("Camera keeps failing -- rebooting to re-initialise it.");
      delay(200);
      esp_restart();
    }
    return false;
  }
  esp_camera_fb_return(fb);
  captureFailures = 0;
  return true;
}

// ---------------------------------------------------------------- HTTP helpers
bool sendBuffer(WiFiClient &client, const uint8_t *data, size_t length) {
  size_t sent = 0;
  unsigned long stallStart = millis();
  while (sent < length) {
    if (!client.connected()) return false;
    size_t chunk = min((size_t)1024, length - sent);
    size_t written = client.write(data + sent, chunk);
    if (written > 0) {
      sent += written;
      stallStart = millis();
      feedWatchdog();
    } else {
      if (millis() - stallStart > 5000) return false;   // client stopped reading
      delay(2);
    }
  }
  return true;
}

void sendText(WiFiClient &client, int code, const char *status, const char *type, const String &body) {
  client.printf("HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %u\r\nConnection: close\r\nCache-Control: no-cache\r\n\r\n",
                code, status, type, (unsigned) body.length());
  client.print(body);
}

void sendRoot(WiFiClient &client) {
  sendText(client, 200, "OK", "text/html",
           "<!DOCTYPE html><html><head><meta charset='UTF-8'><title>Rocky Camera</title></head>"
           "<body><h1>ROCKY CAMERA</h1><p>OV3660 online.</p>"
           "<p><a href='/jpg'>Take JPEG</a> &middot; <a href='/status'>Status</a></p></body></html>");
}

void sendStatus(WiFiClient &client) {
  String s = "uptime_s=" + String(millis() / 1000) +
             " heap=" + String(ESP.getFreeHeap()) +
             " psram=" + String(ESP.getFreePsram()) +
             " rssi=" + String(WiFi.RSSI()) +
             " ip=" + WiFi.localIP().toString() +
             " captures=" + String(capturesServed) +
             " last_capture_ms=" + String(lastCaptureMs) +
             " failures=" + String(captureFailures) +
             " reset=" + String((int) esp_reset_reason()) +
             " fw=" + String(__DATE__) + " " + String(__TIME__) + "\n";
  sendText(client, 200, "OK", "text/plain", s);
}

void sendJPEG(WiFiClient &client) {
  unsigned long t0 = millis();
  camera_fb_t *fb = esp_camera_fb_get();   // may hang forever on a wedged sensor: the watchdog covers it
  if (!fb || fb->len < 1000) {
    if (fb) esp_camera_fb_return(fb);
    captureFailures++;
    sendText(client, 500, "Internal Server Error", "text/plain", "Camera capture failed");
    if (captureFailures >= 3) {
      delay(200);
      esp_restart();
    }
    return;
  }
  captureFailures = 0;
  client.printf("HTTP/1.1 200 OK\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\nConnection: close\r\nCache-Control: no-cache\r\n\r\n",
                (unsigned) fb->len);
  sendBuffer(client, fb->buf, fb->len);
  esp_camera_fb_return(fb);
  capturesServed++;
  lastCaptureMs = millis() - t0;
}

void handleClient(WiFiClient &client) {
  client.setTimeout(3000);
  String requestLine = client.readStringUntil('\n');
  requestLine.trim();
  while (client.connected()) {
    String line = client.readStringUntil('\n');
    line.trim();
    if (line.length() == 0) break;
  }
  if (requestLine.startsWith("GET /jpg")) sendJPEG(client);
  else if (requestLine.startsWith("GET /status")) sendStatus(client);
  else if (requestLine.startsWith("GET / ")) sendRoot(client);
  else sendText(client, 404, "Not Found", "text/plain", "404 - Not Found");
  client.flush();
  delay(10);
  client.stop();
}

// ---------------------------------------------------------------- HTTP OTA
static bool otaUploadAllowed = false;

void setupHttpOta() {
  static const char *headerKeys[] = {"X-Rocky-Token"};
  otaServer.collectHeaders(headerKeys, 1);
  otaServer.on("/version", HTTP_GET, []() {
    otaServer.send(200, "text/plain", String(__DATE__) + " " + __TIME__);
  });
  otaServer.on("/update", HTTP_POST,
    []() {
      bool ok = otaUploadAllowed && !Update.hasError();
      otaServer.send(ok ? 200 : 500, "text/plain", ok ? "OK - rebooting into the new firmware" : "FAILED");
      delay(400);
      if (ok) ESP.restart();
    },
    []() {
      HTTPUpload &up = otaServer.upload();
      if (up.status == UPLOAD_FILE_START) {
        otaUploadAllowed = otaServer.header("X-Rocky-Token") == OTA_TOKEN;
        if (!otaUploadAllowed) return;
        if (!Update.begin(UPDATE_SIZE_UNKNOWN)) Update.printError(Serial);
      } else if (up.status == UPLOAD_FILE_WRITE && otaUploadAllowed) {
        if (Update.write(up.buf, up.currentSize) != up.currentSize) Update.printError(Serial);
        feedWatchdog();
      } else if (up.status == UPLOAD_FILE_END && otaUploadAllowed) {
        if (!Update.end(true)) Update.printError(Serial);
      }
    });
  otaServer.begin();
}

// ---------------------------------------------------------------- setup
void connectWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.persistent(false);
  WiFi.setAutoReconnect(true);
  WiFi.setHostname(HOSTNAME);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  WiFi.setSleep(false);
  WiFi.setTxPower(WIFI_POWER_15dBm);   // Tried 19.5 (max) to fight weak-signal packet loss --
                                        // reverted. RSSI (what this board RECEIVES from the
                                        // router) barely moved either way, since that's set by
                                        // the ROUTER's transmit power and distance, not this
                                        // board's own TX level -- so the higher setting bought
                                        // no real range, only extra current draw, right when
                                        // the camera sensor also needs a current spike to
                                        // capture a frame. That combination produced repeated
                                        // failed captures (3 in a row -> self-reboot, see
                                        // cameraSelfCheck/sendJPEG) even on a "stable" 5V
                                        // supply. Back to the lower, safer draw.

  unsigned long start = millis();
  Serial.print("Connecting to Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) {
    if (millis() - start > 30000) {
      Serial.println("\nWi-Fi did not connect in 30s -- rebooting to retry.");
      delay(200);
      esp_restart();
    }
    delay(500);
    Serial.print(".");
  }
  Serial.println();
}

void setup() {
  Serial.begin(115200);
  delay(1500);
  Serial.println();
  Serial.println("==============================");
  Serial.println("        ROCKY CAMERA");
  Serial.println("==============================");

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM; config.pin_d1 = Y3_GPIO_NUM; config.pin_d2 = Y4_GPIO_NUM; config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM; config.pin_d5 = Y7_GPIO_NUM; config.pin_d6 = Y8_GPIO_NUM; config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM; config.pin_pclk = PCLK_GPIO_NUM; config.pin_vsync = VSYNC_GPIO_NUM; config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM; config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM; config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // Back to VGA: right after the wall-charger power fix, VGA was capturing
  // successfully (11/11, 0 failures) -- the only problem left was slow
  // downloads over weak WiFi. Switching to QVGA was MY change, made right
  // before capture-level failures and sensor hangs started appearing, and
  // never re-tested against the config that was actually working. Reverting
  // it rather than assuming the sensor itself is at fault.
  if (psramFound()) {
    config.frame_size = FRAMESIZE_VGA;
    config.jpeg_quality = 10;
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }
  config.grab_mode = CAMERA_GRAB_LATEST;
  config.fb_location = psramFound() ? CAMERA_FB_IN_PSRAM : CAMERA_FB_IN_DRAM;

  Serial.println("Initializing camera...");
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    // Used to spin forever here, needing a manual power-cycle. Retry via reboot.
    Serial.printf("Camera initialization FAILED: 0x%x -- rebooting to retry.\n", err);
    delay(1500);
    esp_restart();
  }
  Serial.println("Camera initialized!");

  sensor_t *sensor = esp_camera_sensor_get();
  if (sensor->id.PID == OV3660_PID) {
    sensor->set_vflip(sensor, 1);
    sensor->set_brightness(sensor, 1);
    sensor->set_saturation(sensor, -2);
  }
  sensor->set_framesize(sensor, FRAMESIZE_VGA);

  connectWiFi();
  Serial.print("Wi-Fi connected, IP: ");
  Serial.println(WiFi.localIP());

  if (MDNS.begin(HOSTNAME)) {
    MDNS.addService("http", "tcp", 80);
    Serial.println("mDNS: rocky-cam.local");
  }
  server.begin();
  setupHttpOta();

  startWatchdog();
  lastSelfCheckMs = millis();

  Serial.println("==============================");
  Serial.println("     ROCKY CAMERA READY");
  Serial.println("==============================");
}

// ---------------------------------------------------------------- loop
unsigned long wifiDownSince = 0;

void loop() {
  feedWatchdog();
  otaServer.handleClient();

  // Wi-Fi supervision: try to reconnect, and reboot if it stays down.
  if (WiFi.status() != WL_CONNECTED) {
    if (wifiDownSince == 0) wifiDownSince = millis();
    if (millis() - wifiDownSince > 20000) WiFi.reconnect();
    if (millis() - wifiDownSince > 90000) {
      Serial.println("Wi-Fi down for 90s -- rebooting.");
      esp_restart();
    }
  } else {
    wifiDownSince = 0;
  }

  WiFiClient client = server.available();
  if (client) {
    handleClient(client);
    lastSelfCheckMs = millis();          // a real capture counts as a health check
  } else if (millis() - lastSelfCheckMs > 60000) {
    cameraSelfCheck();
    lastSelfCheckMs = millis();
  }
  delay(1);
}
