#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <driver/i2s.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <WiFi.h>
#include <esp_timer.h>
#include <Preferences.h>
#include <ArduinoOTA.h>
#include <ESPmDNS.h>
#include <WebServer.h>
#include <Update.h>
#include <esp_system.h>
#include <time.h>
#include "wake_word.h"
#include "wake_selftest_clip.h"

#define SDA_PIN 8
#define SCL_PIN 9

Adafruit_PWMServoDriver pwm(0x40);

// ================================
// WI-FI + TIME (for the OLED status bar)
// Same network as RockyCamera -- two devices joining one home Wi-Fi is fine.
// ================================
const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
bool wifiConnected = false;
bool g_oledReady = false;  // mirrors setup()'s local oledReady for the status command

// ================================
// CONTROL LINK (replaces USB serial as the RockyBrain protocol transport)
// Serial turned out to be fundamentally unreliable for the audio-heavy parts
// of this protocol -- see body.py's history comments on the Mac side for the
// full debugging story. WiFi/TCP has real flow control and much higher
// real throughput, and is also a required step toward Rocky eventually
// running fully untethered (no USB cable at all, only power). One
// persistent client connection is expected (from RockyBrain on the Mac),
// reconnected automatically if it drops.
// ================================
#define CONTROL_PORT 8266
WiFiServer controlServer(CONTROL_PORT);
WiFiClient controlClient;

// Rocky's own speaker and mic sit on the same small body with no acoustic
// echo cancellation -- without a cooldown, the mic reliably picks up
// Rocky's own just-played voice and the wake-word detector can trigger on
// it, starting a second autonomous listen cycle while RockyBrain is still
// confirming the first and corrupting the simple line-based protocol
// (confirmed live: garbled data on the control link right after a spoken
// response). Suppress wake-word ticking for a bit after any speech finishes
// to let the room's acoustic tail settle.
// Raised 1200 -> 2500 after live testing showed 1200ms still wasn't
// reliably enough: a spoken closing remark ("Here if you need me...")
// could self-trigger a new wake event right after resuming, which finds
// silence, which (on the Python side) speaks again, which can self-trigger
// again -- a real cascading loop, not just a one-off false positive.
// Raised again 2500 -> 4000 after lowering WW_DETECT_THRESHOLD (0.85 ->
// 0.70, see wake_word.h) to fix genuine misses on quieter attempts -- that
// same extra sensitivity made Rocky's own echoed voice cross the bar more
// easily too, and the loop came back. More cooldown margin compensates.
#define WAKE_COOLDOWN_MS 4000
unsigned long wakeWordCooldownUntil = 0;

// Set right after a wake-triggered capture is sent to RockyBrain, cleared
// the moment any new command arrives back from it. Without this, the ESP32
// kept re-arming wake-word detection while RockyBrain was still generating
// a response (often 10+ seconds) -- any residual noise in that window
// could re-trigger it, repeatedly firing the chime and queuing up more
// captures nobody was reading yet, corrupting the control-link protocol
// (confirmed live: continuous chiming with no response ever arriving).
bool awaitingBrainResponse = false;
// Set alongside awaitingBrainResponse, used by loop() to force-clear it if
// RockyBrain's next command never arrives at all (see AWAITING_BRAIN_TIMEOUT_MS
// below for why this exists -- this was a real, confirmed structural bug).
unsigned long awaitingBrainResponseSince = 0;
// The ONLY thing that ever clears awaitingBrainResponse is a new command
// successfully arriving from RockyBrain (see loop() below). If that command
// never arrives -- RockyBrain crashes mid-turn, its closing "dead"/"face"
// command fails to land during a bad connection window, anything -- this
// flag was staying true forever, and since it also gates BOTH tickWakeWord()
// (all "Hey Rocky" detection) and the periodic wake-word self-test/heal
// check, the board could never recover on its own: no further wake words
// would ever be detected, and the self-test that exists specifically to
// catch a dead wake-word detector never got a chance to run either. Only a
// full physical power-cycle ever cleared it -- this is almost certainly the
// actual root cause behind this session's whole recurring "stuck until I
// open the enclosure" pattern, not any single one of the narrower bugs
// fixed earlier tonight. 60s is generous for even a slow LLM+TTS turn
// (measured worst case this session: ~35s) with real margin, while still
// guaranteeing the board can never be wedged waiting on a response that is
// simply never coming.
#define AWAITING_BRAIN_TIMEOUT_MS 60000UL

// IST = UTC+5:30. Change gmtOffsetSec if this isn't your timezone.
const long gmtOffsetSec = 19800;
const int daylightOffsetSec = 0;

void connectWiFi() {
  Serial.print("Connecting to WiFi");
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 10000) {
    delay(300);
    Serial.print(".");
  }
  Serial.println();

  wifiConnected = (WiFi.status() == WL_CONNECTED);
  if (wifiConnected) {
    Serial.print("WiFi connected: ");
    Serial.println(WiFi.localIP());
    configTime(gmtOffsetSec, daylightOffsetSec, "pool.ntp.org", "time.google.com");
  } else {
    Serial.println("WiFi connect FAILED -- continuing without it.");
  }
}

int wifiBars() {
  // 0-3 arcs for the Wi-Fi glyph (not a 4-bar cellular scale).
  if (!wifiConnected || WiFi.status() != WL_CONNECTED) return 0;
  long rssi = WiFi.RSSI();
  if (rssi > -60) return 3;
  if (rssi > -75) return 2;
  if (rssi > -85) return 1;
  return 0;
}

String currentTimeString() {
  if (!wifiConnected) return "--:--";
  struct tm timeinfo;
  if (!getLocalTime(&timeinfo, 100)) return "--:--";
  char buf[6];
  snprintf(buf, sizeof(buf), "%02d:%02d", timeinfo.tm_hour, timeinfo.tm_min);
  return String(buf);
}

// ================================
// OLED face (SSD1306, 128x64, shares the I2C bus with the PCA9685)
// EMO-style: the eyes ARE the face -- shape/size/position carries the
// expression, no separate mouth for most states.
// ================================
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_ADDR 0x3C

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

int batteryPercent();  // defined below, near readBatteryVolts() -- used by drawStatusBar() first

String currentExpression = "neutral";
// Tracks whether Rocky is currently in the flattened idle "dead" pose (set
// by runDeadPose(), cleared by runStandPose()) -- lets a fresh wake word
// know whether it needs to physically stand up first, without re-running
// the stand animation on every follow-up turn in an ongoing conversation.
bool g_isDead = false;
String currentCaption = "";

// Status bar is 13px tall (y 0-12) -- taller than a typical tiny icon strip
// because a proper bold 3-band Wi-Fi glyph needs the room; the caption area
// is sized for size-2 text, so the face/eye area is the remainder.
#define STATUS_H 13

void drawWiFiIcon(int x, int y, int bars) {
  // Thin outline arcs -- a dot plus up to 3 concentric hairline arcs above it.
  if (!wifiConnected) return;
  int cy = y + 11;
  display.fillCircle(x, cy, 1, SSD1306_WHITE);
  if (bars >= 1) display.drawCircleHelper(x, cy, 4, 0x1 | 0x2, SSD1306_WHITE);
  if (bars >= 2) display.drawCircleHelper(x, cy, 7, 0x1 | 0x2, SSD1306_WHITE);
  if (bars >= 3) display.drawCircleHelper(x, cy, 10, 0x1 | 0x2, SSD1306_WHITE);
}

void drawStatusBar() {
  drawWiFiIcon(9, 0, wifiBars());

  // Time, centered -- shows "--:--" until Wi-Fi/NTP actually syncs
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  String t = currentTimeString();
  display.setCursor(SCREEN_WIDTH / 2 - (int)(t.length() * 6) / 2, 2);
  display.print(t);

  // Battery, top-right. Outline always drawn; the fill level only appears
  // once readBatteryVolts() confirms a real, physically plausible reading
  // (see its comment) -- an unwired pin stays an honest blank outline
  // rather than showing a fabricated percentage.
  int bx = SCREEN_WIDTH - 20;
  display.drawRect(bx, 3, 15, 7, SSD1306_WHITE);
  display.fillRect(bx + 15, 5, 2, 3, SSD1306_WHITE);
  int pct = batteryPercent();
  if (pct >= 0) {
    int fillW = (13 * pct) / 100;  // 1px inset on each side of the 15px outline
    if (fillW > 0) display.fillRect(bx + 1, 4, fillW, 5, SSD1306_WHITE);
  }
}

// Face area starts right after the status bar. Eyes ~40px wide each, 6px gap.
#define FACE_Y STATUS_H
#define EYE_W 40
#define EYE_GAP 6
#define EYE_L_X (SCREEN_WIDTH / 2 - EYE_GAP / 2 - EYE_W)
#define EYE_R_X (SCREEN_WIDTH / 2 + EYE_GAP / 2)

void drawEyePair(int w, int h, int radius, int leftX, int rightX, int y) {
  display.fillRoundRect(leftX, y, w, h, radius, SSD1306_WHITE);
  display.fillRoundRect(rightX, y, w, h, radius, SSD1306_WHITE);
}

void drawExpressionEyes() {
  if (currentExpression == "happy") {
    drawEyePair(EYE_W, 14, 7, EYE_L_X, EYE_R_X, FACE_Y + 10);        // squinted, closed-happy crescents
  } else if (currentExpression == "sad") {
    drawEyePair(28, 20, 9, EYE_L_X + 5, EYE_R_X - 5, FACE_Y + 10);   // smaller, drawn lower
  } else if (currentExpression == "curious") {
    display.fillRoundRect(EYE_L_X, FACE_Y - 1, EYE_W + 4, 34, 9, SSD1306_WHITE);  // bigger left eye
    display.fillRoundRect(EYE_R_X, FACE_Y + 6, EYE_W - 8, 20, 7, SSD1306_WHITE);  // smaller right eye
  } else if (currentExpression == "sleepy") {
    drawEyePair(EYE_W, 6, 3, EYE_L_X, EYE_R_X, FACE_Y + 12);         // thin closed bars
  } else if (currentExpression == "surprised") {
    drawEyePair(EYE_W + 4, 34, 11, EYE_L_X - 2, EYE_R_X - 2, FACE_Y - 1);  // big wide eyes
  } else {
    drawEyePair(EYE_W, 30, 9, EYE_L_X, EYE_R_X, FACE_Y);             // neutral
  }
}

void renderFace() {
  display.clearDisplay();
  drawStatusBar();
  drawExpressionEyes();
  display.display();
}

void drawCaptionText(String text);  // defined below, forward-declared for use here

// Single-shot, non-blocking frame draw with a caption -- unlike
// showCaptionChunks() this has no delay() of its own, so it can be called
// from inside the audio-streaming loop to advance the caption in step with
// actual playback progress without stalling the stream.
void renderFaceWithCaption(const String &captionChunk) {
  display.clearDisplay();
  drawStatusBar();
  drawExpressionEyes();
  if (captionChunk.length() > 0) {
    drawCaptionText(captionChunk);
  }
  display.display();
}

void setExpression(String expr) {
  currentExpression = expr;
  renderFace();
}

void blink() {
  String prevExpr = currentExpression;

  display.clearDisplay();
  drawStatusBar();
  drawEyePair(EYE_W, 4, 2, EYE_L_X, EYE_R_X, FACE_Y + 13);
  display.display();
  delay(120);

  currentExpression = prevExpr;
  renderFace();
}

// Captions: short phrase chunks that appear centered, hold briefly, then get
// replaced by the next chunk -- standard video-caption style, not a scroll.
// Size-2 text, single line, so each chunk is capped around 10 characters.
// durationMs (from the say:<ms>:<text> protocol) isn't used for pacing here
// -- chunks use a fixed hold time -- but the protocol still carries it for
// compatibility with the Python side.
void drawCaptionText(String text) {
  display.setTextSize(2);
  display.setTextColor(SSD1306_WHITE);
  int w = text.length() * 12;
  int x = (SCREEN_WIDTH - w) / 2;
  if (x < 0) x = 0;
  display.setCursor(x, FACE_Y + 34);
  display.print(text);
}

void showCaptionChunks(String text) {
  const int maxCharsPerChunk = 10;
  const int holdMs = 650;
  int start = 0;

  while (start < (int)text.length()) {
    int end = start + maxCharsPerChunk;
    if (end >= (int)text.length()) {
      end = text.length();
    } else {
      int lastSpace = text.lastIndexOf(' ', end);
      if (lastSpace > start) end = lastSpace;
    }
    String chunk = text.substring(start, end);
    chunk.trim();

    if (chunk.length() > 0) {
      display.clearDisplay();
      drawStatusBar();
      drawExpressionEyes();
      drawCaptionText(chunk);
      display.display();
      delay(holdMs);
    }

    start = end;
    while (start < (int)text.length() && text.charAt(start) == ' ') start++;
  }

  renderFace();
}

// PCA9685 pulse range used in our successful servo tests
#define SERVO_MIN 150
#define SERVO_MAX 600

// Every pose/movement angle in this file (stand, rest, walk, turn, and the
// whole pose library below) was remapped from the Sesame reference project
// through this robot's own measured calibration, not used as-is -- the
// reference's raw 0-180 assumptions don't match this robot's real geometry.
// Full data and derivation: calibration_log.md (scratchpad, this session).

// ================================
// Sesame servo mapping
// ================================
#define R1 0
#define R2 1
#define L1 2
#define L2 3
#define R4 4
#define R3 5
#define L3 6
#define L4 7

// Movement settings
int frameDelay = 150;
int walkCycles = 1;

// ================================
// I2S MIC (INMP441)
// ================================
#define I2S_WS  15
#define I2S_SCK 14
#define I2S_SD  13
#define I2S_PORT I2S_NUM_0

#define SAMPLE_RATE 16000
#define BUFFER_LEN  512

int32_t audioBuffer[BUFFER_LEN];

// Finite timeout for every I2S read/write. These were portMAX_DELAY (wait
// forever): if the mic or speaker DMA ever stalled once -- a power glitch,
// a loose wire -- the call never returned, loop() froze inside it, and the
// board stayed alive on Wi-Fi (ping works) but never serviced another
// command or heard another wake word until someone reset it. Confirmed as
// the cause of Rocky "stuck" after a speech transfer timed out.
#define I2S_TIMEOUT_TICKS pdMS_TO_TICKS(1000)

// ---- Mic high-pass filter ------------------------------------------------
// Measured on the real mic in a silent room: 88% of the signal's energy sat
// BELOW 30 Hz (strongest components 1-6 Hz), with peaks clipping at full
// scale -- slow drift, not sound. Speech lives at 100Hz-4kHz, so it was
// invisible next to that: total loudness read ~3000 whether or not anyone
// spoke, the silence detector could never tell when a sentence had ended,
// and nearly every recording ran to the full 8-second cap (a fixed ~8s of
// dead time per turn). Two cascaded 2nd-order Butterworth high-pass stages
// at 100 Hz remove it: room loudness ~3000 -> ~450 while speech stays
// ~1500-3400, a clean 4-7x separation. Same overall gain as the old ">> 14".
struct MicHighPass {
  float b0, b1, b2, a1, a2;
  float z1[2], z2[2];
  void init(float fc, float fs) {
    float w0 = 2.0f * 3.14159265f * fc / fs;
    float c = cosf(w0);
    float alpha = sinf(w0) / (2.0f * 0.70710678f);
    float a0 = 1.0f + alpha;
    b0 = (1.0f + c) / 2.0f / a0;
    b1 = -(1.0f + c) / a0;
    b2 = b0;
    a1 = -2.0f * c / a0;
    a2 = (1.0f - alpha) / a0;
    z1[0] = z1[1] = z2[0] = z2[1] = 0.0f;
  }
  float process(float x) {
    for (int s = 0; s < 2; s++) {
      float y = b0 * x + z1[s];
      z1[s] = b1 * x - a1 * y + z2[s];
      z2[s] = b2 * x - a2 * y;
      x = y;
    }
    return x;
  }
};
MicHighPass micHP;

// Raw 32-bit I2S sample -> filtered, clamped int16. The one conversion used
// by every mic consumer (wake word, recording/VAD, mic test).
inline int16_t micToPcm16(int32_t raw) {
  float y = micHP.process((float) raw * (1.0f / 16384.0f));
  if (y > 32767.0f) y = 32767.0f;
  if (y < -32768.0f) y = -32768.0f;
  return (int16_t) y;
}

// Loop watchdog. The stock Arduino/ESP-IDF task watchdog did NOT work here:
// tested with a deliberate infinite loop (the "hangtest" command), the board
// stayed frozen for 100s+ despite esp_task_wdt_init(40, true) -- the core
// pre-configures that watchdog and silently ignores the request. This one is
// deliberately simple and doesn't depend on it: an esp_timer callback (its
// own task, so it keeps running while loop() is stuck) reboots the chip if
// loop()/long operations haven't called feedWatchdog() for WATCHDOG_MS.
#define WATCHDOG_MS 40000
#define OTA_PASSWORD "change-me"  // LAN-only; change it if the network is shared
volatile unsigned long lastProgressMs = 0;

void feedWatchdog() { lastProgressMs = millis(); }

static void watchdogCheck(void *) {
  if (lastProgressMs != 0 && millis() - lastProgressMs > WATCHDOG_MS) {
    Serial.println("Watchdog: loop stalled -- rebooting.");
    esp_restart();
  }
}

// ---- HTTP firmware push (works through the Mac's firewall) ---------------
// ArduinoOTA needs the board to connect BACK to the Mac, which the macOS
// firewall blocks (confirmed: the upload authenticated, then "No response
// from device"). This is the reverse: the Mac connects OUT to the board and
// POSTs the firmware, so no firewall change is needed. Token-checked.
//   curl -H "X-Rocky-Token: <OTA_PASSWORD>" -F firmware=@firmware.bin http://rocky-body.local:8080/update
// tools/ota_push.py does exactly this; `pio run -e ota -t upload` calls it.
WebServer otaServer(8080);
static bool otaUploadAllowed = false;
String buildStatusLine();      // defined further down
float runWakeSelfTest();
String micLevelReport();
void setServoAngle(uint8_t channel, int angle);
void runStandPose();
void runRestPose();
void runWalkPose();
void runWalkBackward();
void runTurnLeft();
void runTurnRight();
void runWavePose();
void runDancePose();
void runSwimPose();
void runPointPose();
void runPushupPose();
void runBowPose();
void runCutePose();
void runFreakyPose();
void runWormPose();
void runShakePose();
void runShrugPose();
void runNodPose();
void runDeadPose();
void runCrabPose();

const char CONTROL_PAGE_HTML[] PROGMEM = R"HTMLPAGE(
<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Rocky Servo Control</title>
<style>
:root{color-scheme:light dark;--bg:#101512;--card:#1a211d;--line:#2b352f;--ink:#eef3ef;--muted:#8fa398;--accent:#4fd18b;--accent2:#5b9dd6;--warn:#e0a45c}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  padding:calc(env(safe-area-inset-top,0px) + 14px) 14px calc(env(safe-area-inset-bottom,0px) + 24px)}
h1{font-size:18px;margin:0 0 4px;letter-spacing:.02em}
p.sub{color:var(--muted);margin:0 0 16px;font-size:13px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px;margin-bottom:12px}
.poses{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:16px}
button{background:var(--accent2);color:#08110d;border:0;border-radius:9px;padding:12px 8px;font-size:14px;font-weight:600;
  -webkit-tap-highlight-color:transparent}
button:active{filter:brightness(0.85)}
button.stand{background:var(--accent)}
.row{display:flex;align-items:center;gap:10px;padding:10px 0;border-bottom:1px solid var(--line)}
.row:last-child{border-bottom:0}
.row label{width:38px;flex:none;font-weight:700;color:var(--accent2)}
.row input[type=range]{flex:1;accent-color:var(--accent)}
.row .val{width:38px;flex:none;text-align:right;font-variant-numeric:tabular-nums;color:var(--muted)}
.status{font-size:12px;color:var(--muted);min-height:16px;margin-top:6px}
.section-title{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0 0 8px}
</style></head>
<body>
<h1>Rocky Servo Control</h1>
<p class="sub">Drag a slider to move that joint live. Angles are 0-180.</p>

<p class="section-title">Poses</p>
<p class="section-title">Movement</p>
<div class="poses">
  <button class="stand" onclick="pose('stand')">Stand</button>
  <button onclick="pose('rest')">Rest</button>
  <button onclick="pose('forward')">Forward</button>
  <button onclick="pose('backward')">Backward</button>
  <button onclick="pose('left')">Left</button>
  <button onclick="pose('right')">Right</button>
</div>

<p class="section-title">Tricks</p>
<div class="poses">
  <button onclick="pose('wave')">Wave</button>
  <button onclick="pose('dance')">Dance</button>
  <button onclick="pose('swim')">Swim</button>
  <button onclick="pose('point')">Point</button>
  <button onclick="pose('pushup')">Pushup</button>
  <button onclick="pose('bow')">Bow</button>
  <button onclick="pose('cute')">Cute</button>
  <button onclick="pose('freaky')">Freaky</button>
  <button onclick="pose('worm')">Worm</button>
  <button onclick="pose('shake')">Shake</button>
  <button onclick="pose('shrug')">Shrug</button>
  <button onclick="pose('dead')">Dead</button>
  <button onclick="pose('crab')">Crab</button>
  <button onclick="pose('nod')">Nod</button>
</div>

<div class="card" id="servos"></div>
<div class="status" id="status">Ready.</div>

<script>
const SERVOS = [
  {ch:0,name:"R1"},{ch:1,name:"R2"},{ch:2,name:"L1"},{ch:3,name:"L2"},
  {ch:4,name:"R4"},{ch:5,name:"R3"},{ch:6,name:"L3"},{ch:7,name:"L4"}
];
const wrap = document.getElementById('servos');
const statusEl = document.getElementById('status');
let timers = {};

SERVOS.forEach(s => {
  const row = document.createElement('div');
  row.className = 'row';
  row.innerHTML = `<label>${s.name}</label>
    <input type="range" min="0" max="180" value="90" id="s${s.ch}">
    <span class="val" id="v${s.ch}">90</span>`;
  wrap.appendChild(row);
  const slider = row.querySelector('input');
  const val = row.querySelector('.val');
  slider.addEventListener('input', () => {
    val.textContent = slider.value;
    clearTimeout(timers[s.ch]);
    timers[s.ch] = setTimeout(() => moveServo(s.ch, slider.value), 60);
  });
});

function setStatus(msg, isErr) {
  statusEl.textContent = msg;
  statusEl.style.color = isErr ? '#e0685c' : '';
}

function moveServo(ch, angle) {
  fetch(`/servo?ch=${ch}&angle=${angle}`)
    .then(r => r.text()).then(t => setStatus(t))
    .catch(e => setStatus('Error: ' + e, true));
}

function pose(name) {
  setStatus(`Running ${name}...`);
  fetch(`/pose?name=${name}`)
    .then(r => r.text()).then(t => setStatus(t))
    .catch(e => setStatus('Error: ' + e, true));
}
</script>
</body></html>
)HTMLPAGE";

void setupHttpOta() {
  static const char *headerKeys[] = {"X-Rocky-Token"};
  otaServer.collectHeaders(headerKeys, 1);

  otaServer.on("/version", HTTP_GET, []() {
    otaServer.send(200, "text/plain", String(__DATE__) + " " + __TIME__);
  });
  // Read-only health endpoints, so diagnose.py can check the board while
  // brain.py holds the (single) control connection.
  otaServer.on("/status", HTTP_GET, []() { otaServer.send(200, "text/plain", buildStatusLine()); });
  otaServer.on("/selftest", HTTP_GET, []() {
    float peak = runWakeSelfTest();
    otaServer.send(200, "text/plain", String("peak=") + String(peak, 3));
  });
  otaServer.on("/mic", HTTP_GET, []() { otaServer.send(200, "text/plain", micLevelReport()); });

  // ---- Manual servo/pose control page -- for calibrating servo angles by
  // hand from a phone, independent of RockyBrain/the control-socket link
  // entirely (works even with the Mac off). Lives on the same read-mostly
  // diagnostics server as /status et al.
  otaServer.on("/servo", HTTP_GET, []() {
    if (!otaServer.hasArg("ch") || !otaServer.hasArg("angle")) {
      otaServer.send(400, "text/plain", "usage: /servo?ch=0-7&angle=0-180"); return;
    }
    int ch = otaServer.arg("ch").toInt();
    int angle = otaServer.arg("angle").toInt();
    if (ch < 0 || ch > 7 || angle < 0 || angle > 180) {
      otaServer.send(400, "text/plain", "ch must be 0-7, angle 0-180"); return;
    }
    setServoAngle((uint8_t) ch, angle);
    otaServer.send(200, "text/plain", "OK ch=" + String(ch) + " angle=" + String(angle));
  });
  otaServer.on("/pose", HTTP_GET, []() {
    String name = otaServer.arg("name");
    if (name == "stand") runStandPose();
    else if (name == "rest") runRestPose();
    else if (name == "forward") runWalkPose();
    else if (name == "backward") runWalkBackward();
    else if (name == "left") runTurnLeft();
    else if (name == "right") runTurnRight();
    else if (name == "wave") runWavePose();
    else if (name == "dance") runDancePose();
    else if (name == "swim") runSwimPose();
    else if (name == "point") runPointPose();
    else if (name == "pushup") runPushupPose();
    else if (name == "bow") runBowPose();
    else if (name == "cute") runCutePose();
    else if (name == "freaky") runFreakyPose();
    else if (name == "worm") runWormPose();
    else if (name == "shake") runShakePose();
    else if (name == "shrug") runShrugPose();
    else if (name == "nod") runNodPose();
    else if (name == "dead") runDeadPose();
    else if (name == "crab") runCrabPose();
    else { otaServer.send(400, "text/plain", "unknown pose: " + name); return; }
    otaServer.send(200, "text/plain", "OK pose=" + name);
  });
  otaServer.on("/control", HTTP_GET, []() {
    otaServer.send(200, "text/html", CONTROL_PAGE_HTML);
  });

  otaServer.on("/update", HTTP_POST,
    []() {  // upload finished
      bool ok = otaUploadAllowed && !Update.hasError();
      otaServer.send(ok ? 200 : 500, "text/plain", ok ? "OK - rebooting into the new firmware" : "FAILED");
      delay(400);
      if (ok) ESP.restart();
    },
    []() {  // upload data
      HTTPUpload &up = otaServer.upload();
      if (up.status == UPLOAD_FILE_START) {
        otaUploadAllowed = otaServer.header("X-Rocky-Token") == OTA_PASSWORD;
        if (!otaUploadAllowed) { Serial.println("HTTP OTA: bad token, rejected."); return; }
        Serial.printf("HTTP OTA: receiving %s\n", up.filename.c_str());
        if (!Update.begin(UPDATE_SIZE_UNKNOWN)) Update.printError(Serial);
      } else if (up.status == UPLOAD_FILE_WRITE && otaUploadAllowed) {
        if (Update.write(up.buf, up.currentSize) != up.currentSize) Update.printError(Serial);
        feedWatchdog();
      } else if (up.status == UPLOAD_FILE_END && otaUploadAllowed) {
        if (Update.end(true)) Serial.printf("HTTP OTA: %u bytes written.\n", (unsigned) up.totalSize);
        else Update.printError(Serial);
      }
    });
  otaServer.begin();
}

void startWatchdog() {
  feedWatchdog();
  esp_timer_create_args_t args = {};
  args.callback = &watchdogCheck;
  args.name = "loopwdt";
  esp_timer_handle_t timer;
  esp_err_t e1 = esp_timer_create(&args, &timer);
  esp_err_t e2 = esp_timer_start_periodic(timer, 1000000);  // check once a second
  Serial.printf("Watchdog armed (create=%d start=%d, %dms)\n", (int)e1, (int)e2, WATCHDOG_MS);
}

void setupMic() {
  micHP.init(100.0f, (float) SAMPLE_RATE);
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 4,
    .dma_buf_len = BUFFER_LEN,
    .use_apll = false
  };

  i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_SD
  };

  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_PORT, &pin_config);
}

void runMicTest(WiFiClient &client) {
  Serial.println("MIC TEST — reading 5 seconds of audio levels...");
  client.println("MIC TEST — reading 5 seconds of audio levels...");
  unsigned long start = millis();

  while (millis() - start < 5000) {
    size_t bytesRead = 0;
    i2s_read(I2S_PORT, audioBuffer, sizeof(audioBuffer), &bytesRead, I2S_TIMEOUT_TICKS);

    int samplesRead = bytesRead / sizeof(int32_t);
    int64_t sumSquares = 0;
    for (int i = 0; i < samplesRead; i++) {
      int32_t sample = micToPcm16(audioBuffer[i]);
      sumSquares += (int64_t)sample * sample;
    }
    float rms = sqrt((float)sumSquares / samplesRead);

    Serial.print("Mic level: ");
    Serial.println(rms);
    client.print("Mic level: ");
    client.println(rms);
  }

  Serial.println("MIC TEST done.");
  client.println("MIC TEST done.");
}

// Standalone bench test for the on-device wake-word detector (see
// wake_word.h) -- runs it against live mic audio and reports every smoothed
// score plus DETECTED markers, without touching the main control loop. Lets
// us validate the trained model against the real room/mic before wiring it
// into loop() to auto-trigger runListen(), the same "test the primitive in
// isolation first" approach used to calibrate the VAD threshold.
void runWakeWordTest(WiFiClient &client, unsigned long duration_ms) {
  wakeword::reset();  // start from a clean baseline, and see reset() below --
                       // without this, leftover smoothing/model state from
                       // this diagnostic run bleeds into whatever runs next
  Serial.println("WAKE WORD TEST -- say \"hey rocky\" during this window.");
  client.print("WAKE WORD TEST for ");
  client.print(duration_ms / 1000);
  client.println("s -- say \"hey rocky\".");

  int16_t chunk[BUFFER_LEN];
  unsigned long start = millis();

  while (millis() - start < duration_ms) {
    size_t bytesRead = 0;
    i2s_read(I2S_PORT, audioBuffer, sizeof(audioBuffer), &bytesRead, I2S_TIMEOUT_TICKS);
    int samplesRead = bytesRead / sizeof(int32_t);

    for (int i = 0; i < samplesRead; i++) {
      int32_t sample32 = micToPcm16(audioBuffer[i]);
      if (sample32 > 32767) sample32 = 32767;
      if (sample32 < -32768) sample32 = -32768;
      chunk[i] = (int16_t) sample32;
    }

    bool detected = false;
    float smoothed = wakeword::processAudio(chunk, samplesRead, &detected);

    if (smoothed >= 0.0f) {
      client.print("WW_DEBUG score=");
      client.print(smoothed, 3);
      if (detected) {
        client.print(" *** DETECTED ***");
      }
      client.println();
    }
  }

  wakeword::reset();  // leave state clean for normal operation afterward,
                       // not mid-window from this test's own last ticks
  Serial.println("WAKE WORD TEST done.");
  client.println("WAKE WORD TEST done.");
}

// ================================
// SPEAKER (MAX98357A I2S amp)
// Independent I2S peripheral (I2S_NUM_1, TX) from the mic's (I2S_NUM_0, RX)
// -- the ESP32-S3 has two, so input and output don't have to share one.
// ================================
#define SPK_BCLK 4
#define SPK_LRC  5
#define SPK_DIN  6
#define SPK_I2S_PORT I2S_NUM_1
#define SPK_SAMPLE_RATE 22050  // matches RockyBrain's Piper voice model's native rate
// VIN moved from 3.3V to 5V for more amp headroom -- retesting at full
// digital scale now that the amp's own clipping point should be higher.
// Prior A/B history at 3.3V VIN: 0.35 clean/quiet, 1.0 distorted, 0.6
// clean/quiet, 0.8 clean/louder. Back off toward 0.8 if this distorts again.
// Speaker volume, 0-100%, persisted in flash (NVS) so it survives reboots.
// Percent maps to gain on a perceptual curve (pct^1.66: 50% is about -10 dB,
// i.e. "half as loud" to the ear) instead of a linear scale, where 50% would
// barely sound quieter than 100%. 100% is the old fixed SPK_VOLUME_SCALE 1.0,
// which was tested clean on the 5V-powered amp.
// Battery monitor hook. GPIO1 chosen because it's a free ADC1 pin (ADC1 is
// safe to read with WiFi active, unlike ADC2) that isn't a strapping pin
// (GPIO0/3/45/46 are) and isn't already claimed by the mic/speaker/I2C
// wiring (GPIO4,5,6,8,9,13,14,15).
//
// Actual pack: a single 3.7V Li-ion cell into a boost converter that
// supplies a regulated 5V rail to everything else -- NOT the 2S pack this
// was originally sized for. Critically, the divider below must tap the
// RAW CELL (the battery's own +/- leads, before the boost converter's
// input), never the boosted 5V output: a boost converter's entire job is
// to hold its output steady as the input sags, so the 5V rail reads ~5V
// right up until the cell is nearly empty and the converter can no longer
// keep up, then collapses fast -- it would carry no usable percentage
// information at all.
//
// Single-cell Li-ion range is 3.0V (empty/cutoff) - 4.2V (full), which
// still exceeds the ESP32-S3 ADC's ~3.3V max, so a divider is still
// required. 100k/100k (equal resistors, simplest to source) gives ratio
// 2.0 and keeps a full cell at ~2.1V at the ADC pin, comfortably inside
// range with good headroom for accuracy -- add a small (~100nF) cap across
// the bottom (ADC-side) resistor to smooth the reading. Still needs that
// physical divider wired to the raw cell before the reading means anything
// -- readBatteryVolts() below refuses to report a value outside what a
// real single cell could ever produce, so an unwired/floating pin still
// shows the honest blank icon instead of a fabricated percentage.
#define BATTERY_ADC_PIN 1
#define BATTERY_DIVIDER_RATIO 2.0f
#define BATTERY_MIN_V 3.0f          // single-cell Li-ion empty/cutoff
#define BATTERY_MAX_V 4.2f          // single-cell Li-ion full charge
#define BATTERY_PLAUSIBLE_MIN_V 2.5f  // below this: not a real cell reading -- floating/unwired pin
#define BATTERY_PLAUSIBLE_MAX_V 4.4f  // above this: same -- treat as unwired rather than fabricate

float readBatteryVolts() {
#if BATTERY_ADC_PIN >= 0
  float v = (float) analogReadMilliVolts(BATTERY_ADC_PIN) * BATTERY_DIVIDER_RATIO / 1000.0f;
  if (v < BATTERY_PLAUSIBLE_MIN_V || v > BATTERY_PLAUSIBLE_MAX_V) return -1.0f;
  return v;
#else
  return -1.0f;
#endif
}

int batteryPercent() {
  float v = readBatteryVolts();
  if (v < 0) return -1;
  float pct = (v - BATTERY_MIN_V) / (BATTERY_MAX_V - BATTERY_MIN_V) * 100.0f;
  if (pct < 0) pct = 0;
  if (pct > 100) pct = 100;
  return (int) (pct + 0.5f);
}

Preferences prefs;
int g_volumePct = 100;
float g_spkVolume = 1.0f;

float volumePctToScale(int pct) {
  if (pct <= 0) return 0.0f;
  if (pct >= 100) return 1.0f;
  return powf((float) pct / 100.0f, 1.66f);
}

void setVolumePct(int pct, bool save) {
  if (pct < 0) pct = 0;
  if (pct > 100) pct = 100;
  g_volumePct = pct;
  g_spkVolume = volumePctToScale(pct);
  if (save) {
    prefs.begin("rocky", false);
    prefs.putInt("vol", pct);
    prefs.end();
  }
}

// Chimes and test tones follow the volume too, but keep a floor so a low (not
// muted) setting doesn't make the wake chime inaudible.
float chimeGain() {
  if (g_volumePct <= 0) return 0.0f;
  return g_spkVolume < 0.12f ? 0.12f : g_spkVolume;
}

void setupSpeaker() {
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX),
    .sample_rate = SPK_SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    // Stereo format, same sample duplicated to both channels on write --
    // more robust across MAX98357A clones than relying on a specific
    // left/right/average selection from the floating SD pin.
    .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,   // 8x256 samples = ~93ms of cushion (was 4 = ~46ms): a
                          // caption redraw over I2C can block playback for
                          // ~100ms, which used to underrun the DMA
    .dma_buf_len = 256,
    .use_apll = false,
    .tx_desc_auto_clear = true  // Without this, the I2S peripheral re-loops
                                // its last DMA buffer forever once it runs
                                // out of new data instead of going silent --
                                // a known ESP32 I2S gotcha. This is why the
                                // wake chime kept audibly repeating for the
                                // entire "thinking" gap before a response,
                                // and very likely the source of the faint
                                // post-speech buzz noticed much earlier in
                                // this project (just too quiet to notice
                                // then -- a multi-second gap after a short
                                // chime makes the same underlying bug far
                                // more obvious).
  };

  i2s_pin_config_t pin_config = {
    .bck_io_num = SPK_BCLK,
    .ws_io_num = SPK_LRC,
    .data_out_num = SPK_DIN,
    .data_in_num = I2S_PIN_NO_CHANGE
  };

  i2s_driver_install(SPK_I2S_PORT, &i2s_config, 0, NULL);
  i2s_set_pin(SPK_I2S_PORT, &pin_config);
}

void runSpeakerTest() {
  Serial.println("SPEAKER TEST -- playing a 1kHz tone for 2 seconds...");

  const int freq = 1000;
  const int durationMs = 2000;
  const int totalSamples = (SPK_SAMPLE_RATE * durationMs) / 1000;
  const int chunkSamples = 256;

  int16_t buffer[chunkSamples * 2];  // interleaved left/right

  int samplesWritten = 0;
  while (samplesWritten < totalSamples) {
    int samplesThisChunk = min(chunkSamples, totalSamples - samplesWritten);
    for (int i = 0; i < samplesThisChunk; i++) {
      float t = (float)(samplesWritten + i) / SPK_SAMPLE_RATE;
      int16_t sample = (int16_t)(3000 * chimeGain() * sin(2 * PI * freq * t));  // modest volume
      buffer[i * 2] = sample;      // left
      buffer[i * 2 + 1] = sample;  // right
    }
    size_t bytesWritten = 0;
    i2s_write(SPK_I2S_PORT, buffer, samplesThisChunk * 2 * sizeof(int16_t), &bytesWritten, I2S_TIMEOUT_TICKS);
    samplesWritten += samplesThisChunk;
  }

  Serial.println("SPEAKER TEST done.");
}

// Short two-note "ding-doot" acknowledgment chime, played the instant the
// wake word is detected (see tickWakeWord()) so there's immediate local
// feedback that Rocky is now listening -- synthesized and played directly
// on-device (no round trip to RockyBrain), so it's genuinely instant rather
// than waiting on anything over WiFi.
void playWakeChime() {
  const int chunkSamples = 128;
  int16_t buffer[chunkSamples * 2];

  auto playTone = [&](int freq, int durationMs) {
    int totalSamples = (SPK_SAMPLE_RATE * durationMs) / 1000;
    int samplesWritten = 0;
    while (samplesWritten < totalSamples) {
      int samplesThisChunk = min(chunkSamples, totalSamples - samplesWritten);
      for (int i = 0; i < samplesThisChunk; i++) {
        float t = (float)(samplesWritten + i) / SPK_SAMPLE_RATE;
        int16_t sample = (int16_t)(3000 * chimeGain() * sin(2 * PI * freq * t));
        buffer[i * 2] = sample;
        buffer[i * 2 + 1] = sample;
      }
      size_t bytesWritten = 0;
      i2s_write(SPK_I2S_PORT, buffer, samplesThisChunk * 2 * sizeof(int16_t), &bytesWritten, I2S_TIMEOUT_TICKS);
      samplesWritten += samplesThisChunk;
    }
  };

  playTone(1046, 70);  // C6
  playTone(1568, 90);  // G6
}

// Single short, quiet "blip" -- NOT a wake acknowledgment, so intentionally
// distinct from playWakeChime()'s two-note pattern: this plays every time
// runListen() starts from the "listen" command (a normal follow-up turn AND
// the silent retry after a transcript RockyBrain couldn't use). Both of
// those used to be completely silent, and combined with runListen()'s own
// silent "Listening..." OLED caption, a failed/uncertain turn looked
// identical to a dead board from across the room -- confirmed live, this is
// what a "not responding" report actually was on one real turn (the
// transcript was ambiguous, RockyBrain correctly stayed silent rather than
// guess, and the user had zero cue a second listening window had even
// opened). A plain tone (not speech) can't be picked up by the wake-word
// model, so this carries no self-trigger risk the way spoken feedback would.
void playFollowUpChime() {
  const int chunkSamples = 128;
  int16_t buffer[chunkSamples * 2];
  int totalSamples = (SPK_SAMPLE_RATE * 55) / 1000;
  int samplesWritten = 0;
  while (samplesWritten < totalSamples) {
    int samplesThisChunk = min(chunkSamples, totalSamples - samplesWritten);
    for (int i = 0; i < samplesThisChunk; i++) {
      float t = (float)(samplesWritten + i) / SPK_SAMPLE_RATE;
      int16_t sample = (int16_t)(2200 * chimeGain() * sin(2 * PI * 784 * t));  // G5, quieter/shorter than the wake chime
      buffer[i * 2] = sample;
      buffer[i * 2 + 1] = sample;
    }
    size_t bytesWritten = 0;
    i2s_write(SPK_I2S_PORT, buffer, samplesThisChunk * 2 * sizeof(int16_t), &bytesWritten, I2S_TIMEOUT_TICKS);
    samplesWritten += samplesThisChunk;
  }
}

// ================================
// SPEECH PLAYBACK
// Receives mono 16-bit PCM audio (at SPK_SAMPLE_RATE) sent right after the
// "speak:<byteCount>:<text>\n" command line, buffers the WHOLE thing in
// PSRAM first, then plays it back -- NOT a live stream-while-playing design.
//
// That two-phase split exists because streaming-while-playing was tried
// first and reliably lost data: direct testing (a checksum-comparison debug
// command, "checktest") proved that once Python sends audio faster than the
// firmware can actually drain it, bytes get silently dropped somewhere in
// the native USB-CDC path -- well before any buffer this code controls
// fills, and regardless of whether the firmware writes anything back
// concurrently (a per-chunk ack handshake was also tried and failed the
// same way). The transfer was NOT corrupting data, it was truncating it,
// consistently around the same byte count regardless of chunk size or
// write strategy. Slow, paced sending (Python side) reliably delivers
// everything; the only way to reconcile "must send slowly to be reliable"
// with "must play in real time" is to decouple the two entirely -- hence
// buffer first, play second. The ESP32-S3 has 8MB of PSRAM, easily enough
// for a several-second utterance.
//
// The OLED caption is paced off actual PLAYBACK progress (phase 2), not
// receive progress, since phase 2 is what "in sync with the words" means.
// ================================
void playStreamedSpeech(String caption, int totalBytes, WiFiClient &client) {
  const int maxCharsPerChunk = 10;
  const int maxChunks = 24;
  String chunks[maxChunks];
  int numChunks = 0;
  {
    int start = 0;
    while (start < (int)caption.length() && numChunks < maxChunks) {
      int end = start + maxCharsPerChunk;
      if (end >= (int)caption.length()) {
        end = caption.length();
      } else {
        int lastSpace = caption.lastIndexOf(' ', end);
        if (lastSpace > start) end = lastSpace;
      }
      String chunk = caption.substring(start, end);
      chunk.trim();
      if (chunk.length() > 0) {
        chunks[numChunks++] = chunk;
      }
      start = end;
      while (start < (int)caption.length() && caption.charAt(start) == ' ') start++;
    }
  }

  const int bytesPerSample = 2;
  float totalDurationMs = totalBytes > 0
      ? ((float)totalBytes / bytesPerSample / SPK_SAMPLE_RATE) * 1000.0
      : 1.0;
  float msPerChunk = numChunks > 0 ? totalDurationMs / numChunks : totalDurationMs;

  // -------- Streaming receive + play ---------------------------------
  // Plays while it receives, with a small jitter buffer, instead of buffering
  // the ENTIRE clip first. Before, every sentence waited for its whole
  // transfer (~0.5s+, and the network can spike far beyond that) before a
  // single sound came out, and again between every sentence. The original
  // buffer-then-play design existed because streaming over USB serial
  // silently dropped bytes (see the history comment above); over Wi-Fi/TCP
  // there is real flow control and ordering, so that constraint is gone.
  // The whole clip still lands in PSRAM (so the integrity check and the
  // caption timing work as before) -- playback just starts once ~250ms of
  // it has arrived, and reception keeps running ahead of playback, so a
  // network hiccup has to outlast the buffered audio before it is audible.
  uint8_t *audioBuf = (uint8_t *) ps_malloc(totalBytes > 0 ? totalBytes : 1);
  if (audioBuf == nullptr) {
    client.println("ERROR: failed to allocate PSRAM audio buffer for speech.");
    return;
  }

  const int PLAY_CHUNK = 512;                 // bytes of mono PCM per i2s_write()
  const int PREBUFFER_BYTES = 11025;          // ~250ms of audio at 22.05kHz mono 16-bit
  const unsigned long STALL_MS = 5000;        // no new data for this long = give up
  int16_t stereoBuf[(PLAY_CHUNK / 2) * 2];

  int bytesReceived = 0;
  int played = 0;
  int currentChunkIndex = 0;
  bool started = false;
  bool aborted = false;
  unsigned long playStart = 0;
  unsigned long lastDataMs = millis();
  const int prebuffer = min(PREBUFFER_BYTES, totalBytes);

  while (played < totalBytes) {
    // 1) Take whatever has arrived. Never block here: a blocking read would
    //    starve playback while waiting for the next packet.
    int avail = client.available();
    if (avail > 0 && bytesReceived < totalBytes) {
      int toRead = min(min(avail, totalBytes - bytesReceived), 4096);
      int n = client.read(audioBuf + bytesReceived, toRead);
      if (n > 0) {
        bytesReceived += n;
        lastDataMs = millis();
      }
    }

    // 2) Start once the jitter buffer has filled.
    if (!started && bytesReceived >= prebuffer) {
      started = true;
      playStart = millis();
      if (numChunks > 0) renderFaceWithCaption(chunks[0]);
    }

    // 3) Play one chunk of whole samples if any is buffered.
    int ready = (bytesReceived - played) & ~1;
    if (started && ready >= 2) {
      int thisChunk = min(PLAY_CHUNK, ready);
      int samples = thisChunk / bytesPerSample;
      for (int i = 0; i < samples; i++) {
        int idx = played + i * 2;
        int16_t raw = (int16_t)(audioBuf[idx] | (audioBuf[idx + 1] << 8));
        int16_t sample = (int16_t)(raw * g_spkVolume);
        stereoBuf[i * 2] = sample;
        stereoBuf[i * 2 + 1] = sample;
      }
      size_t bytesWritten = 0;
      esp_err_t writeErr = i2s_write(SPK_I2S_PORT, stereoBuf, samples * 2 * sizeof(int16_t), &bytesWritten, I2S_TIMEOUT_TICKS);
      if (writeErr != ESP_OK || bytesWritten == 0) {
        Serial.println("Speaker I2S write stalled -- aborting playback.");
        aborted = true;
        break;
      }
      played += thisChunk;

      if (numChunks > 0) {
        unsigned long elapsed = millis() - playStart;
        int targetChunk = (int)(elapsed / msPerChunk);
        if (targetChunk >= numChunks) targetChunk = numChunks - 1;
        if (targetChunk != currentChunkIndex) {
          currentChunkIndex = targetChunk;
          renderFaceWithCaption(chunks[currentChunkIndex]);
        }
      }
    } else {
      // Nothing playable yet (still pre-buffering) or the network fell
      // behind playback. Wait a moment for more data, but not forever.
      if (millis() - lastDataMs > STALL_MS) {
        aborted = true;
        break;
      }
      delay(1);
    }
    feedWatchdog();  // long clips legitimately outlast the watchdog window
  }

  // If we stopped early, swallow whatever payload is still in flight so the
  // command stream stays aligned for the next command.
  if (bytesReceived < totalBytes) {
    unsigned long t = millis();
    uint8_t scratch[512];
    while (bytesReceived < totalBytes && millis() - t < 8000) {
      int a = client.available();
      if (a > 0) {
        int n = client.read(scratch, min(a, (int) sizeof(scratch)));
        if (n > 0) { bytesReceived += n; t = millis(); }
      } else {
        delay(1);
      }
      feedWatchdog();
    }
  }

  free(audioBuf);
  renderFace();

  if (aborted || played != totalBytes) {
    client.print("TRANSFER_INCOMPLETE:");
    client.print(played);
    client.print(":");
    client.println(totalBytes);
  }
  client.println("Speech playback done.");
}

// ================================
// AUDIO RECORD -- voice-activity detection, not a fixed window
// Stops recording as soon as it detects sustained silence AFTER real speech,
// instead of always waiting out a fixed duration -- most utterances are
// well under the old fixed 3s window, so this directly cuts turn latency.
// A hard max duration remains as a safety net in case VAD never triggers
// (e.g. continuous background noise). listen.py on the Python side already
// reads whatever byte count is reported here, not a fixed value, so no
// change was needed there for variable-length recordings.
// ================================
#define MAX_RECORD_SECONDS 25  // Was 8 -- still hit in practice on a real,
                              // longer question (confirmed live: a genuine
                              // multi-sentence question ran past 8s without
                              // a long enough pause to trip the silence-hold
                              // stop below, and got cut off mid-sentence).
                              // Raised, not removed: this is a safety net
                              // against a mic that somehow never sees the
                              // ~1s of silence the hold-based stop needs
                              // (e.g. music/TV running), not the primary
                              // stop mechanism -- see the tolerant
                              // silence-hold logic below for that. 25s
                              // covers any realistic spoken question with
                              // room to spare; PSRAM cost is trivial
                              // (25s * 16kHz * 2 bytes = ~780KB of 8MB).
#define MAX_RECORD_SAMPLES (SAMPLE_RATE * MAX_RECORD_SECONDS)

#define VAD_MIN_RECORD_MS 400     // always record at least this long first --
                                  // also used as the noise-floor calibration
                                  // window (see below)
#define VAD_CHUNK_MS 32  // BUFFER_LEN samples / SAMPLE_RATE, i.e. how much
                         // audio one i2s_read() chunk covers -- used to pace
                         // the tolerant silence-hold accumulator below
// Raised 700 -> 1300: follow-up conversation turns are more spontaneous
// than the first "hey rocky" question (which people tend to have already
// decided on before speaking), so they're more likely to include a natural
// mid-thought pause while formulating the next thing to say. 700ms was
// short enough to mistake that pause for "done talking" and cut the
// recording off early, which is what was making follow-ups feel rushed.
#define VAD_SPEECH_START_CHUNKS 4  // ~128ms of net sustained loudness before speech counts as started
#define VAD_SILENCE_HOLD_MS 1000  // stop after this much continuous silence (was 1300)...
                                  // ...but only once real speech has been heard

// A single hardcoded RMS threshold doesn't survive contact with a real room:
// live testing showed a nearby fan alone produces broadband RMS bursts into
// the low thousands, overlapping the range real speech falls into during
// pauses -- so a fixed value tuned for one fan/room state is wrong for any
// other. Instead, calibrate off the actual ambient noise during the first
// VAD_MIN_RECORD_MS of each recording (before the user has necessarily
// started talking) and require RMS to clear that floor by a margin to count
// as speech/non-silence. This adapts per-recording instead of assuming a
// fixed environment, though a fan with loud transient bursts can still cause
// an early speech-detected reading, or occasionally look like a burst of
// speech resuming -- there's no way to fully separate that from voice using
// amplitude alone. Clamp the floor itself so one freak loud/quiet sample
// during calibration can't send the margin wildly off.
// Retuned for the FILTERED signal (see MicHighPass): room noise is now ~450
// with bursts to ~1300, speech ~1500-3400. Threshold = the larger of 2.5x
// the floor or floor+VAD_MARGIN, so it scales with the room instead of
// being a fixed offset tuned for the old drift-inflated levels (1800 was
// above most real speech once the drift was gone).
#define VAD_NOISE_FLOOR_MIN 100
#define VAD_NOISE_FLOOR_MAX 1500
#define VAD_MARGIN 600

int16_t* recordBuffer = nullptr;

// Continuously updated during idle wake-word ticking (see tickWakeWord()),
// while nobody is actively about to speak -- an exponential moving average
// of the TRUE ambient noise floor. This exists because runListen()'s own
// leading-window calibration routinely gets contaminated: the wake chime is
// the "talk now" cue, so people reliably start talking right as recording
// begins, right inside VAD_MIN_RECORD_MS. This pre-computed estimate, taken
// from before the recording even started, isn't biased by that.
float g_ambientNoiseFloor = 500.0f;  // sane default before any real samples
bool g_ambientNoiseFloorReady = false;

void runListen(WiFiClient &client) {
  if (recordBuffer == nullptr) {
    recordBuffer = (int16_t*) ps_malloc(MAX_RECORD_SAMPLES * sizeof(int16_t));
    if (recordBuffer == nullptr) {
      client.println("ERROR: failed to allocate PSRAM audio buffer.");
      return;
    }
  }

  client.println("LISTENING...");
  renderFaceWithCaption("Listening");  // shared by both the wake-triggered
                                        // path and a plain "listen" command
                                        // (e.g. a conversation follow-up),
                                        // since both call this same function

  int totalSamples = 0;
  unsigned long recordStart = millis();
  bool speechDetected = false;
  // "Leaky bucket" progress toward the required continuous-silence hold,
  // instead of a hard reset on any loud reading. Live testing showed the
  // room's bursty noise (fan, etc.) routinely spikes above threshold for
  // one or two chunks even during genuine pauses, and a hard reset made
  // that noise repeatedly restart the whole 700ms countdown from zero --
  // recordings almost never stopped early because of it. A brief blip now
  // only costs some progress (and sustained real speech still drains it
  // fast enough to correctly prevent stopping), rather than wiping it out.
  float silenceProgressMs = 0;
  int loudRunChunks = 0;  // see VAD_SPEECH_START_CHUNKS

  float noiseFloorMin = 1e9;  // running minimum, not average -- see comment below
  float vadThreshold = 0;  // set once calibration finishes
  bool calibrated = false;

  while (totalSamples < MAX_RECORD_SAMPLES) {
    size_t bytesRead = 0;
    i2s_read(I2S_PORT, audioBuffer, sizeof(audioBuffer), &bytesRead, I2S_TIMEOUT_TICKS);
    int samplesRead = bytesRead / sizeof(int32_t);

    int64_t sumSquares = 0;
    for (int i = 0; i < samplesRead && totalSamples < MAX_RECORD_SAMPLES; i++) {
      int32_t sample32 = micToPcm16(audioBuffer[i]);
      if (sample32 > 32767) sample32 = 32767;
      if (sample32 < -32768) sample32 = -32768;
      recordBuffer[totalSamples++] = (int16_t) sample32;
      sumSquares += (int64_t)sample32 * sample32;
    }

    float rms = samplesRead > 0 ? sqrt((float)sumSquares / samplesRead) : 0;
    unsigned long elapsed = millis() - recordStart;

    if (elapsed < VAD_MIN_RECORD_MS) {
      // Take the minimum, not the average, across this window: users
      // routinely start talking (e.g. the wake word) right as recording
      // begins, and an average would bake that word's energy straight into
      // the "ambient" floor, inflating the threshold for the rest of the
      // recording. A brief word almost always still has a quieter chunk
      // (onset/decay, or before/after it) inside a 400ms window, so the
      // minimum survives that far better than an average does.
      if (rms < noiseFloorMin) noiseFloorMin = rms;
      continue;
    }

    if (!calibrated) {
      // Prefer whichever estimate is lower: the pre-recording ambient EMA
      // is immune to this recording's own contamination (see its comment),
      // while this window's own live minimum still helps on the very first
      // recording after boot, before the EMA has seen much real audio.
      // Contamination only ever pushes the live minimum up, never down, so
      // taking the lower of the two is always safe.
      float noiseFloor = noiseFloorMin;
      if (g_ambientNoiseFloorReady && g_ambientNoiseFloor < noiseFloor) {
        noiseFloor = g_ambientNoiseFloor;
      }
      if (noiseFloor < VAD_NOISE_FLOOR_MIN) noiseFloor = VAD_NOISE_FLOOR_MIN;
      if (noiseFloor > VAD_NOISE_FLOOR_MAX) noiseFloor = VAD_NOISE_FLOOR_MAX;
      vadThreshold = max(noiseFloor * 2.5f, noiseFloor + (float) VAD_MARGIN);
      calibrated = true;
      client.print("VAD_CALIBRATED noiseFloor=");
      client.print(noiseFloor);
      client.print(" threshold=");
      client.println(vadThreshold);
    }

    if (rms >= vadThreshold) {
      // One loud 32ms chunk is a click or pop, not speech. Live test: a
      // noise burst right after the wake chime flipped speechDetected true,
      // the silence hold then counted down while the person was still about
      // to start talking, and the recording ended (2.5s) before their first
      // word. Speech now only "starts" after a sustained run of loud chunks
      // (an up/down count, so the natural on/off flicker of syllables still
      // accumulates while isolated bursts never reach the bar).
      loudRunChunks++;
      if (loudRunChunks >= VAD_SPEECH_START_CHUNKS) speechDetected = true;
      // Loud reading: lose some silence progress, but not all of it -- see
      // silenceProgressMs's declaration for why.
      silenceProgressMs -= VAD_CHUNK_MS * 2.0f;
      if (silenceProgressMs < 0) silenceProgressMs = 0;
    } else {
      if (loudRunChunks > 0) loudRunChunks--;
    }
    if (rms < vadThreshold && speechDetected) {
      // Only accumulate silence progress AFTER real speech has actually
      // been heard -- otherwise a quiet room before the user starts
      // talking would trigger an early stop with nothing captured at all.
      silenceProgressMs += VAD_CHUNK_MS;
      if (silenceProgressMs >= VAD_SILENCE_HOLD_MS) {
        break;
      }
    }
  }

  // Tell RockyBrain whether any real speech was heard. A recording with none
  // (room noise only) used to be transcribed anyway, and Whisper invents
  // words for noise ("You", "Thanks for watching") -- which Rocky then
  // answered, and the automatic follow-up listening chained into a runaway
  // conversation with nobody. RockyBrain skips transcription when speech=0.
  client.printf("VAD_RESULT speech=%d ms=%lu\n", speechDetected ? 1 : 0, (unsigned long)(millis() - recordStart));

  client.println("DONE LISTENING.");
  // Recording's done and about to go to RockyBrain -- show "Thinking" for
  // the transcribe+chat gap until an actual spoken reply's captions
  // (showCaptionChunks()) replace it.
  renderFaceWithCaption("Thinking");

  int totalBytes = totalSamples * sizeof(int16_t);

  // Dump raw PCM data over the WiFi control link with clear markers
  client.println("START_AUDIO");
  client.println(totalBytes); // byte count, as text
  client.write((uint8_t*)recordBuffer, totalBytes);
  client.println();
  client.println("END_AUDIO");
}

// ---- Wake-word self-test -------------------------------------------------
// Observed live: the on-device detector can silently die (exactly 0.000 on
// every frame, even with "Hey Rocky" played at it) and stay dead until the
// board is rebooted -- while everything else (Wi-Fi, mic, commands) looks
// healthy. Cause not pinned down, so this checks the symptom instead: every
// WW_SELFTEST_INTERVAL_MS of idle time, feed a known-good clip through the
// detector; if it no longer recognises it, reset the detector, retest, and
// reboot as a last resort (a reboot restored a working detector 8/8 times).
#define WW_SELFTEST_INTERVAL_MS 180000UL   // every 3 minutes while idle
#define WW_SELFTEST_MIN_PEAK 0.5f          // healthy clip scores ~0.9+
#define WW_SELFTEST_REBOOT 1               // reboot if the detector is dead and a reset doesn't revive it
unsigned long lastSelfTestMs = 0;
float lastSelfTestPeak = 0.0f;

float runWakeSelfTest() {
  lastSelfTestPeak = wakeword::selfTest(WW_SELFTEST_CLIP, WW_SELFTEST_LEN);
  return lastSelfTestPeak;
}

void selfTestAndHeal(WiFiClient &client) {
  lastSelfTestMs = millis();
  float peak = runWakeSelfTest();
  feedWatchdog();
  if (peak >= WW_SELFTEST_MIN_PEAK) return;

  Serial.printf("Wake-word self-test FAILED (peak=%.3f) -- resetting detector.\n", peak);
  wakeword::resetAll();
  peak = runWakeSelfTest();
  if (peak >= WW_SELFTEST_MIN_PEAK) {
    Serial.printf("Self-test recovered after reset (peak=%.3f).\n", peak);
    return;
  }
  Serial.printf("Self-test still failing (peak=%.3f).\n", peak);
#if WW_SELFTEST_REBOOT
  if (client && client.connected()) {
    client.println("WAKEWORD_SELFTEST_FAILED -- rebooting");
  }
  delay(200);
  esp_restart();
#endif
}

// Reinstalls the mic's I2S driver after repeated read timeouts (see
// tickWakeWord) -- the self-heal for a stalled mic DMA, without a reboot.
void recoverMic() {
  Serial.println("Mic I2S stalled -- reinstalling driver.");
  i2s_driver_uninstall(I2S_PORT);
  setupMic();
  wakeword::reset();
}

// Ticks the on-device wake-word detector on one short chunk of live mic
// audio (called from loop() whenever there's no pending command from
// RockyBrain to handle). On a detection, autonomously captures the
// follow-up utterance and pushes it to RockyBrain over the same protocol
// runListen() already uses for a Mac-requested "listen" -- RockyBrain no
// longer has to ask; the ESP32 decides for itself when to listen.
void tickWakeWord(WiFiClient &client) {
  static int micStallCount = 0;
  size_t bytesRead = 0;
  esp_err_t readErr = i2s_read(I2S_PORT, audioBuffer, sizeof(audioBuffer), &bytesRead, pdMS_TO_TICKS(500));
  if (readErr != ESP_OK || bytesRead == 0) {
    // A healthy mic delivers a 512-sample chunk every ~32ms, so a 500ms
    // timeout means the I2S DMA has stalled. Skip this tick (loop() keeps
    // servicing commands), and after 3 in a row reinstall the driver.
    if (++micStallCount >= 3) {
      micStallCount = 0;
      recoverMic();
    }
    return;
  }
  micStallCount = 0;
  int samplesRead = bytesRead / sizeof(int32_t);

  static int16_t chunk[BUFFER_LEN];
  int64_t sumSquares = 0;
  for (int i = 0; i < samplesRead; i++) {
    int32_t sample32 = micToPcm16(audioBuffer[i]);
    if (sample32 > 32767) sample32 = 32767;
    if (sample32 < -32768) sample32 = -32768;
    chunk[i] = (int16_t) sample32;
    sumSquares += (int64_t) sample32 * sample32;
  }

  // Roll this idle chunk's RMS into the ambient noise floor estimate (see
  // g_ambientNoiseFloor's declaration) before it's used for wake-word
  // detection -- this is genuine "nobody's about to speak yet" audio.
  if (samplesRead > 0) {
    float rms = sqrt((float) sumSquares / samplesRead);
    if (!g_ambientNoiseFloorReady) {
      g_ambientNoiseFloor = rms;
      g_ambientNoiseFloorReady = true;
    } else {
      g_ambientNoiseFloor = g_ambientNoiseFloor * 0.9f + rms * 0.1f;
    }
  }

  bool detected = false;
  wakeword::processAudio(chunk, samplesRead, &detected);

  if (detected) {
    Serial.println("Wake word detected -- listening...");
    wakeword::reset();
    setExpression("curious");  // wake up visually the instant we hear it --
                                // no need to wait on anything over WiFi for
                                // this, it's a local, instant state change
    playWakeChime();
    delay(300);  // let the chime's own acoustic tail die out before VAD
                 // calibration starts, so it doesn't get mistaken for room
                 // noise and inflate the silence threshold
    if (g_isDead) {
      // Only on a genuinely fresh wake from idle -- runStandPose() clears
      // g_isDead, so a follow-up question later in the same conversation
      // (which re-enters this branch via the plain "listen" command, not
      // here) never re-plays this ~1-2s animation. Deliberate latency
      // trade-off for the idle/wake theater the user asked for, paid only
      // once per wake instead of once per turn.
      runStandPose();
    }
    runListen(client);  // shows "Listening"/"Thinking" on the OLED itself
    client.println();
    client.println("Ready.");
    awaitingBrainResponse = true;
    awaitingBrainResponseSince = millis();
  }
}

// ================================
// Angle → PCA9685 pulse
// ================================
uint16_t angleToPulse(int angle) {
  angle = constrain(angle, 0, 180);
  return map(angle, 0, 180, SERVO_MIN, SERVO_MAX);
}

// ================================
// Servo output
// ================================
// Tracks each channel's last commanded angle, so a small interruption like
// runNodPose() can restore exactly where a leg actually was, regardless of
// which pose put it there -- rather than assuming "stand" and snapping to
// the wrong position if Rocky was mid-rest, mid-trick, etc.
int g_lastAngle[8] = {90, 90, 90, 90, 90, 90, 90, 90};

void setServoAngle(uint8_t channel, int angle) {
  if (channel < 8) {
    pwm.setPWM(channel, 0, angleToPulse(angle));
    g_lastAngle[channel] = angle;
    delay(frameDelay);
  }
}

// ================================
// REST
// ================================
void runRestPose() {
  Serial.println("REST"); setExpression("neutral");
  // Directly measured via the phone control page, not derived -- these are
  // NOT a uniform 90 like the reference assumes; every joint on this robot
  // sits at a different true center.
  setServoAngle(R1, 50); setServoAngle(R2, 105);
  setServoAngle(L1, 105); setServoAngle(L2, 50);
  setServoAngle(R4, 80); setServoAngle(R3, 60);
  setServoAngle(L3, 80); setServoAngle(L4, 60);
}

// ================================
// STAND
// Original Sesame positions
// ================================
void runStandPose() {
  Serial.println("STAND"); setExpression("curious"); g_isDead = false;
  // Directly measured via the phone control page for all 8 joints, not just
  // R3/L4 -- the reference project's 135/45/45/135/0/180/0/180 was never
  // actually verified against this robot's real geometry until tonight.
  setServoAngle(R1, 90); setServoAngle(R2, 60);
  setServoAngle(L1, 60); setServoAngle(L2, 90);
  setServoAngle(R4, 15); setServoAngle(R3, 130);
  setServoAngle(L3, 15); setServoAngle(L4, 130);
}

// ================================
// MOVEMENT + POSES
// Angles remapped from the Sesame reference through this robot's real
// calibration (see calibration_log.md) -- NOT the reference's raw numbers.
// ================================
void runWalkPose() {
  Serial.println("WALKFORWARD"); setExpression("curious");
  setServoAngle(R3, 95);
  setServoAngle(L3, 48);
  setServoAngle(R2, 115);
  setServoAngle(L1, 40);
  for (int i = 0; i < walkCycles; i++) {
    setServoAngle(R3, 95);
    setServoAngle(L3, 15);
    setServoAngle(L4, 95);
    setServoAngle(L2, 50);
    setServoAngle(R4, 15);
    setServoAngle(R1, 130);
    setServoAngle(R2, 60);
    setServoAngle(L1, 105);
    setServoAngle(R4, 48);
    setServoAngle(L4, 130);
    setServoAngle(R3, 130);
    setServoAngle(L3, 48);
    setServoAngle(R2, 105);
    setServoAngle(L1, 15);
    setServoAngle(L2, 90);
    setServoAngle(R1, 50);
  }
  runStandPose();
}

void runWalkBackward() {
  Serial.println("WALKBACKWARD"); setExpression("curious");
  for (int i = 0; i < walkCycles; i++) {
    setServoAngle(R3, 95);
    setServoAngle(L3, 15);
    setServoAngle(L4, 95);
    setServoAngle(L2, 90);
    setServoAngle(R4, 15);
    setServoAngle(R1, 50);
    setServoAngle(R2, 105);
    setServoAngle(L1, 15);
    setServoAngle(R4, 48);
    setServoAngle(L4, 130);
    setServoAngle(R3, 130);
    setServoAngle(L3, 48);
    setServoAngle(R2, 60);
    setServoAngle(L1, 105);
    setServoAngle(L2, 50);
    setServoAngle(R1, 130);
  }
  runStandPose();
}

void runTurnLeft() {
  Serial.println("TURNLEFT"); setExpression("curious");
  for (int i = 0; i < walkCycles; i++) {
    setServoAngle(R3, 95);
    setServoAngle(L4, 95);
    setServoAngle(R1, 130);
    setServoAngle(L2, 130);
    setServoAngle(R3, 130);
    setServoAngle(L4, 130);
    setServoAngle(R1, 90);
    setServoAngle(L2, 90);
    setServoAngle(R4, 48);
    setServoAngle(L3, 48);
    setServoAngle(R2, 105);
    setServoAngle(L1, 105);
    setServoAngle(R4, 15);
    setServoAngle(L3, 15);
    setServoAngle(R2, 60);
    setServoAngle(L1, 60);
  }
  runStandPose();
}

void runTurnRight() {
  Serial.println("TURNRIGHT"); setExpression("curious");
  for (int i = 0; i < walkCycles; i++) {
    setServoAngle(R4, 48);
    setServoAngle(L3, 48);
    setServoAngle(R2, 15);
    setServoAngle(L1, 15);
    setServoAngle(R4, 15);
    setServoAngle(L3, 15);
    setServoAngle(R2, 60);
    setServoAngle(L1, 60);
    setServoAngle(R3, 95);
    setServoAngle(L4, 95);
    setServoAngle(R1, 50);
    setServoAngle(L2, 50);
    setServoAngle(R3, 130);
    setServoAngle(L4, 130);
    setServoAngle(R1, 90);
    setServoAngle(L2, 90);
  }
  runStandPose();
}

void runWavePose() {
  Serial.println("WAVE"); setExpression("happy");
  runStandPose();
  delay(200);
  setServoAngle(R4, 73);
  setServoAngle(L3, 145);
  setServoAngle(L2, 50);
  setServoAngle(R1, 59);
  delay(200);
  setServoAngle(L3, 145);
  delay(300);
  for (int i = 0; i < 4; i++) {
    setServoAngle(L3, 145);
    delay(300);
    setServoAngle(L3, 87);
    delay(300);
  }
  runStandPose();
}

void runDancePose() {
  Serial.println("DANCE"); setExpression("happy");
  setServoAngle(R1, 50);
  setServoAngle(R2, 105);
  setServoAngle(L1, 105);
  setServoAngle(L2, 50);
  setServoAngle(R4, 131);
  setServoAngle(R3, 114);
  setServoAngle(L3, 22);
  setServoAngle(L4, 0);
  delay(300);
  for (int i = 0; i < 5; i++) {
    setServoAngle(R4, 98);
    setServoAngle(R3, 79);
    setServoAngle(L3, 22);
    setServoAngle(L4, 0);
    delay(300);
    setServoAngle(R4, 131);
    setServoAngle(R3, 114);
    setServoAngle(L3, 62);
    setServoAngle(L4, 41);
    delay(300);
  }
  runStandPose();
}

void runSwimPose() {
  Serial.println("SWIM"); setExpression("surprised");
  setServoAngle(R1, 50);
  setServoAngle(R2, 105);
  setServoAngle(L1, 105);
  setServoAngle(L2, 50);
  setServoAngle(R4, 80);
  setServoAngle(R3, 60);
  setServoAngle(L3, 80);
  setServoAngle(L4, 60);
  for (int i = 0; i < 4; i++) {
    setServoAngle(R1, 90);
    setServoAngle(R2, 60);
    setServoAngle(L1, 60);
    setServoAngle(L2, 90);
    delay(400);
    setServoAngle(R1, 50);
    setServoAngle(R2, 105);
    setServoAngle(L1, 105);
    setServoAngle(L2, 50);
    delay(400);
  }
  runStandPose();
}

void runPointPose() {
  Serial.println("POINT"); setExpression("curious");
  setServoAngle(L2, 50);
  setServoAngle(R1, 90);
  setServoAngle(R2, 115);
  setServoAngle(L4, 130);
  setServoAngle(L1, 40);
  setServoAngle(L3, 120);
  setServoAngle(R4, 73);
  setServoAngle(R3, 122);
  delay(2000);
  runStandPose();
}

void runPushupPose() {
  Serial.println("PUSHUP"); setExpression("surprised");
  runStandPose();
  delay(200);
  setServoAngle(L1, 15);
  setServoAngle(R1, 130);
  setServoAngle(L3, 80);
  setServoAngle(R3, 60);
  delay(500);
  for (int i = 0; i < 4; i++) {
    setServoAngle(L3, 15);
    setServoAngle(R3, 130);
    delay(600);
    setServoAngle(L3, 80);
    setServoAngle(R3, 60);
    delay(500);
  }
  runStandPose();
}

void runBowPose() {
  Serial.println("BOW"); setExpression("happy");
  runStandPose();
  delay(200);
  setServoAngle(L1, 15);
  setServoAngle(R1, 130);
  setServoAngle(L3, 15);
  setServoAngle(R3, 130);
  setServoAngle(L2, 130);
  setServoAngle(R2, 15);
  setServoAngle(R4, 15);
  setServoAngle(L4, 130);
  delay(600);
  setServoAngle(L3, 80);
  setServoAngle(R3, 60);
  delay(3000);
  runStandPose();
}

void runCutePose() {
  Serial.println("CUTE"); setExpression("happy");
  runStandPose();
  delay(200);
  setServoAngle(L2, 112);
  setServoAngle(R2, 35);
  setServoAngle(R4, 145);
  setServoAngle(L4, 0);
  setServoAngle(L1, 15);
  setServoAngle(R1, 130);
  setServoAngle(L3, 145);
  setServoAngle(R3, 0);
  delay(200);
  for (int i = 0; i < 5; i++) {
    setServoAngle(R4, 145);
    setServoAngle(L4, 25);
    delay(300);
    setServoAngle(R4, 112);
    setServoAngle(L4, 0);
    delay(300);
  }
  runStandPose();
}

void runFreakyPose() {
  Serial.println("FREAKY"); setExpression("surprised");
  runStandPose();
  delay(200);
  setServoAngle(L1, 15);
  setServoAngle(R1, 130);
  setServoAngle(L2, 130);
  setServoAngle(R2, 15);
  setServoAngle(R4, 80);
  setServoAngle(R3, 0);
  delay(200);
  for (int i = 0; i < 3; i++) {
    setServoAngle(R3, 9);
    delay(400);
    setServoAngle(R3, 0);
    delay(400);
  }
  runStandPose();
}

void runWormPose() {
  Serial.println("WORM"); setExpression("surprised");
  runStandPose();
  delay(200);
  setServoAngle(R1, 130);
  setServoAngle(R2, 15);
  setServoAngle(L1, 15);
  setServoAngle(L2, 130);
  setServoAngle(R4, 80);
  setServoAngle(R3, 60);
  setServoAngle(L3, 80);
  setServoAngle(L4, 60);
  delay(200);
  for (int i = 0; i < 5; i++) {
    setServoAngle(R3, 25);
    setServoAngle(L3, 112);
    setServoAngle(R4, 48);
    setServoAngle(L4, 95);
    delay(300);
    setServoAngle(R3, 95);
    setServoAngle(L3, 48);
    setServoAngle(R4, 112);
    setServoAngle(L4, 25);
    delay(300);
  }
  runStandPose();
}

void runShakePose() {
  Serial.println("SHAKE"); setExpression("curious");
  runStandPose();
  delay(200);
  setServoAngle(R1, 90);
  setServoAngle(L1, 60);
  setServoAngle(L3, 80);
  setServoAngle(R3, 60);
  setServoAngle(L2, 50);
  setServoAngle(R2, 105);
  delay(200);
  for (int i = 0; i < 5; i++) {
    setServoAngle(R4, 48);
    setServoAngle(L4, 95);
    delay(300);
    setServoAngle(R4, 15);
    setServoAngle(L4, 130);
    delay(300);
  }
  runStandPose();
}

void runNodPose() {
  Serial.println("NOD"); setExpression("happy");
  int r1 = g_lastAngle[R1], l2 = g_lastAngle[L2];  // restore exactly where
                                                    // these were, whatever
                                                    // pose Rocky was in
  setServoAngle(R1, 65);
  setServoAngle(L2, 65);
  delay(120);
  setServoAngle(R1, r1);
  setServoAngle(L2, l2);
}

void runShrugPose() {
  Serial.println("SHRUG"); setExpression("sad");
  runStandPose();
  delay(200);
  setServoAngle(R3, 60);
  setServoAngle(R4, 80);
  setServoAngle(L3, 80);
  setServoAngle(L4, 60);
  delay(1000);
  setServoAngle(R3, 0);
  setServoAngle(R4, 145);
  setServoAngle(L3, 145);
  setServoAngle(L4, 0);
  delay(1500);
  runStandPose();
}

void runDeadPose() {
  Serial.println("DEAD"); setExpression("sleepy");
  runStandPose();  // known-safe transition baseline, same as every other pose --
                    // but this clears g_isDead, so it must be re-set to true
                    // below, AFTER this call, not before it (setting it here
                    // first was the actual bug: runStandPose() immediately
                    // wiped it back to false, so every "dead" command silently
                    // left g_isDead false despite the robot visibly resting --
                    // tickWakeWord() then saw g_isDead==false on the next "Hey
                    // Rocky" and skipped standing entirely, confirmed as the
                    // root cause of wake-word not standing up from rest).
  delay(200);
  setServoAngle(R3, 60);
  setServoAngle(R4, 80);
  setServoAngle(L3, 80);
  setServoAngle(L4, 60);
  g_isDead = true;
}

void runCrabPose() {
  Serial.println("CRAB"); setExpression("surprised");
  runStandPose();
  delay(200);
  setServoAngle(R1, 50);
  setServoAngle(R2, 105);
  setServoAngle(L1, 105);
  setServoAngle(L2, 50);
  setServoAngle(R4, 15);
  setServoAngle(R3, 130);
  setServoAngle(L3, 48);
  setServoAngle(L4, 95);
  for (int i = 0; i < 5; i++) {
    setServoAngle(R4, 48);
    setServoAngle(R3, 95);
    setServoAngle(L3, 15);
    setServoAngle(L4, 130);
    delay(300);
    setServoAngle(R4, 15);
    setServoAngle(R3, 130);
    setServoAngle(L3, 48);
    setServoAngle(L4, 95);
    delay(300);
  }
  runStandPose();
}

// ================================
// Serial command system
// ================================
String buildStatusLine() {
  // Real bus probes, not assumptions: the boot banner used to print
  // "PCA9685 ready." unconditionally even when the servo driver was missing
  // from the I2C bus entirely.
  auto i2cPresent = [](uint8_t addr) {
    Wire.beginTransmission(addr);
    return Wire.endTransmission() == 0;
  };
  char buf[440];
  snprintf(buf, sizeof(buf),
           "STATUS fw=%s %s uptime_s=%lu heap=%u psram=%u rssi=%d ip=%s volume=%d reset=%d wakeword=%d wifi=%d oled=%d pca9685=%d oled_bus=%d batt_v=%.2f batt_pct=%d",
           __DATE__, __TIME__, millis() / 1000, (unsigned) ESP.getFreeHeap(), (unsigned) ESP.getFreePsram(),
           (int) WiFi.RSSI(), WiFi.localIP().toString().c_str(), g_volumePct, (int) esp_reset_reason(),
           (int)(lastSelfTestPeak * 1000), (int) wifiConnected, (int) g_oledReady,
           (int) i2cPresent(0x40), (int) i2cPresent(0x3C), readBatteryVolts(), batteryPercent());
  return String(buf);
}

// ~1 second of mic levels after the high-pass filter: what the room sounds
// like to the wake word and the silence detector.
String micLevelReport() {
  const int chunks = 30;
  float rmsList[chunks];
  int n = 0, clipped = 0;
  for (int c = 0; c < chunks; c++) {
    size_t bytesRead = 0;
    if (i2s_read(I2S_PORT, audioBuffer, sizeof(audioBuffer), &bytesRead, pdMS_TO_TICKS(500)) != ESP_OK || bytesRead == 0) continue;
    int samples = bytesRead / sizeof(int32_t);
    int64_t sumSquares = 0;
    for (int i = 0; i < samples; i++) {
      int16_t v = micToPcm16(audioBuffer[i]);
      if (v >= 32767 || v <= -32768) clipped++;
      sumSquares += (int64_t) v * v;
    }
    if (samples > 0) rmsList[n++] = sqrtf((float) sumSquares / samples);
  }
  if (n == 0) return "chunks=0 (mic delivered no data)";
  for (int i = 1; i < n; i++) {           // tiny insertion sort for percentiles
    float k = rmsList[i]; int j = i - 1;
    while (j >= 0 && rmsList[j] > k) { rmsList[j + 1] = rmsList[j]; j--; }
    rmsList[j + 1] = k;
  }
  char buf[160];
  snprintf(buf, sizeof(buf), "chunks=%d median=%.0f p95=%.0f max=%.0f clipped=%d", n, rmsList[n / 2], rmsList[(n * 95) / 100], rmsList[n - 1], clipped);
  return String(buf);
}

void executeCommand(String command, WiFiClient &client) {

  // Checked before trim/lowercase so caption text keeps its original casing.
  // Format: say:<durationMs>:<text> -- durationMs is parsed out for protocol
  // compatibility with the Python side but not used for pacing (chunks use a
  // fixed hold time instead). Falls back gracefully if it's missing/malformed,
  // so a bare say:<text> (no duration) still works.
  if (command.startsWith("say:")) {
    String rest = command.substring(4);
    int colonIdx = rest.indexOf(':');
    String caption = rest;

    if (colonIdx > 0) {
      String durStr = rest.substring(0, colonIdx);
      bool isNumeric = durStr.length() > 0;
      for (unsigned int i = 0; i < durStr.length(); i++) {
        if (!isDigit(durStr.charAt(i))) { isNumeric = false; break; }
      }
      if (isNumeric) {
        caption = rest.substring(colonIdx + 1);
      }
    }

    caption.trim();
    showCaptionChunks(caption);
    return;
  }

  // Format: speak:<byteCount>:<text>\n immediately followed by exactly
  // byteCount raw bytes of mono 16-bit PCM audio at SPK_SAMPLE_RATE -- this
  // is how RockyBrain routes actual TTS speech to the onboard speaker,
  // captions synced to real playback progress (see playStreamedSpeech).
  command.trim();

  if (command.startsWith("volume")) {
    // "volume" queries; "volume:<0-100>" sets and saves.
    int colon = command.indexOf(':');
    if (colon > 0) {
      setVolumePct(command.substring(colon + 1).toInt(), true);
    }
    client.print("VOLUME ");
    client.println(g_volumePct);
    return;
  }

  if (command == "status") {
    client.println(buildStatusLine());
    return;
  }

  if (command.startsWith("wwfeed:")) {
    size_t n = (size_t) command.substring(7).toInt();
    int16_t *buf = (n > 0 && n <= 400000) ? (int16_t *) ps_malloc(n * sizeof(int16_t)) : nullptr;
    if (buf == nullptr) { client.println("WWFEED error: bad length or out of memory."); return; }
    client.setTimeout(8000);
    size_t got = client.readBytes((uint8_t *) buf, n * sizeof(int16_t));
    if (got != n * sizeof(int16_t)) {
      client.printf("WWFEED error: got %u of %u bytes.\n", (unsigned) got, (unsigned)(n * sizeof(int16_t)));
    } else {
      float peak = wakeword::selfTest(buf, n);
      client.print("WWFEED peak: ");
      client.println(peak, 3);
    }
    free(buf);
    return;
  }

  if (command.startsWith("speak:")) {
    String rest = command.substring(6);
    int colonIdx = rest.indexOf(':');
    if (colonIdx > 0) {
      String byteCountStr = rest.substring(0, colonIdx);
      String caption = rest.substring(colonIdx + 1);
      caption.trim();
      int totalBytes = byteCountStr.toInt();
      playStreamedSpeech(caption, totalBytes, client);
      wakeWordCooldownUntil = millis() + WAKE_COOLDOWN_MS;
    } else {
      client.println("Malformed speak command.");
    }
    return;
  }

  command.trim();
  command.toLowerCase();

  if (command.startsWith("servo:")) {
    // servo:<channel 0-7>:<angle 0-180> -- moves ONE joint, for safely
    // checking each one against the real mechanism before ever using a
    // full pose that moves all 8 at once.
    int c1 = command.indexOf(':');
    int c2 = command.indexOf(':', c1 + 1);
    if (c2 > 0) {
      int channel = command.substring(c1 + 1, c2).toInt();
      int angle = command.substring(c2 + 1).toInt();
      if (channel >= 0 && channel <= 7 && angle >= 0 && angle <= 180) {
        setServoAngle((uint8_t) channel, angle);
        client.printf("SERVO channel=%d angle=%d\n", channel, angle);
      } else {
        client.println("ERROR: channel must be 0-7, angle 0-180.");
      }
    } else {
      client.println("ERROR: usage servo:<channel>:<angle>");
    }
  }
  else if (command == "stand") {
    runStandPose();
  }
  else if (command == "rest") {
    runRestPose();
  }
  else if (command == "forward") {
    runWalkPose();
  }
  else if (command == "backward") {
    runWalkBackward();
  }
  else if (command == "left") {
    runTurnLeft();
  }
  else if (command == "right") {
    runTurnRight();
  }
  else if (command == "wave") { runWavePose(); }
  else if (command == "dance") { runDancePose(); }
  else if (command == "swim") { runSwimPose(); }
  else if (command == "point") { runPointPose(); }
  else if (command == "pushup") { runPushupPose(); }
  else if (command == "bow") { runBowPose(); }
  else if (command == "cute") { runCutePose(); }
  else if (command == "freaky") { runFreakyPose(); }
  else if (command == "worm") { runWormPose(); }
  else if (command == "shake") { runShakePose(); }
  else if (command == "shrug") { runShrugPose(); }
  else if (command == "nod") { runNodPose(); }
  else if (command == "dead") { runDeadPose(); }
  else if (command == "crab") { runCrabPose(); }
  else if (command == "mictest") {
    runMicTest(client);
  }
  else if (command == "waketest") {
    runWakeWordTest(client, 15000);
  }
  else if (command == "speakertest") {
    runSpeakerTest();
  }
  else if (command == "listen") {
    setExpression("curious");  // in case the last thing shown was a mode's
                                // reaction face (e.g. sarcastic's half-shut
                                // "sleepy" look) or idle's closed eyes --
                                // listening should always look awake
    playFollowUpChime();
    delay(150);  // let the blip's tail decay before VAD noise-floor calibration starts
    runListen(client);
  }
  else if (command == "face neutral") {
    setExpression("neutral");
  }
  else if (command == "face happy") {
    setExpression("happy");
  }
  else if (command == "face sad") {
    setExpression("sad");
  }
  else if (command == "face curious") {
    setExpression("curious");
  }
  else if (command == "face sleepy") {
    setExpression("sleepy");
  }
  else if (command == "face surprised") {
    setExpression("surprised");
  }
  else if (command == "blink") {
    blink();
  }
  else if (command == "selftest") {
    float peak = runWakeSelfTest();
    client.print("Wake-word self-test peak: ");
    client.println(peak, 3);
    client.println(peak >= WW_SELFTEST_MIN_PEAK ? "SELFTEST PASS" : "SELFTEST FAIL");
  }
  else if (command == "wwkill") {
    // Test-only: reproduces the observed "dead detector" symptom (every
    // score exactly 0.000, and reset() doesn't help) by wrecking the input
    // scaling, so the self-test + self-heal path can be verified end to end.
    wakeword::g_input->params.scale = 1.0e9f;
    client.println("Detector deliberately broken (test). Run selftest / selfheal.");
  }
  else if (command == "selfheal") {
    selfTestAndHeal(client);
    client.println("selfheal: detector healthy or recovered (no reboot needed).");
  }
  else if (command == "hangtest") {
    client.println("Hanging on purpose -- the watchdog should reboot me within ~40s.");
    while (true) { delay(1000); }
  }
  else {
    client.println("Unknown command.");
    client.println("Use: stand, rest, forward, backward, left, right, mictest, speakertest, listen,");
    client.println("     face neutral/happy/sad/curious/sleepy/surprised, blink, selftest, selfheal, wwkill, hangtest, volume[:0-100], status,");
    client.println("     say:<ms>:<text>, speak:<byteCount>:<text>+PCM");
  }
}

// ================================
// SETUP
// ================================
void scanI2C() {
  Serial.println();
  Serial.println("Scanning I2C bus...");
  int found = 0;
  for (uint8_t addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    uint8_t error = Wire.endTransmission();
    if (error == 0) {
      Serial.print("  Found device at 0x");
      if (addr < 16) Serial.print("0");
      Serial.println(addr, HEX);
      found++;
    }
  }
  if (found == 0) {
    Serial.println("  No I2C devices found!");
  } else {
    Serial.print("  ");
    Serial.print(found);
    Serial.println(" device(s) found.");
  }
  Serial.println();
}

void setup() {

  Serial.begin(921600);
  delay(1000);

  Wire.begin(SDA_PIN, SCL_PIN);
  scanI2C();

  pwm.begin();
  pwm.setOscillatorFrequency(27000000);
  pwm.setPWMFreq(50);

  bool oledReady = display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR);
  g_oledReady = oledReady;
  if (!oledReady) {
    Serial.println("OLED init FAILED -- check wiring/address.");
  }

  connectWiFi();

  if (wifiConnected) {
    // Over-the-air updates: once the enclosure is closed there must be a way
    // to fix or improve the firmware without opening it. Flash with
    // `pio run -e ota -t upload` (see platformio.ini). mDNS also makes the
    // board reachable as rocky-body.local regardless of its DHCP address.
    ArduinoOTA.setHostname("rocky-body");
    ArduinoOTA.setPassword(OTA_PASSWORD);
    ArduinoOTA.onStart([]() { Serial.println("OTA update starting..."); });
    ArduinoOTA.onProgress([](unsigned int done, unsigned int total) { feedWatchdog(); });
    ArduinoOTA.onEnd([]() { Serial.println("OTA update done -- rebooting."); });
    ArduinoOTA.onError([](ota_error_t e) { Serial.printf("OTA error %u\n", (unsigned) e); });
    ArduinoOTA.begin();
    MDNS.addService("rocky", "tcp", CONTROL_PORT);
    setupHttpOta();

    controlServer.begin();
    Serial.print("Control link listening on ");
    Serial.print(WiFi.localIP());
    Serial.print(":");
    Serial.println(CONTROL_PORT);
  } else {
    Serial.println("No WiFi -- control link unavailable (RockyBrain needs WiFi now, not USB).");
  }

  if (oledReady) {
    setExpression("sleepy");  // idle/resting look (closed eyes) -- also draws the now-live status bar
  }

  prefs.begin("rocky", true);
  setVolumePct(prefs.getInt("vol", 100), false);
  prefs.end();

  setupMic();
  setupSpeaker();
  wakeword::init();
  Serial.printf("Wake-word boot self-test peak: %.3f\n", runWakeSelfTest());
  lastSelfTestMs = millis();

  delay(500);

  Serial.println();
  Serial.println("==============================");
  Serial.println("       ROCKY ONLINE");
  Serial.println("==============================");
  Serial.println("PCA9685 ready.");
  Serial.print("OLED ");
  Serial.println(oledReady ? "ready." : "NOT ready.");
  Serial.print("WiFi ");
  Serial.println(wifiConnected ? "connected." : "NOT connected.");
  Serial.println("Mic ready.");
  Serial.println("Speaker ready.");
  Serial.println("Waiting for RockyBrain to connect over WiFi...");
  Serial.println();

  // Last-resort backstop: if loop() ever stops making progress for 40s
  // (a hang the finite I2S timeouts above didn't catch), reboot instead of
  // sitting frozen forever. Armed only after the slow one-time setup (Wi-Fi
  // connect etc.); long legitimate operations (speech playback/receive) feed
  // it themselves. RockyBrain's keepalive + reconnect handles the reboot.
  startWatchdog();

  // Deliberately NOT auto-moving to a stand pose here anymore: this used to
  // drive all 8 servos on every boot, including an unattended watchdog/
  // brownout/OTA reboot -- confirmed today those happen often enough during
  // normal operation that a silent full-pose move on every one of them is a
  // real safety problem now that servos are actually connected. Movement
  // only ever happens on an explicit command from here on.
}

// ================================
// LOOP
// ================================
unsigned long lastStatusRefresh = 0;

void loop() {
  feedWatchdog();
  ArduinoOTA.handle();
  otaServer.handleClient();

  // Always check for a new incoming connection first, every loop pass --
  // never gated behind "do we think we already have one". WiFiClient's
  // connected() can keep reporting true for a peer that is actually gone
  // (RockyBrain crashed, its process was killed, its socket half-closed
  // without a clean FIN) -- the old code only called controlServer.
  // available() while controlClient looked dead, so a falsely-alive stale
  // client permanently blocked accept() from ever running again. The
  // listen backlog is small; once a new connection attempt landed there
  // unaccepted, every later attempt from RockyBrain just queued behind it
  // and timed out at the TCP level -- even though the board was otherwise
  // completely healthy (HTTP on a separate port kept working the whole
  // time, confirmed live). Only a full reboot ever cleared it, matching
  // this session's recurring "have to power-cycle the board" pattern.
  // There is only ever one legitimate RockyBrain client, so a new
  // connection always wins outright -- drop whatever we had and take it.
  WiFiClient incoming = controlServer.available();
  if (incoming) {
    if (controlClient && controlClient.connected()) {
      controlClient.stop();
    }
    controlClient = incoming;
    Serial.println("RockyBrain connected.");
  }

  if (awaitingBrainResponse && millis() - awaitingBrainResponseSince > AWAITING_BRAIN_TIMEOUT_MS) {
    Serial.println("No response from RockyBrain within timeout -- resuming wake-word listening.");
    awaitingBrainResponse = false;
    // Whatever RockyBrain's lost closing command was going to do (most
    // likely "dead", the normal end-of-turn pose), it never landed --
    // don't just resume listening with the robot potentially still
    // standing and the face stuck on "curious" from the wake chime. Settle
    // back to a known, honest idle state instead of leaving it ambiguous
    // until the next successful conversation happens to fix it.
    runDeadPose();
  }

  if (controlClient && controlClient.connected() && controlClient.available()) {
    String command = controlClient.readStringUntil('\n');
    awaitingBrainResponse = false;  // RockyBrain is actively responding to
                                     // this turn again -- see tickWakeWord()
    executeCommand(command, controlClient);
    controlClient.println();
    controlClient.println("Ready.");
  } else if (!awaitingBrainResponse && millis() >= wakeWordCooldownUntil) {
    // Idle: no pending command from RockyBrain, we're not waiting on it to
    // finish responding to a previous wake-up, and we're past the
    // post-speech cooldown -- so listen for the wake word ourselves instead
    // of doing nothing. This always runs after the "is there a command
    // waiting?" check above, so an incoming command is never delayed
    // behind a wake-word tick.
    if (millis() - lastSelfTestMs > WW_SELFTEST_INTERVAL_MS) {
      selfTestAndHeal(controlClient);
    }
    tickWakeWord(controlClient);
  }

  if (millis() - lastStatusRefresh > 30000) {
    renderFace();
    lastStatusRefresh = millis();
  }
}

