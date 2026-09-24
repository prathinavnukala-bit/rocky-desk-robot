#!/usr/bin/env python3
"""
Rocky system diagnostics. One command, every layer, evidence for each verdict.

    python3 diagnose.py            # fast checks (~10s), safe to run while Rocky is running
    python3 diagnose.py --deep     # also loads/times the speech, chat and voice models (~30s)
    python3 diagnose.py --soak 30  # watch everything for 30 minutes and report any failure or reboot

Reads the board over its HTTP health endpoints (/status, /selftest, /mic) so it
never fights brain.py for the board's single control connection. Exit code is
non-zero when anything FAILS, so it can be scripted.

Verdicts: PASS = works; WARN = works but degraded / needs attention; FAIL = broken.
"""
import argparse
import os
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import netfind  # noqa: E402

BODY_NAME, CAM_NAME = "rocky-body.local", "rocky-cam.local"
results = []


def record(area, name, status, detail=""):
    results.append((area, name, status, detail))
    mark = {"PASS": "\033[32mPASS\033[0m", "WARN": "\033[33mWARN\033[0m", "FAIL": "\033[31mFAIL\033[0m", "INFO": "INFO"}[status]
    print(f"  [{mark}] {area:9s} {name:34s} {detail}", flush=True)


def http_get(url, timeout=6):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode(errors="replace").strip(), r.status


def parse_kv(line):
    return dict(re.findall(r"(\w+)=(\S+)", line))


# ----------------------------------------------------------------- checks
def check_network():
    body_ip = netfind.resolve_local(BODY_NAME, 3)
    if body_ip:
        record("network", "body board via rocky-body.local", "PASS", body_ip)
    else:
        body_ip = netfind.find_rocky_body("192.168.0.", 8266)
        record("network", "body board", "WARN" if body_ip else "FAIL",
               f"{body_ip} (found by scan; .local name did not resolve)" if body_ip else "not found on the network")
    cam_ip = netfind.resolve_local(CAM_NAME, 2)
    if cam_ip:
        record("network", "camera via rocky-cam.local", "PASS", cam_ip)
    else:
        cam_ip = netfind.find_rocky_camera("192.168.0.")
        record("network", "camera board", "WARN" if cam_ip else "FAIL",
               f"{cam_ip} (found by scan; camera firmware has no .local name yet)" if cam_ip else "not found on the network")
    return body_ip, cam_ip


def check_body(ip):
    if not ip:
        return None
    try:
        t = time.time()
        line, _ = http_get(f"http://{ip}:8080/status")
        ms = (time.time() - t) * 1000
    except Exception as e:
        record("body", "health endpoint (:8080/status)", "FAIL",
               f"no answer ({e}). Firmware without HTTP diagnostics, or the loop is frozen.")
        return None
    kv = parse_kv(line)
    record("body", "responds over Wi-Fi", "PASS" if ms < 400 else "WARN", f"{ms:.0f} ms round trip")
    m = re.search(r"fw=(.+?) uptime_s", line)
    record("body", "firmware build", "INFO", m.group(1) if m else "?")
    up = int(kv.get("uptime_s", 0))
    record("body", "uptime", "WARN" if up < 60 else "PASS",
           f"{up // 3600}h {up % 3600 // 60}m {up % 60}s" + (" (just rebooted)" if up < 60 else ""))
    reset = int(kv.get("reset", 0))
    reasons = {1: "power-on", 3: "software reset", 4: "panic/crash", 5: "interrupt watchdog", 6: "task watchdog",
               7: "other watchdog", 8: "deep sleep", 9: "brownout (power dip!)", 10: "SDIO"}
    record("body", "last reset reason", "FAIL" if reset in (4, 5, 6, 7) else "WARN" if reset == 9 else "INFO",
           f"{reasons.get(reset, reset)}")
    heap, psram = int(kv.get("heap", 0)), int(kv.get("psram", 0))
    record("body", "free memory", "PASS" if heap > 60000 and psram > 2_000_000 else "WARN",
           f"heap {heap // 1024} KB, PSRAM {psram // 1024} KB")
    rssi = int(kv.get("rssi", -100))
    record("body", "Wi-Fi signal", "PASS" if rssi > -70 else "WARN" if rssi > -80 else "FAIL", f"{rssi} dBm")
    record("body", "OLED display", "PASS" if kv.get("oled") == "1" and kv.get("oled_bus") == "1" else "FAIL",
           f"driver ready={kv.get('oled')}, on I2C bus={kv.get('oled_bus')}")
    record("body", "servo driver (PCA9685) on I2C bus", "PASS" if kv.get("pca9685") == "1" else "FAIL",
           "detected at 0x40" if kv.get("pca9685") == "1" else
           "NOT detected -- check its power (VCC/GND), the SDA->GPIO8 / SCL->GPIO9 wires and V+ servo supply; no servo can move until this passes")
    vol = kv.get("volume")
    record("body", "speaker volume setting", "INFO", f"{vol}%")
    batt = float(kv.get("batt_v", -1))
    record("body", "battery monitor", "INFO" if batt < 0 else ("PASS" if batt > 7.0 else "WARN" if batt > 6.4 else "FAIL"),
           "not wired yet (needs a voltage divider on an ADC pin)" if batt < 0 else f"{batt:.2f} V")
    return kv


def check_body_active(ip):
    if not ip:
        return
    try:
        t = time.time()
        line, _ = http_get(f"http://{ip}:8080/selftest", timeout=15)
        peak = float(line.split("=")[1])
        record("wakeword", "detector self-test", "PASS" if peak >= 0.5 else "FAIL",
               f"peak {peak:.3f} on the built-in 'Hey Rocky' clip (healthy ~0.92; below 0.5 = dead detector, board self-reboots)")
    except Exception as e:
        record("wakeword", "detector self-test", "FAIL", str(e))
    try:
        line, _ = http_get(f"http://{ip}:8080/mic", timeout=10)
        kv = parse_kv(line)
        median, p95, clipped = float(kv.get("median", -1)), float(kv.get("p95", -1)), int(kv.get("clipped", 0))
        if kv.get("chunks", "0") == "0":
            record("mic", "microphone data", "FAIL", "the mic delivered no audio")
        else:
            ok = 30 <= median <= 1500 and clipped == 0
            record("mic", "room level after high-pass filter", "PASS" if ok else "WARN",
                   f"median {median:.0f}, p95 {p95:.0f}, clipped {clipped} (healthy quiet room: ~300-700, no clipping; "
                   f"3000+ means low-frequency drift or a noisy supply)")
            record("mic", "silence detector margin", "PASS" if median * 2.5 < 1500 else "WARN",
                   f"threshold would be ~{max(median * 2.5, median + 600):.0f}; speech is typically 1500-3400")
    except Exception as e:
        record("mic", "mic level probe", "FAIL", str(e))


def check_camera(ip):
    if not ip:
        return
    try:
        line, _ = http_get(f"http://{ip}/status", timeout=6)
        kv = parse_kv(line)
        record("camera", "status endpoint", "PASS", f"uptime {kv.get('uptime_s', '?')}s, captures {kv.get('captures', '?')}, "
               f"failures {kv.get('failures', '?')}, rssi {kv.get('rssi', '?')}")
    except Exception:
        record("camera", "status endpoint", "WARN", "old camera firmware (no /status) -- flash the new one for self-healing + OTA")
    try:
        t = time.time()
        with urllib.request.urlopen(f"http://{ip}/jpg", timeout=10) as r:
            data = r.read()
        ms = (time.time() - t) * 1000
        record("camera", "captures a frame", "PASS" if len(data) > 3000 else "FAIL", f"{len(data)} bytes in {ms:.0f} ms")
        with open("/tmp/diag_frame.jpg", "wb") as f:
            f.write(data)
        try:
            import face_id
            import cv2
            im = cv2.imread("/tmp/diag_frame.jpg")
            faces = face_id._detections(im) if im is not None else []
            record("camera", "face detector runs on the frame", "PASS", f"{len(faces)} face(s) in view")
            enrolled = face_id._load_faces()
            record("people", "enrolled people", "PASS" if enrolled else "WARN",
                   ", ".join(f"{n} ({len(face_id._embeddings_of(e))} samples)" for n, e in enrolled.items()) or "nobody enrolled yet")
        except Exception as e:
            record("camera", "face pipeline", "WARN", f"could not run: {e}")
    except Exception as e:
        record("camera", "captures a frame", "FAIL", f"{e}")


def check_mac():
    free = shutil.disk_usage("/System/Volumes/Data").free / 1e9
    record("mac", "free disk space", "PASS" if free > 10 else "WARN" if free > 4 else "FAIL", f"{free:.1f} GB")
    try:
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
        used = float(re.search(r"used = ([\d.]+)M", out).group(1)) if "used" in out else 0
        record("mac", "swap in use", "PASS" if used < 1500 else "WARN", f"{used:.0f} MB (heavy swapping makes model calls slow)")
    except Exception:
        pass
    procs = subprocess.run(["pgrep", "-fl", "brain.py"], capture_output=True, text=True).stdout.strip()
    running = [l for l in procs.splitlines() if "python" in l.lower() and "pgrep" not in l]
    record("brain", "brain.py running", "PASS" if running else "WARN", "yes" if running else "not running (start it, or use rocky_ctl.sh)")
    try:
        out, _ = http_get("http://127.0.0.1:11434/api/ps", timeout=3)
        import json
        models = [m["name"] for m in json.loads(out).get("models", [])]
        record("brain", "Ollama server", "PASS", f"up; loaded: {', '.join(models) or 'none'}")
    except Exception as e:
        record("brain", "Ollama server", "FAIL", f"not answering ({e})")
    try:
        import stt
        up = stt.is_up()
        record("brain", "speech engine (mlx-whisper GPU)", "PASS" if up else "WARN",
               "server up" if up else "server not running -- brain falls back to the slower CPU model")
    except Exception as e:
        record("brain", "speech engine (mlx-whisper GPU)", "WARN", str(e))
    try:
        import json as _json
        out, _ = http_get("http://127.0.0.1:11434/api/tags", timeout=3)
        have = [m["name"] for m in _json.loads(out).get("models", [])]
        record("brain", "vision model installed", "PASS" if any(n.startswith("qwen3-vl") for n in have) else "FAIL",
               ", ".join(n for n in have if n.startswith("qwen3-vl")) or "qwen3-vl missing (ollama pull qwen3-vl:2b-instruct)")
    except Exception:
        pass
    record("brain", "object detector model", "PASS" if os.path.exists(os.path.join(HERE, "models", "yolox_s.onnx")) else "FAIL",
           "yolox_s.onnx present" if os.path.exists(os.path.join(HERE, "models", "yolox_s.onnx")) else "models/yolox_s.onnx missing")
    db = os.path.join(HERE, "rocky_memory.db")
    if os.path.exists(db):
        try:
            con = sqlite3.connect(db)
            ok = con.execute("PRAGMA integrity_check").fetchone()[0]
            turns = con.execute("SELECT COUNT(*) FROM docs WHERE kind='turn'").fetchone()[0]
            facts = con.execute("SELECT COUNT(*) FROM docs WHERE kind!='turn' AND active=1").fetchone()[0]
            record("memory", "database integrity", "PASS" if ok == "ok" else "FAIL", f"{ok}; {turns} conversation turns, {facts} facts")
        except Exception as e:
            record("memory", "database integrity", "FAIL", str(e))
    else:
        record("memory", "database", "WARN", "not created yet (start brain.py once)")


def check_deep():
    print("\nDeep checks (loading models)...", flush=True)
    try:
        import ollama
        t = time.time()
        r = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": "Say hi in three words."}],
                        options={"num_predict": 12}, keep_alive=-1)
        record("brain", "chat model answers", "PASS" if time.time() - t < 8 else "WARN",
               f"{time.time() - t:.1f}s: {r['message']['content'].strip()[:40]!r}")
    except Exception as e:
        record("brain", "chat model answers", "FAIL", str(e))
    try:
        import tts
        tts.warmup()
        t = time.time(); tts.synthesize("This is a speed check.", "/tmp/diag_tts.wav")
        record("voice", "speech synthesis", "PASS" if time.time() - t < 1 else "WARN", f"{(time.time() - t) * 1000:.0f} ms per sentence")
    except Exception as e:
        record("voice", "speech synthesis", "FAIL", str(e))
    try:
        import stt
        clip = os.path.join(HERE, "recording.wav")   # a real mic clip: "go back to sleep"
        if not stt.start(wait_s=60):
            raise RuntimeError("speech server would not start")
        stt.transcribe(clip)
        t = time.time(); r = stt.transcribe(clip)
        record("brain", "speech recognition", "PASS" if r and r[0] and time.time() - t < 1.5 else "WARN",
               f"{time.time() - t:.2f}s -> {r[0] if r else None!r}")
    except Exception as e:
        record("brain", "speech recognition", "FAIL", str(e))
    try:
        import objects
        import numpy as np
        objects.detect(np.full((480, 640, 3), 127, np.uint8))
        t = time.time(); objects.detect(np.full((480, 640, 3), 127, np.uint8))
        record("brain", "object detector runs", "PASS" if time.time() - t < 1 else "WARN", f"{(time.time() - t) * 1000:.0f} ms per frame")
    except Exception as e:
        record("brain", "object detector runs", "FAIL", str(e))


# ----------------------------------------------------------------- soak
def soak(minutes, interval=45):
    print(f"Soak test for {minutes} minutes (a check every {interval}s). Ctrl+C to stop early.\n", flush=True)
    end = time.time() + minutes * 60
    last_uptime, failures, samples, reboots = None, 0, 0, 0
    body_ip = netfind.resolve_local(BODY_NAME, 3) or netfind.find_rocky_body("192.168.0.", 8266)
    while time.time() < end:
        stamp = time.strftime("%H:%M:%S")
        try:
            line, _ = http_get(f"http://{body_ip}:8080/status", timeout=8)
            kv = parse_kv(line); up = int(kv.get("uptime_s", 0))
            sl, _ = http_get(f"http://{body_ip}:8080/selftest", timeout=15)
            peak = float(sl.split("=")[1])
            note = ""
            if last_uptime is not None and up < last_uptime:
                reboots += 1; note = f"  ** REBOOTED (reset reason {kv.get('reset')}) **"
            last_uptime = up
            bad = peak < 0.5
            failures += bad
            print(f"[{stamp}] up {up:6d}s  heap {int(kv['heap'])//1024:4d}K  rssi {kv['rssi']:>4}  selftest {peak:.3f}"
                  + ("  ** DETECTOR DEAD **" if bad else "") + note, flush=True)
        except Exception as e:
            failures += 1
            print(f"[{stamp}] ** board unreachable: {e}", flush=True)
            body_ip = netfind.resolve_local(BODY_NAME, 3) or body_ip
        samples += 1
        time.sleep(interval)
    print(f"\nSoak finished: {samples} samples, {failures} failure(s), {reboots} unexpected reboot(s).")
    return failures + reboots


def main():
    ap = argparse.ArgumentParser(description="Rocky system diagnostics")
    ap.add_argument("--deep", action="store_true", help="also load and time the speech/chat/voice models")
    ap.add_argument("--soak", type=int, metavar="MIN", help="watch the system for MIN minutes")
    args = ap.parse_args()
    print("Rocky diagnostics -- " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    print("Network"); body_ip, cam_ip = check_network()
    print("\nBody board"); check_body(body_ip); check_body_active(body_ip)
    print("\nCamera"); check_camera(cam_ip)
    print("\nMac / brain"); check_mac()
    if args.deep:
        check_deep()
    fails = [r for r in results if r[2] == "FAIL"]
    warns = [r for r in results if r[2] == "WARN"]
    print(f"\n{len(results)} checks: {sum(r[2] == 'PASS' for r in results)} pass, {len(warns)} warn, {len(fails)} FAIL")
    for a, n, s, d in fails:
        print(f"  FAIL  {a}/{n}: {d}")
    if args.soak:
        return 1 if soak(args.soak) else (1 if fails else 0)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
