#!/usr/bin/env python3
"""Push a firmware image to Rocky's camera board over Wi-Fi (no USB, no firewall
change: this machine connects OUT to the board). Used by `pio run -e ota -t upload`
and runnable by hand:  python3 tools/ota_push.py [path/to/firmware.bin]

Standard library only, on purpose: PlatformIO runs this with whatever python3
is first on its PATH, which is not necessarily one with third-party packages."""
import os
import socket
import sys
import time
import urllib.error
import urllib.request
import uuid

TOKEN = "change-me"  # must match OTA_PASSWORD in src/main.cpp
PORT = 8080
DEFAULT_BIN = os.path.join(os.path.dirname(__file__), "..", ".pio", "build", "ota", "firmware.bin")


def find_board():
    try:
        return socket.gethostbyname("rocky-cam.local")
    except OSError:
        pass
    from concurrent.futures import ThreadPoolExecutor
    for prefix in ("192.168.0.", "192.168.1."):
        def probe(i, prefix=prefix):
            ip = f"{prefix}{i}"
            try:
                socket.create_connection((ip, PORT), timeout=1.5).close()
                return ip
            except OSError:
                return None
        with ThreadPoolExecutor(64) as ex:
            for ip in ex.map(probe, range(1, 255)):
                if ip:
                    return ip
    return None


def get_version(url):
    with urllib.request.urlopen(f"{url}/version", timeout=5) as r:
        return r.read().decode().strip()


def multipart_body(filename, data):
    boundary = uuid.uuid4().hex
    head = (f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="firmware"; filename="{filename}"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n").encode()
    tail = f"\r\n--{boundary}--\r\n".encode()
    return boundary, head + data + tail


def main():
    image = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BIN)
    if not os.path.exists(image):
        sys.exit(f"Firmware image not found: {image}")
    host = find_board()
    if not host:
        sys.exit("Could not find Rocky's camera board on the network (is it powered and on Wi-Fi?).")
    url = f"http://{host}:{PORT}"
    before = get_version(url)
    with open(image, "rb") as f:
        data = f.read()
    print(f"Board at {host}, running build: {before}")
    print(f"Uploading {os.path.basename(image)} ({len(data) // 1024} KB)...")
    boundary, body = multipart_body(os.path.basename(image), data)
    req = urllib.request.Request(
        f"{url}/update", data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}", "X-Rocky-Token": TOKEN})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            print(f"Board says: HTTP {r.status} {r.read().decode().strip()}")
    except urllib.error.HTTPError as e:
        sys.exit(f"Board rejected the update: HTTP {e.code} {e.read().decode().strip()}")
    print("Waiting for the board to reboot...")
    time.sleep(6)
    for _ in range(25):
        try:
            after = get_version(url)
            print(f"Back online, now running build: {after}")
            if after == before:
                print("NOTE: build stamp unchanged (the image was built from identical sources).")
            return
        except (urllib.error.URLError, OSError):
            time.sleep(1.5)
    sys.exit("Board did not come back within ~45s.")


if __name__ == "__main__":
    main()
