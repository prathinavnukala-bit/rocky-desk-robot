"""
Standalone tool to manage Rocky's known faces -- independent of the voice
pipeline (doesn't load Whisper/Ollama), so it stays usable any time, from
any photo, regardless of what else changes in the project. Enrollments
persist in faces.json permanently.

Usage:
    python3 people.py enroll <name> <image_path> [<image_path> ...]
    python3 people.py enroll-live <name> [num_shots]
    python3 people.py list
    python3 people.py forget <name>
"""
import os
import sys
import time

import requests

import face_id

CAMERA_URL = "http://192.168.0.11/jpg"


def fetch_live_frame(save_path):
    resp = requests.get(CAMERA_URL, timeout=15)
    resp.raise_for_status()
    with open(save_path, "wb") as f:
        f.write(resp.content)
    return save_path


def cmd_enroll(name, image_paths):
    ok_count = 0
    for path in image_paths:
        ok, message = face_id.enroll(name, path)
        print(f"{path}: {message}")
        ok_count += ok
    print(f"\n{name}: {ok_count}/{len(image_paths)} sample(s) enrolled.")


def cmd_enroll_live(name, shots=3):
    print(f"Enrolling {name} from {shots} live camera frame(s). Look at the camera.")
    for i in range(shots):
        path = fetch_live_frame(f"_enroll_{name}_{i}.jpg")
        ok, message = face_id.enroll(name, path)
        print(f"  shot {i + 1}/{shots}: {message}")
        os.remove(path)
        if i < shots - 1:
            time.sleep(1.5)


def cmd_list():
    faces = face_id._load_faces()
    if not faces:
        print("No one enrolled yet.")
        return
    for name, entry in faces.items():
        print(f"{name}: {entry['samples']} sample(s)")


def cmd_forget(name):
    faces = face_id._load_faces()
    if name in faces:
        del faces[name]
        face_id._save_faces(faces)
        print(f"Forgot {name}.")
    else:
        print(f"{name} not found. Known: {', '.join(faces) or '(none)'}")


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return

    command, rest = args[0], args[1:]

    if command == "enroll" and len(rest) >= 2:
        cmd_enroll(rest[0], rest[1:])
    elif command == "enroll-live" and len(rest) >= 1:
        shots = int(rest[1]) if len(rest) > 1 else 3
        cmd_enroll_live(rest[0], shots)
    elif command == "list":
        cmd_list()
    elif command == "forget" and len(rest) >= 1:
        cmd_forget(rest[0])
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
