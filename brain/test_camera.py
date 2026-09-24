import requests
import cv2
import numpy as np
import time

CAMERA_URL = "http://192.168.0.11/jpg"

print("==============================")
print("       ROCKY CAMERA TEST")
print("==============================")
print()
print("Camera:", CAMERA_URL)
print()

while True:
    try:
        start = time.time()

        response = requests.get(
            CAMERA_URL,
            timeout=5
        )

        if response.status_code != 200:
            print("HTTP error:", response.status_code)
            time.sleep(1)
            continue

        image = np.frombuffer(
            response.content,
            dtype=np.uint8
        )

        frame = cv2.imdecode(
            image,
            cv2.IMREAD_COLOR
        )

        if frame is None:
            print("Could not decode JPEG")
            time.sleep(1)
            continue

        height, width = frame.shape[:2]
        elapsed = time.time() - start

        print(
            f"Frame: {width}x{height} | "
            f"{len(response.content)} bytes | "
            f"{elapsed:.2f}s"
        )

        cv2.imshow("ROCKY CAMERA", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    except Exception as e:
        print("Camera error:", e)
        time.sleep(1)

cv2.destroyAllWindows()
