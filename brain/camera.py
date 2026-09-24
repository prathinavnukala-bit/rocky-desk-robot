import cv2
import requests
import numpy as np
import time

CAMERA_URL = "http://192.168.0.11/jpg"


def get_frame():
    try:
        response = requests.get(
            CAMERA_URL,
            timeout=10,
            headers={"Connection": "close"}
        )

        response.raise_for_status()

        image_data = np.frombuffer(
            response.content,
            dtype=np.uint8
        )

        frame = cv2.imdecode(
            image_data,
            cv2.IMREAD_COLOR
        )

        return frame

    except Exception as e:
        print(f"Frame fetch failed: {e}")
        return None


def main():

    print("================================")
    print("       ROCKY CAMERA")
    print("================================")
    print("Starting at 1 FPS")
    print("Press Q to quit")
    print()

    frame_count = 0
    fail_count = 0
    last_save = time.time()

    while True:

        frame = get_frame()

        if frame is not None:

            frame_count += 1
            fail_count = 0

            height, width = frame.shape[:2]

            print(
                f"Frame {frame_count} received "
                f"({width}x{height})"
            )

            cv2.imshow("Rocky's Eye", frame)

            # Save one frame every 5 seconds
            if time.time() - last_save >= 5:

                cv2.imwrite("latest.jpg", frame)

                print("Saved -> latest.jpg")

                last_save = time.time()

        else:

            fail_count += 1

            print(
                f"Camera failed. "
                f"Retry #{fail_count}"
            )

            # Give ESP32 extra recovery time
            time.sleep(
                min(3.0, 0.5 * fail_count)
            )

        # Check for Q
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # IMPORTANT:
        # Only request approximately once per second.
        time.sleep(1.0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
