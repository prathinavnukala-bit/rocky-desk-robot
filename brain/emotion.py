import os

import cv2
import numpy as np
import onnxruntime as ort

import face_id

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "emotion-ferplus-8.onnx")

# Fixed output order for this model (FER+ training label set).
EMOTION_LABELS = [
    "neutral", "happy", "surprised", "sad",
    "angry", "disgusted", "scared", "contemptuous",
]

_session = None


def _get_session():
    global _session
    if _session is None:
        _session = ort.InferenceSession(MODEL_PATH)
    return _session


def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def detect_face_emotion(image_path):
    """Return (label, confidence) for the largest face in the image, or (None, 0.0)."""
    img = cv2.imread(image_path)
    if img is None:
        return None, 0.0

    face = face_id._largest_face(img)
    if face is None:
        return None, 0.0

    x, y, w, h = [max(0, int(v)) for v in face[:4]]
    crop = img[y:y + h, x:x + w]
    if crop.size == 0:
        return None, 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64)).astype(np.float32)
    input_tensor = resized.reshape(1, 1, 64, 64)

    session = _get_session()
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_tensor})[0][0]

    probs = _softmax(output)
    idx = int(np.argmax(probs))
    return EMOTION_LABELS[idx], float(probs[idx])
