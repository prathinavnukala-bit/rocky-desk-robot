import json
import os
from collections import Counter

import cv2
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
DETECTOR_MODEL = os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx")
RECOGNIZER_MODEL = os.path.join(MODEL_DIR, "face_recognition_sface_2021dec.onnx")
FACES_FILE = os.path.join(os.path.dirname(__file__), "faces.json")

# OpenCV zoo's recommended SFace threshold (TAR@FAR=1e-3) for cosine similarity.
COSINE_MATCH_THRESHOLD = 0.363
# Higher bar for the unprompted "greet on sight" path: a wrong greeting at a
# stranger is worse than a missed one, and it runs every few seconds.
PROACTIVE_MATCH_THRESHOLD = 0.45
# Keep up to this many separate fingerprints per person (different angles,
# lighting, expressions) instead of one blurred average.
MAX_EMBEDDINGS = 12
# A new enrollment sample this similar to one already stored adds nothing.
DUPLICATE_SIMILARITY = 0.93

_detector = None
_recognizer = None


def _get_detector(input_size):
    global _detector
    if _detector is None:
        # 0.9 (opencv_zoo's demo default) is too strict for this blurry, low-res
        # ESP32-CAM feed -- 0.6 still rejects junk while catching real faces.
        _detector = cv2.FaceDetectorYN_create(
            DETECTOR_MODEL, "", input_size, score_threshold=0.6
        )
    else:
        _detector.setInputSize(input_size)
    return _detector


def _get_recognizer():
    global _recognizer
    if _recognizer is None:
        _recognizer = cv2.FaceRecognizerSF_create(RECOGNIZER_MODEL, "")
    return _recognizer


def _load_faces():
    if os.path.exists(FACES_FILE):
        with open(FACES_FILE) as f:
            return json.load(f)
    return {}


def _save_faces(faces):
    with open(FACES_FILE, "w") as f:
        json.dump(faces, f, indent=2)


def _embeddings_of(entry):
    """All stored fingerprints for a person (older entries only had one
    averaged 'embedding')."""
    embs = entry.get("embeddings")
    if embs:
        return [np.array(e, dtype=np.float32) for e in embs]
    if entry.get("embedding") is not None:
        return [np.array(entry["embedding"], dtype=np.float32)]
    return []


def _detections(img):
    h, w = img.shape[:2]
    detector = _get_detector((w, h))
    _, detections = detector.detect(img)
    if detections is None:
        return []
    return sorted(detections, key=lambda d: d[2] * d[3], reverse=True)


def _largest_face(img):
    dets = _detections(img)
    return dets[0] if dets else None


def _feature(img, face):
    recognizer = _get_recognizer()
    aligned = recognizer.alignCrop(img, face)
    return recognizer.feature(aligned)


def _embed(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    face = _largest_face(img)
    if face is None:
        return None
    return _feature(img, face)


def _cosine(a, b):
    a = a.flatten(); b = b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _score_against(feature, faces):
    """Best cosine similarity of `feature` to each enrolled person -- the max
    over that person's individual fingerprints, so one good angle is enough."""
    scores = {}
    vec = feature.flatten()
    for name, entry in faces.items():
        embs = _embeddings_of(entry)
        if embs:
            scores[name] = max(_cosine(vec, e) for e in embs)
    return scores


def _add_embedding(entry, vec):
    embs = entry.setdefault("embeddings", [e.tolist() for e in _embeddings_of(entry)])
    for e in embs:
        if _cosine(vec, np.array(e, dtype=np.float32)) >= DUPLICATE_SIMILARITY:
            return False
    embs.append(vec.tolist())
    if len(embs) > MAX_EMBEDDINGS:
        del embs[0]
    entry["embedding"] = np.mean(np.array(embs, dtype=np.float32), axis=0).tolist()  # legacy field
    entry["samples"] = len(embs)
    return True


def enroll(name, image_path):
    """Register one photo of a person (see enroll_frames for the multi-shot version)."""
    return enroll_frames(name, [image_path])


def enroll_frames(name, image_paths):
    """Register several frames of the same person. Frames without a usable
    face are skipped; near-duplicate frames add nothing. Returns (ok, message)."""
    faces = _load_faces()
    entry = faces.get(name, {"samples": 0, "embedding": None})
    added, no_face = 0, 0
    for path in image_paths:
        feature = _embed(path)
        if feature is None:
            no_face += 1
            continue
        if _add_embedding(entry, feature.flatten()):
            added += 1
    if added == 0 and not entry.get("embeddings"):
        return False, "No face detected in the image."
    faces[name] = entry
    _save_faces(faces)
    return True, (f"Enrolled {name}: {added} new sample(s), {entry['samples']} stored"
                  + (f" ({no_face} frame(s) had no face)." if no_face else "."))


def identify(image_path, min_score=None):
    """Return (name, score) for the best match above threshold, else (None, best_score)."""
    threshold = COSINE_MATCH_THRESHOLD if min_score is None else min_score
    feature = _embed(image_path)
    if feature is None:
        return None, 0.0
    faces = _load_faces()
    if not faces:
        return None, 0.0
    scores = _score_against(feature, faces)
    if not scores:
        return None, 0.0
    best_name = max(scores, key=scores.get)
    best = scores[best_name]
    return (best_name, best) if best >= threshold else (None, best)


def identify_frames(image_paths, min_score=None):
    """Identify from several frames and vote: a single blurry or half-turned
    frame can't decide it. Returns (name, mean_score_of_winning_votes)."""
    votes, scores = Counter(), {}
    for path in image_paths:
        name, score = identify(path, min_score=min_score)
        if name:
            votes[name] += 1
            scores.setdefault(name, []).append(score)
    if not votes:
        return None, 0.0
    winner, count = votes.most_common(1)[0]
    needed = 1 if len(image_paths) == 1 else 2
    if count < needed:
        return None, 0.0
    return winner, float(np.mean(scores[winner]))


def identify_all(image_path, min_score=None):
    """Everyone visible in one frame: [(name_or_None, score, (x, y, w, h)), ...],
    largest face first. Lets Rocky notice more than one person at once."""
    threshold = COSINE_MATCH_THRESHOLD if min_score is None else min_score
    img = cv2.imread(image_path)
    if img is None:
        return []
    faces = _load_faces()
    out = []
    for det in _detections(img):
        feature = _feature(img, det)
        scores = _score_against(feature, faces) if faces else {}
        name, best = None, 0.0
        if scores:
            cand = max(scores, key=scores.get)
            best = scores[cand]
            if best >= threshold:
                name = cand
        out.append((name, best, tuple(int(v) for v in det[:4])))
    return out
