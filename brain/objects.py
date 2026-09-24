"""Fast object detection for Rocky's camera: YOLOX-s (COCO, 80 classes) run
through OpenCV's DNN module -- ~0.2 s per frame, no extra Python packages, no
LLM involved. Used for "what do you see", "do you see my phone", room scans
and remembering where things were last seen."""
import os
import re

import cv2
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "yolox_s.onnx")
INPUT = 640
CONF = 0.35
NMS_IOU = 0.5

COCO = ("person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant stop_sign "
        "parking_meter bench bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag "
        "tie suitcase frisbee skis snowboard sports_ball kite baseball_bat baseball_glove skateboard surfboard "
        "tennis_racket bottle wine_glass cup fork knife spoon bowl banana apple sandwich orange broccoli carrot "
        "hot_dog pizza donut cake chair couch potted_plant bed dining_table toilet tv laptop mouse remote keyboard "
        "cell_phone microwave oven toaster sink refrigerator book clock vase scissors teddy_bear hair_drier "
        "toothbrush").split()
COCO = [c.replace("_", " ") for c in COCO]

# What people say -> what the detector calls it.
ALIASES = {
    "phone": "cell phone", "mobile": "cell phone", "smartphone": "cell phone", "iphone": "cell phone",
    "mug": "cup", "coffee cup": "cup", "glass": "wine glass", "television": "tv", "monitor": "tv", "screen": "tv",
    "computer": "laptop", "notebook": "laptop", "macbook": "laptop", "table": "dining table", "desk": "dining table",
    "sofa": "couch", "plant": "potted plant", "controller": "remote", "remote control": "remote",
    "bag": "backpack", "rucksack": "backpack", "teddy": "teddy bear", "person": "person", "human": "person",
    "people": "person", "man": "person", "woman": "person", "guy": "person", "girl": "person", "boy": "person",
}
_PLURAL_SKIP = {"glasses", "scissors"}

_net = None
_grids = None


def _get_net():
    global _net
    if _net is None:
        _net = cv2.dnn.readNetFromONNX(MODEL_PATH)
    return _net


def _grid_and_strides():
    global _grids
    if _grids is None:
        grids, strides = [], []
        for s in (8, 16, 32):
            n = INPUT // s
            xv, yv = np.meshgrid(np.arange(n), np.arange(n))
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            strides.append(np.full((n * n, 1), s))
        _grids = (np.concatenate(grids, 0).astype(np.float32), np.concatenate(strides, 0).astype(np.float32))
    return _grids


def detect(img, conf=CONF):
    """img: BGR ndarray or a path. Returns [(label, score, (x, y, w, h)), ...] best first."""
    if isinstance(img, str):
        img = cv2.imread(img)
    if img is None:
        return []
    h, w = img.shape[:2]
    ratio = min(INPUT / h, INPUT / w)
    rh, rw = int(h * ratio), int(w * ratio)
    padded = np.full((INPUT, INPUT, 3), 114, np.float32)
    padded[:rh, :rw] = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    net = _get_net()
    net.setInput(padded.transpose(2, 0, 1)[None])
    out = net.forward()[0]                      # (8400, 85)
    grid, stride = _grid_and_strides()
    xy = (out[:, :2] + grid) * stride
    wh = np.exp(out[:, 2:4]) * stride
    scores_all = out[:, 4:5] * out[:, 5:]
    cls = scores_all.argmax(1)
    score = scores_all[np.arange(len(cls)), cls]
    keep = score > conf
    if not keep.any():
        return []
    xy, wh, cls, score = xy[keep], wh[keep], cls[keep], score[keep]
    boxes = np.concatenate([(xy - wh / 2) / ratio, wh / ratio], 1)
    idx = cv2.dnn.NMSBoxes(boxes.tolist(), score.tolist(), conf, NMS_IOU)
    result = []
    for i in np.array(idx).flatten():
        x, y, bw, bh = boxes[i]
        result.append((COCO[cls[i]], float(score[i]), (int(x), int(y), int(bw), int(bh))))
    result.sort(key=lambda d: -d[1])
    return result


def count_labels(detections):
    counts = {}
    for label, _, _ in detections:
        counts[label] = counts.get(label, 0) + 1
    return counts


def _plural(label, n):
    if n == 1 or label in _PLURAL_SKIP:
        return label
    return {"person": "people", "knife": "knives", "sheep": "sheep"}.get(label, label + "s")


_NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def describe(counts, limit=6, ignore=("person",)):
    """{'cup': 2, 'laptop': 1} -> 'two cups and a laptop'. People are reported
    separately (by face recognition), so they are skipped by default."""
    items = [(l, n) for l, n in sorted(counts.items(), key=lambda kv: -kv[1]) if l not in ignore][:limit]
    if not items:
        return ""
    bits = []
    for label, n in items:
        if n == 1:
            bits.append(("an " if label[0] in "aeiou" else "a ") + label)
        else:
            bits.append(f"{_NUMBER_WORDS.get(n, str(n))} {_plural(label, n)}")
    return bits[0] if len(bits) == 1 else ", ".join(bits[:-1]) + " and " + bits[-1]


def resolve_label(word):
    """'my phone' / 'the mugs' -> a COCO label, or None if the detector can't know it."""
    w = re.sub(r"^(?:my|the|a|an|any|some)\s+", "", word.lower().strip(" ?.!,"))
    if w in ALIASES:
        return ALIASES[w]
    if w in COCO:
        return w
    if w.endswith("s") and w[:-1] in COCO:
        return w[:-1]
    if w.endswith("s") and w[:-1] in ALIASES:
        return ALIASES[w[:-1]]
    return None


if __name__ == "__main__":
    import sys
    import time
    path = sys.argv[1] if len(sys.argv) > 1 else "latest.jpg"
    detect(path)
    t = time.time()
    dets = detect(path)
    print(f"{(time.time() - t) * 1000:.0f} ms:", [(l, round(s, 2)) for l, s, _ in dets])
    print(describe(count_labels(dets)))
