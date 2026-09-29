# YOLO Object Detection — Pipeline Documentation

## 1. Overview

**YOLO** ("You Only Look Once") is a family of real-time object detection
models. Its core idea: treat detection as a single regression problem, so
one neural network looks at an entire image **once** and directly predicts
both *where* objects are and *what* they are — instead of scanning the
image in multiple separate passes like older two-stage detectors.

This single-pass design is what makes YOLO fast enough to run on live
video, and it's the same design choice that shows up in both how the
model is trained and how it runs at inference time.

---

## 2. Inference Pipeline

What happens when a trained YOLO model detects objects in an image or
video frame:

```
Input image
    |
    v
CNN backbone            (extracts features, one forward pass)
    |
    v
Grid cell predictions   (each grid cell predicts box + confidence + class)
    |
    v
Non-max suppression     (removes duplicate/overlapping boxes)
    |
    v
Final detections        (boxes + labels, ready to use)
```

**Note:** Newer versions (e.g. YOLO26) are trained to be end-to-end and
NMS-free — the network directly outputs clean, non-duplicate boxes
without a separate filtering step. Functionally the output looks the
same; it's just handled differently under the hood, and it's part of
why newer versions deploy more easily on constrained hardware.

---

## 3. Training Pipeline

How the model learns to make those predictions in the first place
(already done for you if you're using pretrained weights like
`yolo26n.pt`, trained on the COCO dataset's 80 object classes):

```
Forward pass
    |
    v
Compute loss             (compare predictions to ground-truth boxes)
    |
    v
Backpropagation          (compute gradients of the loss)
    |
    v
Update weights           (adjust network parameters)
    |
    +--- repeats over many images / epochs ---> back to Forward pass
```

**The loss is a combination of three parts, computed together:**

| Component | What it measures |
|---|---|
| Localization loss | How far off the predicted box coordinates are from the real box |
| Confidence loss | Whether the model correctly says "yes/no, something is here" |
| Classification loss | Whether the predicted class label is correct |

Because all three are learned **jointly, in one network**, the resulting
model can also do all three in a **single pass** at inference time. The
training design and the inference speed are directly connected — not a
coincidence.

---

## 4. Common Applications

Anywhere detection speed matters as much as accuracy:

- Autonomous vehicles / ADAS (pedestrians, vehicles, signs)
- Surveillance and security (live camera monitoring)
- Robotics (locating objects to pick up or avoid, frame by frame)
- Retail and inventory (shelf monitoring, stock counting)
- Sports and video analytics (live player/ball tracking)
- Industrial inspection (defect spotting on moving lines)
- **Edge / embedded deployment** — lightweight nano/tiny variants
  quantized to run directly on microcontrollers or NPUs, without a
  cloud round-trip (relevant for PSoC 6–class targets)

---

## 5. Hands-On Testing Progression

The practical path from zero to live detection, in increasing realism —
same pipeline throughout, only the `source` changes:

| Stage | Source | Purpose |
|---|---|---|
| 1. Static image | `"https://ultralytics.com/images/bus.jpg"` | Confirm the pipeline works at all, zero setup |
| 2. Video file / URL | YouTube URL, e.g. `"https://youtu.be/LNwODJXcvt4"` | Object-rich footage, no physical props needed |
| 3. Live webcam | `0` (default camera) | See true real-time detection, frame by frame |

Minimal pattern used across all three:

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")          # nano = smallest/fastest variant
results = model.predict(
    source=SOURCE,                   # image path/URL, video URL, or 0 for webcam
    conf=0.25,                       # confidence threshold
    show=True,                       # live annotated display window
    stream=True,                     # process frame-by-frame (needed for video/webcam)
)
for result in results:
    pass                             # must iterate to actually process frames
```

Lowering `conf` shows more (sometimes wrong) detections; raising it
shows fewer but more certain ones — useful for building intuition
around confidence scores.

---

## 6. Environment Setup Notes (Windows)

Common gotchas encountered when setting this up in a Windows `venv`:

- **PowerShell blocks venv activation by default.** Fix once with:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```
  This only allows locally-created scripts (like `Activate.ps1`) to run;
  it does not weaken protection against untrusted remote scripts.
- **Confirm the venv is actually active** — the prompt should show
  `(venv)` at the start of the line before installing packages or
  running scripts. If it doesn't, `pip install` and `python` are
  silently using your system Python instead.
- **Check file extensions** — make sure script files are saved as
  `.py`, not just a bare filename (Windows sometimes hides known
  extensions in File Explorer).

---

## 7. Key Takeaway

One architecture, two roles: the same single-pass network that gets
trained on labeled images (learning location + objectness + class
jointly) is what makes it possible to run that same network live, frame
by frame, on a webcam feed — no separate detection stages, no per-frame
region proposals, just one forward pass repeated fast enough to feel
real time.
