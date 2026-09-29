"""
Real-time YOLO26 object detection on a YouTube video.

Same idea as yolo_realtime_webcam.py, but instead of your physical
camera, it pulls object-rich footage straight from a YouTube URL --
useful for testing without needing physical objects on hand.

A window will pop up playing the video with live boxes and labels
drawn on it. Press 'q' with that window focused to quit.
"""

from ultralytics import YOLO

# --- Hardcoded config ---------------------------------------------------
MODEL_NAME = "yolo26n.pt"     # "n" = nano, smallest/fastest variant

# Ultralytics' own example source (busy street scene, from their docs) --
# swap this for any other YouTube URL you want to test against.
VIDEO_SOURCE = "https://youtu.be/LNwODJXcvt4"

CONFIDENCE_THRESHOLD = 0.25    # ignore detections below this confidence
# -------------------------------------------------------------------------

def main():
    model = YOLO(MODEL_NAME)

    # First run may need to auto-install a small extra package to read
    # YouTube URLs -- if you see an error mentioning a missing package,
    # let it install and just re-run the script.
    results = model.predict(
        source=VIDEO_SOURCE,
        conf=CONFIDENCE_THRESHOLD,
        show=True,
        stream=True,
    )

    for result in results:
        pass  # each `result` is one frame's detections


if __name__ == "__main__":
    main()
