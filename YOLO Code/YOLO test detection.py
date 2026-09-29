"""
Basic YOLO26 object detection test.

What this does:
  1. Loads a pretrained YOLO26 nano model (auto-downloads weights on first run)
  2. Runs detection on a single image
  3. Saves an annotated copy of the image with boxes + labels drawn
  4. Prints every detected object with its confidence score

No training required -- the pretrained model already knows 80 everyday
object classes (person, car, dog, laptop, bottle, chair, etc.) from the
COCO dataset.
"""

from ultralytics import YOLO

# --- Hardcoded config ---------------------------------------------------
MODEL_NAME = "yolo26n.pt"     # "n" = nano, smallest/fastest variant

# Using Ultralytics' own hosted sample photo so this runs with zero setup.
# Once it works, swap this for a path to your own photo, e.g. "test_image.jpg"
# (the file must actually exist at that path in this folder).
IMAGE_PATH = "https://ultralytics.com/images/bus.jpg"

OUTPUT_PATH = "output.jpg"    # annotated image will be saved here
CONFIDENCE_THRESHOLD = 0.25   # ignore detections below this confidence
# -------------------------------------------------------------------------

def main():
    # Load the pretrained model (downloads weights automatically first time)
    model = YOLO(MODEL_NAME)

    # Run detection on the image
    results = model(IMAGE_PATH, conf=CONFIDENCE_THRESHOLD)

    for result in results:
        # Save the annotated image (boxes + labels drawn on it)
        result.save(filename=OUTPUT_PATH)

        # Print out what was found
        print(f"\nFound {len(result.boxes)} object(s):")
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = model.names[class_id]
            print(f"  - {label}: {confidence:.2f} confidence")

    print(f"\nAnnotated image saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()