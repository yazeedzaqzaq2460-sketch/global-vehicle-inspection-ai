import cv2
from pathlib import Path

from ultralytics import YOLO


class PlateDetector:
    def __init__(self):
        self.model_path = Path(
            "models/plate_detection/license-plate-finetune-v1n.pt"
        )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Plate detection model not found: {self.model_path}"
            )

        self.model = YOLO(str(self.model_path))

        self.model_name = "YOLO11 License Plate Detector"

        print(f"Plate Detector initialized ({self.model_name}).")

    def detect(self, image_path: str):
        results = self.model.predict(
            source=image_path,
            conf=0.25,
            verbose=False,
        )

        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return None

        box = result.boxes[0]

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        confidence = float(box.conf[0])

        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")

        cropped_plate = image[y1:y2, x1:x2]

        output_dir = Path("outputs/plates")
        output_dir.mkdir(parents=True, exist_ok=True)

        cropped_path = output_dir / "detected_plate.jpg"

        cv2.imwrite(str(cropped_path), cropped_plate)

        return {
            "bbox": [x1, y1, x2, y2],
            "confidence": confidence,
            "cropped_image": cropped_path,
        }