from pathlib import Path

from ultralytics import YOLO


class DamageDetector:
    def __init__(self, model_path: str):
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = YOLO(str(model_path))
        print("YOLO model loaded successfully.")

    def predict(self, image_path: str) -> list[dict]:
        results = self.model.predict(
            source=image_path,
            conf=0.25,
            imgsz=640,
            device=0,
            save=True,
            project="outputs",
            name="damage_detection",
            exist_ok=True,
        )

        detections = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                coordinates = box.xyxy[0].tolist()

                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": result.names[class_id],
                        "confidence": round(confidence, 4),
                        "bbox": {
                            "x1": round(coordinates[0], 2),
                            "y1": round(coordinates[1], 2),
                            "x2": round(coordinates[2], 2),
                            "y2": round(coordinates[3], 2),
                        },
                    }
                )

        return detections