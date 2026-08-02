from pathlib import Path
from typing import Any

from ultralytics import YOLO


class VehiclePartDetector:
    def __init__(
        self,
        model_path: str = "models/vehicle_parts/best.pt",
        confidence_threshold: float = 0.33,
        iou_threshold: float = 0.50,
        device: int | str = 0,
    ) -> None:
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Vehicle-parts model not found: {self.model_path}"
            )

        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                "confidence_threshold must be between 0 and 1."
            )

        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError(
                "iou_threshold must be between 0 and 1."
            )

        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device

        self.model = YOLO(str(self.model_path))

        print("Vehicle Part Detector initialized.")

    def detect(self, image_path: str) -> list[dict[str, Any]]:
        image_path_object = Path(image_path)

        if not image_path_object.exists():
            raise FileNotFoundError(
                f"Input image not found: {image_path_object}"
            )

        results = self.model.predict(
            source=str(image_path_object),
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=640,
            device=self.device,
            retina_masks=True,
            verbose=False,
        )

        detections: list[dict[str, Any]] = []

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            masks = result.masks.data.cpu().numpy() if result.masks else None

            for index, box in enumerate(result.boxes):
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                coordinates = box.xyxy[0].tolist()

                detection: dict[str, Any] = {
                    "class_id": class_id,
                    "class_name": result.names[class_id],
                    "confidence": round(confidence, 4),
                    "bbox": {
                        "x1": round(float(coordinates[0]), 2),
                        "y1": round(float(coordinates[1]), 2),
                        "x2": round(float(coordinates[2]), 2),
                        "y2": round(float(coordinates[3]), 2),
                    },
                    "mask": None,
                }

                if masks is not None and index < len(masks):
                    detection["mask"] = masks[index]

                detections.append(detection)

        return detections