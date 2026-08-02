from pathlib import Path
from typing import Any

from src.core.decision_engine import DecisionEngine
from src.detection.damage_detector import DamageDetector
from src.ocr.ocr_engine import OCREngine
from src.reasoning.association_engine import AssociationEngine
from src.reasoning.severity_engine import SeverityEngine
from src.vehicle_parts.vehicle_part_detector import VehiclePartDetector


class InspectionPipeline:
    def __init__(self) -> None:
        self.damage_detector = DamageDetector(
            model_path="models/yolo/best.pt",
            confidence_threshold=0.25,
            iou_threshold=0.45,
            device=0,
        )

        self.vehicle_part_detector = VehiclePartDetector(
            model_path="models/vehicle_parts/best.pt",
            confidence_threshold=0.33,
            iou_threshold=0.50,
            device=0,
        )

        self.association_engine = AssociationEngine(
            minimum_mask_overlap=0.10,
            minimum_bbox_iou=0.05,
            maximum_center_distance_ratio=0.35,
        )

        self.severity_engine = SeverityEngine()

        self.ocr_engine = OCREngine()
        self.decision_engine = DecisionEngine()

    def inspect_image(
        self,
        image_path: str,
    ) -> dict[str, Any]:
        image_path_object = Path(image_path)

        if not image_path_object.exists():
            raise FileNotFoundError(
                f"Input image not found: {image_path_object}"
            )

        damages = self.damage_detector.predict(
            str(image_path_object)
        )

        vehicle_parts = self.vehicle_part_detector.detect(
            str(image_path_object)
        )

        associated_damages = self.association_engine.associate(
            damages=damages,
            vehicle_parts=vehicle_parts,
        )

        evaluated_damages = self.severity_engine.evaluate(
            damages=associated_damages,
            vehicle_parts=vehicle_parts,
        )

        vehicle_data = self.ocr_engine.extract(
            str(image_path_object)
        )

        highest_confidence = max(
            (
                damage["confidence"]
                for damage in evaluated_damages
            ),
            default=0.0,
        )

        highest_severity_score = max(
            (
                damage["severity_score"]
                for damage in evaluated_damages
            ),
            default=0.0,
        )

        decision = self.decision_engine.evaluate(
            evaluated_damages
        )

        return {
            "image_path": image_path,
            "decision": decision,
            "damage_count": len(evaluated_damages),
            "highest_confidence": round(
                highest_confidence,
                4,
            ),
            "highest_severity_score": round(
                highest_severity_score,
                2,
            ),
            "damages": evaluated_damages,
            "vehicle_parts": vehicle_parts,
            "vehicle_data": vehicle_data,
        }