from src.core.decision_engine import DecisionEngine
from src.detection.damage_detector import DamageDetector
from src.ocr.ocr_engine import OCREngine


class InspectionPipeline:
    def __init__(self):
        self.damage_detector = DamageDetector("models/yolo/best.pt")
        self.ocr_engine = OCREngine()
        self.decision_engine = DecisionEngine()

    def inspect_image(self, image_path: str) -> dict:
        damages = self.damage_detector.predict(image_path)
        vehicle_data = self.ocr_engine.extract(image_path)

        highest_confidence = max(
            (damage["confidence"] for damage in damages),
            default=0.0,
        )

        decision = self.decision_engine.evaluate(damages)

        return {
            "image_path": image_path,
            "decision": decision,
            "damage_count": len(damages),
            "highest_confidence": round(highest_confidence, 4),
            "damages": damages,
            "vehicle_data": vehicle_data,
        }