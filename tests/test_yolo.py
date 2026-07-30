import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.damage_detector import DamageDetector

detector = DamageDetector("models/yolo/best.pt")

detections = detector.predict("test_images/car_damage.jpg")

print("\nDetected Damages:")
print("-" * 50)

if not detections:
    print("No damage detected.")
else:
    for i, detection in enumerate(detections, start=1):
        print(f"Damage #{i}")
        print(f"Class      : {detection['class_name']}")
        print(f"Confidence : {detection['confidence']:.2%}")
        print(f"BBox       : {detection['bbox']}")
        print("-" * 50)
        output_path = PROJECT_ROOT / "outputs" / "damage_detection_result.json"
output_path.parent.mkdir(parents=True, exist_ok=True)

with output_path.open("w", encoding="utf-8") as file:
    json.dump(detections, file, indent=4, ensure_ascii=False)

print(f"\nJSON result saved to: {output_path}")