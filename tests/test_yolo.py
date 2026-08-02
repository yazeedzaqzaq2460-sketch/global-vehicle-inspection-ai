import json
from pathlib import Path

import numpy as np

from src.detection.damage_detector import DamageDetector


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_IMAGE = PROJECT_ROOT / "test_images" / "car_damage.jpg"
MODEL_PATH = PROJECT_ROOT / "models" / "yolo" / "best.pt"


def make_json_serializable(
    detections: list[dict],
) -> list[dict]:
    serializable_detections = []

    for detection in detections:
        serialized = {
            key: value
            for key, value in detection.items()
            if key != "mask"
        }

        mask = detection.get("mask")

        serialized["mask_available"] = isinstance(
            mask,
            np.ndarray,
        )

        if isinstance(mask, np.ndarray):
            serialized["mask_shape"] = list(mask.shape)

        serializable_detections.append(serialized)

    return serializable_detections


def test_damage_detector_returns_expected_structure():
    detector = DamageDetector(str(MODEL_PATH))

    detections = detector.predict(str(TEST_IMAGE))

    assert isinstance(detections, list)

    for detection in detections:
        assert "class_id" in detection
        assert "class_name" in detection
        assert "confidence" in detection
        assert "bbox" in detection
        assert "mask" in detection

        assert isinstance(detection["class_id"], int)
        assert isinstance(detection["class_name"], str)
        assert 0.0 <= detection["confidence"] <= 1.0

        assert set(detection["bbox"].keys()) == {
            "x1",
            "y1",
            "x2",
            "y2",
        }

        assert (
            detection["mask"] is None
            or isinstance(detection["mask"], np.ndarray)
        )


def test_damage_results_can_be_saved_as_json():
    detector = DamageDetector(str(MODEL_PATH))

    detections = detector.predict(str(TEST_IMAGE))
    serializable = make_json_serializable(detections)

    output_path = (
        PROJECT_ROOT
        / "outputs"
        / "damage_detection_result.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            serializable,
            file,
            indent=4,
            ensure_ascii=False,
        )

    assert output_path.exists()