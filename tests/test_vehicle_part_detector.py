from src.vehicle_parts.vehicle_part_detector import VehiclePartDetector


def test_vehicle_part_detector_returns_expected_structure():
    detector = VehiclePartDetector()

    result = detector.detect(
        "test_images/car_damage.jpg"
    )

    assert isinstance(result, list)

    if result:
        first_detection = result[0]

        assert "class_id" in first_detection
        assert "class_name" in first_detection
        assert "confidence" in first_detection
        assert "bbox" in first_detection
        assert "mask" in first_detection

        assert isinstance(first_detection["class_id"], int)
        assert isinstance(first_detection["class_name"], str)
        assert 0.0 <= first_detection["confidence"] <= 1.0

        assert set(first_detection["bbox"].keys()) == {
            "x1",
            "y1",
            "x2",
            "y2",
        }