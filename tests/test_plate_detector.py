from src.plate_detection.plate_detector import PlateDetector


def test_plate_detector_returns_detection():
    detector = PlateDetector()

    result = detector.detect("test_images/plate_test.jpg")

    assert isinstance(result, dict)
    assert "bbox" in result
    assert "confidence" in result

    assert len(result["bbox"]) == 4
    assert 0.0 <= result["confidence"] <= 1.0