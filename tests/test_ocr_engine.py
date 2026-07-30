from src.ocr.ocr_engine import OCREngine


def test_ocr_engine_returns_expected_structure():
    engine = OCREngine()

    result = engine.extract("test_images/plate_test.jpg")

    print(result)

    assert isinstance(result, dict)
    assert "license_plate" in result
    assert result["license_plate"] is None or isinstance(result["license_plate"], str)
    assert "vin" in result
    assert "engine_number" in result