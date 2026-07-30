from src.pipelines.inspection_pipeline import InspectionPipeline


def test_pipeline_returns_expected_structure():
    pipeline = InspectionPipeline()

    result = pipeline.inspect_image("test_images/car_damage.jpg")

    assert "image_path" in result
    assert "decision" in result
    assert "damage_count" in result
    assert "highest_confidence" in result
    assert "damages" in result
    assert "vehicle_data" in result

    assert result["image_path"] == "test_images/car_damage.jpg"
    assert result["decision"] in {"pass", "manual_review", "fail"}
    assert isinstance(result["damage_count"], int)
    assert isinstance(result["highest_confidence"], float)
    assert isinstance(result["damages"], list)
    assert isinstance(result["vehicle_data"], dict)