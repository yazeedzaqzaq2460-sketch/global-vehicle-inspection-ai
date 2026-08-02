from pathlib import Path

from src.pipelines.inspection_pipeline import InspectionPipeline
from src.utils.association_visualizer import AssociationVisualizer


VALID_DECISIONS = {
    "PASS_RECOMMENDATION",
    "HUMAN_REVIEW_REQUIRED",
    "FAIL_RECOMMENDATION",
    "RETEST_REQUIRED",
    "RECAPTURE_REQUIRED",
    "INSUFFICIENT_EVIDENCE",
}


def test_pipeline_returns_expected_structure():
    image_path = "test_images/association_test.jpg"

    pipeline = InspectionPipeline()

    result = pipeline.inspect_image(image_path)

    print("\n")
    print("=" * 80)
    print("FINAL PIPELINE RESULT")
    print("=" * 80)

    print(f"\nImage: {result['image_path']}")
    print(f"Decision: {result['decision']}")
    print(f"Damage Count: {result['damage_count']}")
    print(f"Highest Confidence: {result['highest_confidence']}")
    print(
        f"Highest Severity Score: "
        f"{result['highest_severity_score']}"
    )

    print("\nDetected Vehicle Parts")
    print("-" * 80)

    if not result["vehicle_parts"]:
        print("No vehicle parts detected.")
    else:
        for index, part in enumerate(
            result["vehicle_parts"],
            start=1,
        ):
            print(f"\nPart #{index}")
            print(f"Name       : {part['class_name']}")
            print(f"Confidence : {part['confidence']:.4f}")
            print(f"BBox       : {part['bbox']}")
            print(
                "Mask       : "
                f"{'Available' if part.get('mask') is not None else 'None'}"
            )

    print("\nAssociated Damages")
    print("-" * 80)

    if not result["damages"]:
        print("No damages detected.")
    else:
        for index, damage in enumerate(
            result["damages"],
            start=1,
        ):
            print(f"\nDamage #{index}")
            print(
                f"Type                  : "
                f"{damage['class_name']}"
            )
            print(
                f"Confidence            : "
                f"{damage['confidence']:.4f}"
            )
            print(f"BBox                  : {damage['bbox']}")
            print(
                "Mask                  : "
                f"{'Available' if damage.get('mask') is not None else 'None'}"
            )

            print(
                f"Vehicle Part          : "
                f"{damage.get('vehicle_part')}"
            )
            print(
                f"Part Class ID         : "
                f"{damage.get('part_class_id')}"
            )
            print(
                f"Part Confidence       : "
                f"{damage.get('part_confidence')}"
            )

            print(
                f"Association Status    : "
                f"{damage.get('association_status')}"
            )
            print(
                f"Association Method    : "
                f"{damage.get('association_method')}"
            )
            print(
                f"Association Confidence: "
                f"{damage.get('association_confidence')}"
            )

            print(
                f"Mask Overlap          : "
                f"{damage.get('mask_overlap_score')}"
            )
            print(
                f"BBox IoU              : "
                f"{damage.get('bbox_iou_score')}"
            )
            print(
                f"Center Distance Ratio : "
                f"{damage.get('center_distance_ratio')}"
            )

            print(
                f"Damage Area Ratio     : "
                f"{damage.get('damage_area_ratio')}"
            )
            print(
                f"Severity Score        : "
                f"{damage.get('severity_score')}"
            )
            print(
                f"Severity              : "
                f"{damage.get('severity')}"
            )
            print(
                f"Recommendation        : "
                f"{damage.get('repair_recommendation')}"
            )

    print("\nVehicle Data")
    print("-" * 80)

    for key, value in result["vehicle_data"].items():
        print(f"{key}: {value}")

    output_path = AssociationVisualizer().save(
        image_path=image_path,
        damages=result["damages"],
        vehicle_parts=result["vehicle_parts"],
    )

    print("\nAssociation Visualization")
    print("-" * 80)
    print(f"Saved to: {output_path}")

    print("=" * 80)
    print()

    # ===== Assertions =====

    assert "image_path" in result
    assert "decision" in result
    assert "damage_count" in result
    assert "highest_confidence" in result
    assert "highest_severity_score" in result
    assert "damages" in result
    assert "vehicle_parts" in result
    assert "vehicle_data" in result

    assert result["image_path"] == image_path
    assert result["decision"] in VALID_DECISIONS

    assert isinstance(result["damage_count"], int)
    assert isinstance(result["highest_confidence"], float)
    assert isinstance(result["highest_severity_score"], float)
    assert isinstance(result["damages"], list)
    assert isinstance(result["vehicle_parts"], list)
    assert isinstance(result["vehicle_data"], dict)

    assert Path(output_path) == Path(
        "outputs/association_debug.jpg"
    )

    for part in result["vehicle_parts"]:
        assert "class_id" in part
        assert "class_name" in part
        assert "confidence" in part
        assert "bbox" in part
        assert "mask" in part

    for damage in result["damages"]:
        assert "class_id" in damage
        assert "class_name" in damage
        assert "confidence" in damage
        assert "bbox" in damage
        assert "mask" in damage

        assert "vehicle_part" in damage
        assert "part_class_id" in damage
        assert "part_confidence" in damage

        assert "association_status" in damage
        assert "association_method" in damage
        assert "association_confidence" in damage
        assert "mask_overlap_score" in damage
        assert "bbox_iou_score" in damage
        assert "center_distance_ratio" in damage

        assert "damage_area_ratio" in damage
        assert "severity_score" in damage
        assert "severity" in damage
        assert "repair_recommendation" in damage

        assert damage["severity"] in {
            "MINOR",
            "MODERATE",
            "SEVERE",
            "CRITICAL",
        }

        assert 0.0 <= damage["severity_score"] <= 100.0
        assert 0.0 <= damage["damage_area_ratio"] <= 1.0