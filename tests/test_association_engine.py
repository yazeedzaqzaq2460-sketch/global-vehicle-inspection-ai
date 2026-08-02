import numpy as np

from src.reasoning.association_engine import AssociationEngine


def test_confirmed_association_uses_mask_overlap():
    engine = AssociationEngine(
        minimum_mask_overlap=0.10,
        minimum_bbox_iou=0.05,
        maximum_center_distance_ratio=0.35,
    )

    damage_mask = np.zeros((100, 100), dtype=np.float32)
    damage_mask[20:60, 20:60] = 1.0

    hood_mask = np.zeros((100, 100), dtype=np.float32)
    hood_mask[10:70, 10:70] = 1.0

    damages = [
        {
            "class_id": 1,
            "class_name": "Dent",
            "confidence": 0.90,
            "bbox": {
                "x1": 20.0,
                "y1": 20.0,
                "x2": 60.0,
                "y2": 60.0,
            },
            "mask": damage_mask,
        }
    ]

    vehicle_parts = [
        {
            "class_id": 16,
            "class_name": "hood",
            "confidence": 0.95,
            "bbox": {
                "x1": 10.0,
                "y1": 10.0,
                "x2": 70.0,
                "y2": 70.0,
            },
            "mask": hood_mask,
        }
    ]

    result = engine.associate(
        damages=damages,
        vehicle_parts=vehicle_parts,
    )

    associated_damage = result[0]

    assert associated_damage["vehicle_part"] == "hood"
    assert associated_damage["part_class_id"] == 16
    assert associated_damage["part_confidence"] == 0.95
    assert associated_damage["association_status"] == "CONFIRMED"
    assert associated_damage["association_method"] == "mask_overlap"
    assert associated_damage["mask_overlap_score"] == 1.0
    assert associated_damage["association_confidence"] == 1.0


def test_inferred_association_uses_bbox_iou():
    engine = AssociationEngine(
        minimum_mask_overlap=0.50,
        minimum_bbox_iou=0.05,
        maximum_center_distance_ratio=0.35,
    )

    damage_mask = np.zeros((100, 100), dtype=np.float32)
    damage_mask[70:90, 70:90] = 1.0

    part_mask = np.zeros((100, 100), dtype=np.float32)
    part_mask[0:20, 0:20] = 1.0

    damages = [
        {
            "class_id": 2,
            "class_name": "Scratch",
            "confidence": 0.82,
            "bbox": {
                "x1": 20.0,
                "y1": 20.0,
                "x2": 80.0,
                "y2": 80.0,
            },
            "mask": damage_mask,
        }
    ]

    vehicle_parts = [
        {
            "class_id": 8,
            "class_name": "front_bumper",
            "confidence": 0.88,
            "bbox": {
                "x1": 10.0,
                "y1": 10.0,
                "x2": 90.0,
                "y2": 50.0,
            },
            "mask": part_mask,
        }
    ]

    result = engine.associate(
        damages=damages,
        vehicle_parts=vehicle_parts,
    )

    associated_damage = result[0]

    assert associated_damage["vehicle_part"] == "front_bumper"
    assert associated_damage["association_status"] == "INFERRED"
    assert associated_damage["association_method"] == "bbox_iou"
    assert associated_damage["bbox_iou_score"] >= 0.05
    assert associated_damage["association_confidence"] > 0.0


def test_likely_association_uses_center_distance():
    engine = AssociationEngine(
        minimum_mask_overlap=0.50,
        minimum_bbox_iou=0.50,
        maximum_center_distance_ratio=0.60,
    )

    damages = [
        {
            "class_id": 1,
            "class_name": "Dent",
            "confidence": 0.80,
            "bbox": {
                "x1": 50.0,
                "y1": 50.0,
                "x2": 70.0,
                "y2": 70.0,
            },
            "mask": None,
        }
    ]

    vehicle_parts = [
        {
            "class_id": 16,
            "class_name": "hood",
            "confidence": 0.90,
            "bbox": {
                "x1": 0.0,
                "y1": 0.0,
                "x2": 100.0,
                "y2": 40.0,
            },
            "mask": None,
        }
    ]

    result = engine.associate(
        damages=damages,
        vehicle_parts=vehicle_parts,
    )

    associated_damage = result[0]

    assert associated_damage["vehicle_part"] == "hood"
    assert associated_damage["association_status"] == "LIKELY"
    assert associated_damage["association_method"] == "center_distance"
    assert associated_damage["center_distance_ratio"] is not None
    assert associated_damage["association_confidence"] > 0.0


def test_unresolved_association_returns_none():
    engine = AssociationEngine(
        minimum_mask_overlap=0.50,
        minimum_bbox_iou=0.50,
        maximum_center_distance_ratio=0.10,
    )

    damages = [
        {
            "class_id": 6,
            "class_name": "Wreck",
            "confidence": 0.92,
            "bbox": {
                "x1": 0.0,
                "y1": 0.0,
                "x2": 10.0,
                "y2": 10.0,
            },
            "mask": None,
        }
    ]

    vehicle_parts = [
        {
            "class_id": 22,
            "class_name": "wheel",
            "confidence": 0.85,
            "bbox": {
                "x1": 90.0,
                "y1": 90.0,
                "x2": 100.0,
                "y2": 100.0,
            },
            "mask": None,
        }
    ]

    result = engine.associate(
        damages=damages,
        vehicle_parts=vehicle_parts,
    )

    associated_damage = result[0]

    assert associated_damage["vehicle_part"] is None
    assert associated_damage["part_class_id"] is None
    assert associated_damage["part_confidence"] is None
    assert associated_damage["association_status"] == "UNRESOLVED"
    assert associated_damage["association_method"] is None
    assert associated_damage["association_confidence"] == 0.0


def test_different_mask_sizes_are_supported():
    engine = AssociationEngine()

    damage_mask = np.ones((50, 50), dtype=np.float32)
    part_mask = np.ones((100, 100), dtype=np.float32)

    damages = [
        {
            "class_id": 6,
            "class_name": "Wreck",
            "confidence": 0.92,
            "bbox": {
                "x1": 0.0,
                "y1": 0.0,
                "x2": 50.0,
                "y2": 50.0,
            },
            "mask": damage_mask,
        }
    ]

    vehicle_parts = [
        {
            "class_id": 8,
            "class_name": "front_bumper",
            "confidence": 0.89,
            "bbox": {
                "x1": 0.0,
                "y1": 0.0,
                "x2": 100.0,
                "y2": 100.0,
            },
            "mask": part_mask,
        }
    ]

    result = engine.associate(
        damages=damages,
        vehicle_parts=vehicle_parts,
    )

    assert result[0]["vehicle_part"] == "front_bumper"
    assert result[0]["association_status"] == "CONFIRMED"
    assert result[0]["mask_overlap_score"] == 1.0