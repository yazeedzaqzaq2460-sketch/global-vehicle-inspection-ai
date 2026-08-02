import numpy as np

from src.reasoning.severity_engine import SeverityEngine


def create_mask(width=100, height=100):
    return np.ones((height, width), dtype=np.float32)


def test_minor_scratch():
    engine = SeverityEngine()

    part_mask = create_mask(100, 100)

    damage_mask = np.zeros((100, 100), dtype=np.float32)
    damage_mask[10:20, 10:20] = 1

    part = {
        "class_name": "front_bumper",
        "mask": part_mask,
    }

    damage = {
        "class_name": "Scratch",
        "mask": damage_mask,
        "confidence": 0.80,
        "vehicle_part": "front_bumper",
    }

    result = engine.evaluate_damage(damage, [part])

    assert result["severity"] == "MINOR"


def test_severe_broken_glass():
    engine = SeverityEngine()

    part_mask = create_mask(100, 100)

    damage_mask = np.zeros((100, 100), dtype=np.float32)
    damage_mask[10:90, 10:90] = 1

    part = {
        "class_name": "front_glass",
        "mask": part_mask,
    }

    damage = {
        "class_name": "Broken Glass",
        "mask": damage_mask,
        "confidence": 0.95,
        "vehicle_part": "front_glass",
    }

    result = engine.evaluate_damage(damage, [part])

    assert result["severity"] in ["SEVERE", "CRITICAL"]


def test_unknown_damage_type():
    engine = SeverityEngine()

    part_mask = create_mask()

    damage_mask = np.zeros((100, 100), dtype=np.float32)
    damage_mask[20:30, 20:30] = 1

    part = {
        "class_name": "hood",
        "mask": part_mask,
    }

    damage = {
        "class_name": "Unknown",
        "mask": damage_mask,
        "confidence": 0.5,
        "vehicle_part": "hood",
    }

    result = engine.evaluate_damage(damage, [part])

    assert "severity_score" in result
    assert result["severity"] in [
        "MINOR",
        "MODERATE",
        "SEVERE",
        "CRITICAL",
    ]


def test_batch_processing():
    engine = SeverityEngine()

    part = {
        "class_name": "front_bumper",
        "mask": create_mask(),
    }

    damages = [
        {
            "class_name": "Scratch",
            "mask": create_mask(10, 10),
            "confidence": 0.7,
            "vehicle_part": "front_bumper",
        },
        {
            "class_name": "Dent",
            "mask": create_mask(30, 30),
            "confidence": 0.8,
            "vehicle_part": "front_bumper",
        },
    ]

    results = engine.evaluate(damages, [part])

    assert len(results) == 2