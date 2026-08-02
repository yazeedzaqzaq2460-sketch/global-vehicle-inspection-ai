from src.reasoning.damage_fusion_engine import DamageFusionEngine


def test_same_damage_is_fused():
    engine = DamageFusionEngine()

    damages = [
        {
            "class": "Dent",
            "vehicle_part": "front_bumper",
            "severity_score": 55.0,
            "confidence": 0.81,
            "source_view": "front_left",
        },
        {
            "class": "Dent",
            "vehicle_part": "front_bumper",
            "severity_score": 58.0,
            "confidence": 0.92,
            "source_view": "left",
        },
    ]

    fused = engine.fuse(damages)

    assert len(fused) == 1
    assert fused[0]["confidence"] == 0.92
    assert fused[0]["severity_score"] == 58.0
    assert sorted(fused[0]["source_views"]) == [
        "front_left",
        "left",
    ]


def test_different_damage_types_are_not_fused():
    engine = DamageFusionEngine()

    damages = [
        {
            "class": "Dent",
            "vehicle_part": "front_bumper",
            "severity_score": 50.0,
            "confidence": 0.8,
            "source_view": "front",
        },
        {
            "class": "Scratch",
            "vehicle_part": "front_bumper",
            "severity_score": 50.0,
            "confidence": 0.8,
            "source_view": "front_left",
        },
    ]

    fused = engine.fuse(damages)

    assert len(fused) == 2


def test_different_parts_are_not_fused():
    engine = DamageFusionEngine()

    damages = [
        {
            "class": "Dent",
            "vehicle_part": "hood",
            "severity_score": 50.0,
            "confidence": 0.8,
            "source_view": "front",
        },
        {
            "class": "Dent",
            "vehicle_part": "front_bumper",
            "severity_score": 50.0,
            "confidence": 0.8,
            "source_view": "front_left",
        },
    ]

    fused = engine.fuse(damages)

    assert len(fused) == 2


def test_large_severity_difference_is_not_fused():
    engine = DamageFusionEngine(
        severity_difference_threshold=10.0,
    )

    damages = [
        {
            "class": "Dent",
            "vehicle_part": "front_bumper",
            "severity_score": 20.0,
            "confidence": 0.8,
            "source_view": "front",
        },
        {
            "class": "Dent",
            "vehicle_part": "front_bumper",
            "severity_score": 60.0,
            "confidence": 0.9,
            "source_view": "left",
        },
    ]

    fused = engine.fuse(damages)

    assert len(fused) == 2


def test_empty_input_returns_empty_list():
    engine = DamageFusionEngine()

    fused = engine.fuse([])

    assert fused == []