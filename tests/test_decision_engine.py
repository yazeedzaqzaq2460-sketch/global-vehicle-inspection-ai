from src.core.decision_engine import DecisionEngine


def test_no_damage_returns_pass_recommendation():
    damages = []

    decision = DecisionEngine.evaluate(damages)

    assert decision == "PASS_RECOMMENDATION"


def test_all_minor_damages_return_pass_recommendation():
    damages = [
        {
            "class_name": "Scratch",
            "confidence": 0.55,
            "severity": "MINOR",
            "severity_score": 25.0,
            "vehicle_part": "front_bumper",
            "association_status": "CONFIRMED",
        }
    ]

    decision = DecisionEngine.evaluate(damages)

    assert decision == "PASS_RECOMMENDATION"


def test_moderate_damage_returns_human_review():
    damages = [
        {
            "class_name": "Dent",
            "confidence": 0.70,
            "severity": "MODERATE",
            "severity_score": 55.0,
            "vehicle_part": "hood",
            "association_status": "CONFIRMED",
        }
    ]

    decision = DecisionEngine.evaluate(damages)

    assert decision == "HUMAN_REVIEW_REQUIRED"


def test_severe_damage_returns_human_review():
    damages = [
        {
            "class_name": "Dent",
            "confidence": 0.85,
            "severity": "SEVERE",
            "severity_score": 72.0,
            "vehicle_part": "front_bumper",
            "association_status": "CONFIRMED",
        }
    ]

    decision = DecisionEngine.evaluate(damages)

    assert decision == "HUMAN_REVIEW_REQUIRED"


def test_high_severity_score_returns_fail_recommendation():
    damages = [
        {
            "class_name": "Lamp Broken",
            "confidence": 0.65,
            "severity": "SEVERE",
            "severity_score": 80.81,
            "vehicle_part": "front_bumper",
            "association_status": "CONFIRMED",
        }
    ]

    decision = DecisionEngine.evaluate(damages)

    assert decision == "FAIL_RECOMMENDATION"


def test_critical_damage_returns_fail_recommendation():
    damages = [
        {
            "class_name": "Broken Glass",
            "confidence": 0.92,
            "severity": "CRITICAL",
            "severity_score": 91.0,
            "vehicle_part": "front_glass",
            "association_status": "CONFIRMED",
        }
    ]

    decision = DecisionEngine.evaluate(damages)

    assert decision == "FAIL_RECOMMENDATION"


def test_severe_damage_on_critical_part_returns_fail():
    damages = [
        {
            "class_name": "Lamp Broken",
            "confidence": 0.78,
            "severity": "SEVERE",
            "severity_score": 74.0,
            "vehicle_part": "front_right_light",
            "association_status": "CONFIRMED",
        }
    ]

    decision = DecisionEngine.evaluate(damages)

    assert decision == "FAIL_RECOMMENDATION"


def test_all_unresolved_damages_return_insufficient_evidence():
    damages = [
        {
            "class_name": "Scratch",
            "confidence": 0.60,
            "severity": "MINOR",
            "severity_score": 28.0,
            "vehicle_part": None,
            "association_status": "UNRESOLVED",
        }
    ]

    decision = DecisionEngine.evaluate(damages)

    assert decision == "INSUFFICIENT_EVIDENCE"


def test_mixed_resolved_and_unresolved_returns_insufficient_evidence():
    damages = [
        {
            "class_name": "Scratch",
            "confidence": 0.65,
            "severity": "MINOR",
            "severity_score": 30.0,
            "vehicle_part": "front_bumper",
            "association_status": "CONFIRMED",
        },
        {
            "class_name": "Dent",
            "confidence": 0.55,
            "severity": "MINOR",
            "severity_score": 35.0,
            "vehicle_part": None,
            "association_status": "UNRESOLVED",
        },
    ]

    decision = DecisionEngine.evaluate(damages)

    assert decision == "INSUFFICIENT_EVIDENCE"


def test_four_moderate_or_higher_damages_return_fail():
    damages = [
        {
            "class_name": "Dent",
            "confidence": 0.70,
            "severity": "MODERATE",
            "severity_score": 50.0,
            "vehicle_part": "hood",
            "association_status": "CONFIRMED",
        }
        for _ in range(4)
    ]

    decision = DecisionEngine.evaluate(damages)

    assert decision == "FAIL_RECOMMENDATION"