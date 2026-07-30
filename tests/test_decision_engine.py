from src.core.decision_engine import DecisionEngine


def test_no_damage_returns_pass():
    damages = []

    decision = DecisionEngine.evaluate(damages)

    assert decision == "pass"


def test_low_confidence_damage_returns_manual_review():
    damages = [
        {
            "class_name": "Scratch",
            "confidence": 0.55,
        }
    ]

    decision = DecisionEngine.evaluate(damages)

    assert decision == "manual_review"


def test_high_confidence_damage_returns_fail():
    damages = [
        {
            "class_name": "Dent",
            "confidence": 0.91,
        }
    ]

    decision = DecisionEngine.evaluate(damages)

    assert decision == "fail"