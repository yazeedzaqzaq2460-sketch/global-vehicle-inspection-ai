class DecisionEngine:
    @staticmethod
    def evaluate(damages: list[dict]) -> str:
        highest_confidence = max(
            (damage["confidence"] for damage in damages),
            default=0.0,
        )

        if highest_confidence >= 0.80:
            return "fail"

        if damages:
            return "manual_review"

        return "pass"