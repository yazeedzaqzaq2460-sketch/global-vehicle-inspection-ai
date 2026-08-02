from typing import Any


class DecisionEngine:
    CRITICAL_PARTS = {
        "front_glass",
        "back_glass",
        "front_left_light",
        "front_right_light",
        "back_left_light",
        "back_right_light",
        "wheel",
    }

    @classmethod
    def evaluate(
        cls,
        damages: list[dict[str, Any]],
    ) -> str:
        if not damages:
            return "PASS_RECOMMENDATION"

        unresolved_count = sum(
            1
            for damage in damages
            if damage.get("association_status") == "UNRESOLVED"
        )

        if unresolved_count == len(damages):
            return "INSUFFICIENT_EVIDENCE"

        severities = [
            str(damage.get("severity", "MINOR")).upper()
            for damage in damages
        ]

        highest_severity_score = max(
            (
                float(damage.get("severity_score", 0.0))
                for damage in damages
            ),
            default=0.0,
        )

        critical_damages = [
            damage
            for damage in damages
            if str(damage.get("severity", "")).upper()
            == "CRITICAL"
        ]

        severe_damages = [
            damage
            for damage in damages
            if str(damage.get("severity", "")).upper()
            == "SEVERE"
        ]

        moderate_or_higher = [
            damage
            for damage in damages
            if str(damage.get("severity", "")).upper()
            in {"MODERATE", "SEVERE", "CRITICAL"}
        ]

        critical_part_damages = [
            damage
            for damage in damages
            if damage.get("vehicle_part") in cls.CRITICAL_PARTS
            and str(damage.get("severity", "")).upper()
            in {"SEVERE", "CRITICAL"}
        ]

        if critical_damages:
            return "FAIL_RECOMMENDATION"

        if highest_severity_score >= 80.0:
            return "FAIL_RECOMMENDATION"

        if critical_part_damages:
            return "FAIL_RECOMMENDATION"

        if len(moderate_or_higher) >= 4:
            return "FAIL_RECOMMENDATION"

        if severe_damages:
            return "HUMAN_REVIEW_REQUIRED"

        if "MODERATE" in severities:
            return "HUMAN_REVIEW_REQUIRED"

        if unresolved_count > 0:
            return "INSUFFICIENT_EVIDENCE"

        if all(
            severity == "MINOR"
            for severity in severities
        ):
            return "PASS_RECOMMENDATION"

        return "HUMAN_REVIEW_REQUIRED"