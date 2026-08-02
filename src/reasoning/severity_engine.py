from typing import Any

import numpy as np


class SeverityEngine:
    DAMAGE_BASE_SCORES = {
        "Scratch": 20,
        "Dent": 40,
        "Crack": 55,
        "Broken Glass": 75,
        "Lamp Broken": 70,
        "Tire Flat": 80,
        "Wreck": 95,
    }

    CRITICAL_PARTS = {
        "front_glass",
        "back_glass",
        "front_left_light",
        "front_right_light",
        "back_left_light",
        "back_right_light",
        "wheel",
    }

    def __init__(
        self,
        moderate_threshold: float = 0.10,
        severe_threshold: float = 0.30,
        critical_threshold: float = 0.60,
    ) -> None:
        if not (
            0.0
            < moderate_threshold
            < severe_threshold
            < critical_threshold
            <= 1.0
        ):
            raise ValueError(
                "Severity thresholds must be ordered between 0 and 1."
            )

        self.moderate_threshold = moderate_threshold
        self.severe_threshold = severe_threshold
        self.critical_threshold = critical_threshold

    @staticmethod
    def _calculate_mask_area(mask: np.ndarray | None) -> int:
        if mask is None:
            return 0

        return int(np.count_nonzero(mask > 0.5))

    def _calculate_area_ratio(
        self,
        damage_mask: np.ndarray | None,
        part_mask: np.ndarray | None,
    ) -> float:
        damage_area = self._calculate_mask_area(damage_mask)
        part_area = self._calculate_mask_area(part_mask)

        if damage_area == 0 or part_area == 0:
            return 0.0

        return min(float(damage_area / part_area), 1.0)

    def _classify_severity(
        self,
        score: float,
    ) -> str:
        if score >= 85:
            return "CRITICAL"

        if score >= 65:
            return "SEVERE"

        if score >= 40:
            return "MODERATE"

        return "MINOR"

    def evaluate_damage(
        self,
        damage: dict[str, Any],
        vehicle_parts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        damage_type = damage.get("class_name", "")
        vehicle_part = damage.get("vehicle_part")

        matching_part = next(
            (
                part
                for part in vehicle_parts
                if part.get("class_name") == vehicle_part
            ),
            None,
        )

        area_ratio = self._calculate_area_ratio(
            damage.get("mask"),
            (
                matching_part.get("mask")
                if matching_part is not None
                else None
            ),
        )

        base_score = self.DAMAGE_BASE_SCORES.get(
            damage_type,
            30,
        )

        area_bonus = area_ratio * 35.0

        critical_part_bonus = (
            15.0
            if vehicle_part in self.CRITICAL_PARTS
            else 0.0
        )

        confidence = float(
            damage.get("confidence", 0.0)
        )

        confidence_adjustment = confidence * 10.0

        severity_score = min(
            base_score
            + area_bonus
            + critical_part_bonus
            + confidence_adjustment,
            100.0,
        )

        severity = self._classify_severity(
            severity_score
        )

        recommendation = "REPAIR"

        if severity == "CRITICAL":
            recommendation = "REPLACE_OR_IMMEDIATE_REVIEW"
        elif severity == "SEVERE":
            recommendation = "REPAIR_OR_REPLACE"
        elif severity == "MODERATE":
            recommendation = "REPAIR"
        else:
            recommendation = "MINOR_REPAIR_OR_MONITOR"

        return {
            **damage,
            "damage_area_ratio": round(
                area_ratio,
                4,
            ),
            "severity_score": round(
                severity_score,
                2,
            ),
            "severity": severity,
            "repair_recommendation": recommendation,
        }

    def evaluate(
        self,
        damages: list[dict[str, Any]],
        vehicle_parts: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            self.evaluate_damage(
                damage,
                vehicle_parts,
            )
            for damage in damages
        ]