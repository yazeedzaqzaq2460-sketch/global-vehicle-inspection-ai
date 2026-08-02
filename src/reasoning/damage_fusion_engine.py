from typing import Any


class DamageFusionEngine:
    def __init__(
        self,
        severity_difference_threshold: float = 10.0,
    ) -> None:
        self.severity_difference_threshold = (
            severity_difference_threshold
        )

    def fuse(
        self,
        damages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        fused: list[dict[str, Any]] = []

        for damage in damages:
            matched = False

            for existing in fused:

                same_type = (
                    damage["class"]
                    == existing["class"]
                )

                same_part = (
                    damage.get("vehicle_part")
                    == existing.get("vehicle_part")
                )

                severity_difference = abs(
                    damage.get("severity_score", 0.0)
                    - existing.get("severity_score", 0.0)
                )

                if (
                    same_type
                    and same_part
                    and severity_difference
                    <= self.severity_difference_threshold
                ):
                    existing.setdefault(
                        "source_views",
                        [],
                    ).append(
                        damage["source_view"]
                    )

                    existing["confidence"] = max(
                        existing["confidence"],
                        damage["confidence"],
                    )

                    existing["severity_score"] = max(
                        existing["severity_score"],
                        damage["severity_score"],
                    )

                    matched = True
                    break

            if not matched:

                new_damage = damage.copy()

                new_damage["source_views"] = [
                    damage["source_view"]
                ]

                fused.append(new_damage)

        return fused