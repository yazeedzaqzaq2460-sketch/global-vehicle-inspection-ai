from pathlib import Path
from typing import Any

from src.pipelines.inspection_pipeline import InspectionPipeline
from src.reasoning.damage_fusion_engine import DamageFusionEngine


class MultiViewInspectionPipeline:
    REQUIRED_VIEWS = {
        "front",
        "rear",
        "left",
        "right",
        "front_left",
        "front_right",
        "rear_left",
        "rear_right",
    }

    def __init__(self) -> None:
        self.single_view_pipeline = InspectionPipeline()

        self.damage_fusion_engine = DamageFusionEngine(
            severity_difference_threshold=10.0,
        )

    def inspect(
        self,
        images: dict[str, str],
    ) -> dict[str, Any]:
        unknown_views = set(images) - self.REQUIRED_VIEWS

        if unknown_views:
            raise ValueError(
                "Unsupported view names: "
                + ", ".join(sorted(unknown_views))
            )

        missing_views = sorted(
            self.REQUIRED_VIEWS - set(images)
        )

        view_results: dict[str, dict[str, Any]] = {}

        for view_name, image_path in images.items():
            path = Path(image_path)

            if not path.exists():
                raise FileNotFoundError(
                    f"Image for view '{view_name}' not found: {path}"
                )

            view_results[view_name] = (
                self.single_view_pipeline.inspect_image(
                    str(path)
                )
            )

        acceptable_views = [
            view_name
            for view_name, result in view_results.items()
            if result["image_quality"]["is_acceptable"]
        ]

        rejected_views = [
            view_name
            for view_name, result in view_results.items()
            if not result["image_quality"]["is_acceptable"]
        ]

        coverage_ratio = (
            len(set(images) & self.REQUIRED_VIEWS)
            / len(self.REQUIRED_VIEWS)
        )

        all_damages: list[dict[str, Any]] = []

        for view_name, result in view_results.items():
            for damage in result["damages"]:
                all_damages.append(
                    {
                        **damage,
                        "class": damage["class_name"],
                        "source_view": view_name,
                        "source_image": result["image_path"],
                    }
                )

        fused_damages = self.damage_fusion_engine.fuse(
            all_damages
        )

        highest_severity_score = max(
            (
                damage.get("severity_score", 0.0)
                for damage in fused_damages
            ),
            default=0.0,
        )

        return {
            "required_views": sorted(self.REQUIRED_VIEWS),
            "received_views": sorted(images),
            "missing_views": missing_views,
            "acceptable_views": sorted(acceptable_views),
            "rejected_views": sorted(rejected_views),
            "coverage_ratio": round(
                coverage_ratio,
                4,
            ),
            "coverage_percentage": round(
                coverage_ratio * 100.0,
                2,
            ),
            "view_results": view_results,
            "damages": all_damages,
            "fused_damages": fused_damages,
            "damage_count_before_fusion": len(
                all_damages
            ),
            "damage_count_after_fusion": len(
                fused_damages
            ),
            "highest_severity_score": round(
                highest_severity_score,
                2,
            ),
        }