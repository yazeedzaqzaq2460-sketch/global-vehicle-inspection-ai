from math import hypot
from typing import Any

import cv2
import numpy as np


class AssociationEngine:
    def __init__(
        self,
        minimum_mask_overlap: float = 0.10,
        minimum_bbox_iou: float = 0.05,
        maximum_center_distance_ratio: float = 0.35,
    ) -> None:
        if not 0.0 <= minimum_mask_overlap <= 1.0:
            raise ValueError(
                "minimum_mask_overlap must be between 0 and 1."
            )

        if not 0.0 <= minimum_bbox_iou <= 1.0:
            raise ValueError(
                "minimum_bbox_iou must be between 0 and 1."
            )

        if maximum_center_distance_ratio <= 0.0:
            raise ValueError(
                "maximum_center_distance_ratio must be greater than 0."
            )

        self.minimum_mask_overlap = minimum_mask_overlap
        self.minimum_bbox_iou = minimum_bbox_iou
        self.maximum_center_distance_ratio = (
            maximum_center_distance_ratio
        )

    @staticmethod
    def _resize_mask(
        mask: np.ndarray,
        target_shape: tuple[int, int],
    ) -> np.ndarray:
        target_height, target_width = target_shape

        resized = cv2.resize(
            mask.astype(np.float32),
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST,
        )

        return resized > 0.5

    @classmethod
    def _calculate_mask_overlap(
        cls,
        damage_mask: np.ndarray,
        part_mask: np.ndarray,
    ) -> float:
        damage_binary = damage_mask > 0.5
        part_binary = part_mask > 0.5

        if damage_binary.shape != part_binary.shape:
            part_binary = cls._resize_mask(
                part_binary,
                damage_binary.shape,
            )

        damage_area = np.count_nonzero(damage_binary)

        if damage_area == 0:
            return 0.0

        intersection = np.count_nonzero(
            damage_binary & part_binary
        )

        return float(intersection / damage_area)

    @staticmethod
    def _calculate_bbox_iou(
        first_bbox: dict[str, float],
        second_bbox: dict[str, float],
    ) -> float:
        intersection_x1 = max(
            first_bbox["x1"],
            second_bbox["x1"],
        )
        intersection_y1 = max(
            first_bbox["y1"],
            second_bbox["y1"],
        )
        intersection_x2 = min(
            first_bbox["x2"],
            second_bbox["x2"],
        )
        intersection_y2 = min(
            first_bbox["y2"],
            second_bbox["y2"],
        )

        intersection_width = max(
            0.0,
            intersection_x2 - intersection_x1,
        )
        intersection_height = max(
            0.0,
            intersection_y2 - intersection_y1,
        )

        intersection_area = (
            intersection_width * intersection_height
        )

        first_area = max(
            0.0,
            first_bbox["x2"] - first_bbox["x1"],
        ) * max(
            0.0,
            first_bbox["y2"] - first_bbox["y1"],
        )

        second_area = max(
            0.0,
            second_bbox["x2"] - second_bbox["x1"],
        ) * max(
            0.0,
            second_bbox["y2"] - second_bbox["y1"],
        )

        union_area = (
            first_area
            + second_area
            - intersection_area
        )

        if union_area <= 0.0:
            return 0.0

        return float(intersection_area / union_area)

    @staticmethod
    def _calculate_center_distance_ratio(
        first_bbox: dict[str, float],
        second_bbox: dict[str, float],
    ) -> float:
        first_center_x = (
            first_bbox["x1"] + first_bbox["x2"]
        ) / 2.0
        first_center_y = (
            first_bbox["y1"] + first_bbox["y2"]
        ) / 2.0

        second_center_x = (
            second_bbox["x1"] + second_bbox["x2"]
        ) / 2.0
        second_center_y = (
            second_bbox["y1"] + second_bbox["y2"]
        ) / 2.0

        distance = hypot(
            first_center_x - second_center_x,
            first_center_y - second_center_y,
        )

        second_width = max(
            1.0,
            second_bbox["x2"] - second_bbox["x1"],
        )
        second_height = max(
            1.0,
            second_bbox["y2"] - second_bbox["y1"],
        )

        second_diagonal = hypot(
            second_width,
            second_height,
        )

        return float(distance / second_diagonal)

    def associate(
        self,
        damages: list[dict[str, Any]],
        vehicle_parts: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        associated_damages: list[dict[str, Any]] = []

        for damage in damages:
            best_part: dict[str, Any] | None = None
            association_status = "UNRESOLVED"
            association_method = None
            association_confidence = 0.0

            best_mask_overlap = 0.0
            best_bbox_iou = 0.0
            best_distance_ratio = float("inf")

            damage_mask = damage.get("mask")
            damage_bbox = damage.get("bbox")

            # 1. Confirmed association using mask overlap
            if damage_mask is not None:
                for part in vehicle_parts:
                    part_mask = part.get("mask")

                    if part_mask is None:
                        continue

                    mask_overlap = (
                        self._calculate_mask_overlap(
                            damage_mask,
                            part_mask,
                        )
                    )

                    if mask_overlap > best_mask_overlap:
                        best_mask_overlap = mask_overlap
                        best_part = part

                if (
                    best_part is not None
                    and best_mask_overlap
                    >= self.minimum_mask_overlap
                ):
                    association_status = "CONFIRMED"
                    association_method = "mask_overlap"
                    association_confidence = (
                        best_mask_overlap
                    )

            # 2. Inferred association using bbox IoU
            if (
                association_status == "UNRESOLVED"
                and damage_bbox is not None
            ):
                best_part = None

                for part in vehicle_parts:
                    part_bbox = part.get("bbox")

                    if part_bbox is None:
                        continue

                    bbox_iou = self._calculate_bbox_iou(
                        damage_bbox,
                        part_bbox,
                    )

                    if bbox_iou > best_bbox_iou:
                        best_bbox_iou = bbox_iou
                        best_part = part

                if (
                    best_part is not None
                    and best_bbox_iou
                    >= self.minimum_bbox_iou
                ):
                    association_status = "INFERRED"
                    association_method = "bbox_iou"
                    association_confidence = best_bbox_iou

            # 3. Likely association using center distance
            if (
                association_status == "UNRESOLVED"
                and damage_bbox is not None
            ):
                best_part = None

                for part in vehicle_parts:
                    part_bbox = part.get("bbox")

                    if part_bbox is None:
                        continue

                    distance_ratio = (
                        self._calculate_center_distance_ratio(
                            damage_bbox,
                            part_bbox,
                        )
                    )

                    if distance_ratio < best_distance_ratio:
                        best_distance_ratio = distance_ratio
                        best_part = part

                if (
                    best_part is not None
                    and best_distance_ratio
                    <= self.maximum_center_distance_ratio
                ):
                    association_status = "LIKELY"
                    association_method = "center_distance"

                    association_confidence = max(
                        0.0,
                        1.0 - best_distance_ratio,
                    )

            associated_damages.append(
                {
                    **damage,
                    "vehicle_part": (
                        best_part["class_name"]
                        if best_part is not None
                        and association_status
                        != "UNRESOLVED"
                        else None
                    ),
                    "part_class_id": (
                        best_part["class_id"]
                        if best_part is not None
                        and association_status
                        != "UNRESOLVED"
                        else None
                    ),
                    "part_confidence": (
                        best_part["confidence"]
                        if best_part is not None
                        and association_status
                        != "UNRESOLVED"
                        else None
                    ),
                    "mask_overlap_score": round(
                        best_mask_overlap,
                        4,
                    ),
                    "bbox_iou_score": round(
                        best_bbox_iou,
                        4,
                    ),
                    "center_distance_ratio": (
                        round(best_distance_ratio, 4)
                        if best_distance_ratio
                        != float("inf")
                        else None
                    ),
                    "association_status": (
                        association_status
                    ),
                    "association_method": (
                        association_method
                    ),
                    "association_confidence": round(
                        association_confidence,
                        4,
                    ),
                    # Kept temporarily for compatibility
                    "overlap_score": round(
                        best_mask_overlap,
                        4,
                    ),
                }
            )

        return associated_damages