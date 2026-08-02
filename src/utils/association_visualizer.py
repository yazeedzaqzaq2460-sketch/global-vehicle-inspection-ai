from pathlib import Path
from typing import Any

import cv2
import numpy as np


class AssociationVisualizer:
    @staticmethod
    def _resize_mask(
        mask: np.ndarray,
        image_shape: tuple[int, int],
    ) -> np.ndarray:
        height, width = image_shape

        resized = cv2.resize(
            mask.astype(np.float32),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )

        return resized > 0.5

    def save(
        self,
        image_path: str,
        damages: list[dict[str, Any]],
        vehicle_parts: list[dict[str, Any]],
        output_path: str = "outputs/association_debug.jpg",
    ) -> str:
        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(
                f"Unable to read image: {image_path}"
            )

        height, width = image.shape[:2]
        overlay = image.copy()

        # Vehicle parts: green
        for part in vehicle_parts:
            mask = part.get("mask")

            if isinstance(mask, np.ndarray):
                mask_binary = self._resize_mask(
                    mask,
                    (height, width),
                )

                overlay[mask_binary] = (
                    overlay[mask_binary] * 0.5
                    + np.array([0, 255, 0]) * 0.5
                ).astype(np.uint8)

            bbox = part.get("bbox", {})
            x1 = int(bbox.get("x1", 0))
            y1 = int(bbox.get("y1", 0))
            x2 = int(bbox.get("x2", 0))
            y2 = int(bbox.get("y2", 0))

            cv2.rectangle(
                overlay,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                overlay,
                f"PART: {part['class_name']}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

        # Damages: red
        for damage in damages:
            mask = damage.get("mask")

            if isinstance(mask, np.ndarray):
                mask_binary = self._resize_mask(
                    mask,
                    (height, width),
                )

                overlay[mask_binary] = (
                    overlay[mask_binary] * 0.5
                    + np.array([0, 0, 255]) * 0.5
                ).astype(np.uint8)

            bbox = damage.get("bbox", {})
            x1 = int(bbox.get("x1", 0))
            y1 = int(bbox.get("y1", 0))
            x2 = int(bbox.get("x2", 0))
            y2 = int(bbox.get("y2", 0))

            cv2.rectangle(
                overlay,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2,
            )

            cv2.putText(
                overlay,
                f"DAMAGE: {damage['class_name']}",
                (x1, min(y2 + 22, height - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
            )

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        if not cv2.imwrite(str(output), overlay):
            raise RuntimeError(
                f"Failed to save visualization: {output}"
            )

        return str(output)