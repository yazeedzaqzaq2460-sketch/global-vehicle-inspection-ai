from pathlib import Path
from typing import Any

import cv2
import numpy as np


class ImageQualityEngine:
    def __init__(
        self,
        blur_threshold: float = 100.0,
        minimum_brightness: float = 45.0,
        maximum_brightness: float = 220.0,
        minimum_contrast: float = 30.0,
    ) -> None:
        self.blur_threshold = blur_threshold
        self.minimum_brightness = minimum_brightness
        self.maximum_brightness = maximum_brightness
        self.minimum_contrast = minimum_contrast

    def evaluate(self, image_path: str) -> dict[str, Any]:
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Input image not found: {path}"
            )

        image = cv2.imread(str(path))

        if image is None:
            raise ValueError(
                f"Unable to read image: {path}"
            )

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        blur_score = float(
            cv2.Laplacian(
                gray,
                cv2.CV_64F,
            ).var()
        )

        brightness = float(
            np.mean(gray)
        )

        contrast = float(
            np.std(gray)
        )

        issues: list[str] = []

        if blur_score < self.blur_threshold:
            issues.append("BLURRY_IMAGE")

        if brightness < self.minimum_brightness:
            issues.append("IMAGE_TOO_DARK")

        if brightness > self.maximum_brightness:
            issues.append("IMAGE_TOO_BRIGHT")

        if contrast < self.minimum_contrast:
            issues.append("LOW_CONTRAST")

        is_acceptable = len(issues) == 0

        return {
            "is_acceptable": is_acceptable,
            "blur_score": round(blur_score, 2),
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "issues": issues,
            "recommendation": (
                "ACCEPT_IMAGE"
                if is_acceptable
                else "RECAPTURE_REQUIRED"
            ),
        }