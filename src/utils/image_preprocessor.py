from pathlib import Path

import cv2
import numpy as np


class ImagePreprocessor:
    def preprocess_plate(self, image_path: str):
        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(
                f"Cannot read image: {image_path}"
            )

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        denoised = cv2.fastNlMeansDenoising(
            gray,
            None,
            h=10,
            templateWindowSize=7,
            searchWindowSize=21,
        )

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )

        enhanced = clahe.apply(denoised)

        binary = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            10,
        )

        sharpening_kernel = np.array(
            [
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0],
            ],
            dtype=np.float32,
        )

        sharpened = cv2.filter2D(
            binary,
            -1,
            sharpening_kernel,
        )

        output_dir = Path("outputs/preprocessed")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / "processed_plate.jpg"

        saved = cv2.imwrite(
            str(output_path),
            sharpened,
        )

        if not saved:
            raise RuntimeError(
                f"Failed to save processed image: {output_path}"
            )

        return sharpened