import re

import easyocr

from src.plate_detection.plate_detector import PlateDetector
from src.utils.image_preprocessor import ImagePreprocessor


class OCREngine:
    def __init__(self):
        self.engine_name = "EasyOCR"

        self.reader = easyocr.Reader(
            ["en"],
            gpu=True,
        )

        self.plate_detector = PlateDetector()
        self.preprocessor = ImagePreprocessor()

        print(f"OCR Engine initialized ({self.engine_name}).")

    def extract_license_plate(self, image_path: str):
        processed_image = self.preprocessor.preprocess_plate(
            image_path
        )

        results = self.reader.readtext(processed_image)

        ignored_words = {
            "JORDAN",
            "JOR",
            "HKJ",
        }

        detected_parts = []

        for _, text, confidence in results:
            cleaned_text = re.sub(
                r"[^A-Z0-9\u0600-\u06FF]",
                "",
                text.strip().upper(),
            )

            if confidence < 0.50:
                continue

            if cleaned_text in ignored_words:
                continue

            if cleaned_text == "":
                continue

            detected_parts.append(cleaned_text)

        if not detected_parts:
            return None

        return " ".join(detected_parts)

    def extract_vin(self, image_path: str):
        return None

    def extract_engine_number(self, image_path: str):
        return None

    def extract(self, image_path: str) -> dict:
        plate_detection = self.plate_detector.detect(image_path)

        if plate_detection is None:
            license_plate = None
        else:
            cropped_image_path = str(
                plate_detection["cropped_image"]
            )

            license_plate = self.extract_license_plate(
                cropped_image_path
            )

        return {
            "license_plate": license_plate,
            "vin": self.extract_vin(image_path),
            "engine_number": self.extract_engine_number(image_path),
        }