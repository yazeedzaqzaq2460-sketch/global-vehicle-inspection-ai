from pathlib import Path

import cv2
import numpy as np

from src.pipelines.multi_view_pipeline import MultiViewInspectionPipeline


TEST_IMAGE_DIR = Path("outputs/multi_view_tests")


def _create_test_image(filename: str) -> str:
    TEST_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    image = np.full((400, 600, 3), 120, dtype=np.uint8)

    cv2.rectangle(image, (80, 120), (520, 300), (255, 255, 255), 4)
    cv2.putText(
        image,
        filename,
        (70, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
    )

    path = TEST_IMAGE_DIR / filename
    cv2.imwrite(str(path), image)

    return str(path)


def test_missing_views_are_reported():
    pipeline = MultiViewInspectionPipeline()

    images = {
        "front": _create_test_image("front.jpg"),
        "rear": _create_test_image("rear.jpg"),
    }

    result = pipeline.inspect(images)

    assert result["coverage_percentage"] == 25.0
    assert "left" in result["missing_views"]
    assert "right" in result["missing_views"]


def test_invalid_view_name_raises_error():
    pipeline = MultiViewInspectionPipeline()

    try:
        pipeline.inspect(
            {
                "roof": _create_test_image("roof.jpg"),
            }
        )

        raise AssertionError("ValueError was not raised.")

    except ValueError:
        pass


def test_missing_file_raises_error():
    pipeline = MultiViewInspectionPipeline()

    try:
        pipeline.inspect(
            {
                "front": "does_not_exist.jpg",
            }
        )

        raise AssertionError("FileNotFoundError was not raised.")

    except FileNotFoundError:
        pass