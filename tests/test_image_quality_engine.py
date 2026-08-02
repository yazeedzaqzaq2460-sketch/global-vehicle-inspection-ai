from pathlib import Path

import cv2
import numpy as np

from src.quality.image_quality_engine import ImageQualityEngine


TEST_OUTPUT_DIR = Path("outputs/quality_tests")


def _save_test_image(
    filename: str,
    image: np.ndarray,
) -> str:
    TEST_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = TEST_OUTPUT_DIR / filename

    saved = cv2.imwrite(
        str(output_path),
        image,
    )

    if not saved:
        raise RuntimeError(
            f"Failed to save test image: {output_path}"
        )

    return str(output_path)


def test_clear_image_is_acceptable():
    engine = ImageQualityEngine(
        blur_threshold=20.0,
        minimum_brightness=40.0,
        maximum_brightness=230.0,
        minimum_contrast=20.0,
    )

    # Create a realistic image with good brightness,
    # good contrast and sharp edges.
    image = np.full(
        (300, 300, 3),
        120,
        dtype=np.uint8,
    )

    cv2.rectangle(
        image,
        (40, 40),
        (260, 260),
        (255, 255, 255),
        4,
    )

    cv2.line(
        image,
        (40, 150),
        (260, 150),
        (0, 0, 0),
        4,
    )

    cv2.line(
        image,
        (150, 40),
        (150, 260),
        (0, 0, 0),
        4,
    )

    cv2.circle(
        image,
        (150, 150),
        50,
        (255, 255, 255),
        4,
    )

    cv2.putText(
        image,
        "AI",
        (105, 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 0, 255),
        3,
    )

    image_path = _save_test_image(
        "clear_image.jpg",
        image,
    )

    result = engine.evaluate(image_path)

    print(result)

    assert result["is_acceptable"] is True
    assert result["issues"] == []
    assert result["recommendation"] == "ACCEPT_IMAGE"


def test_blurry_image_is_rejected():
    engine = ImageQualityEngine(
        blur_threshold=100.0,
    )

    image = np.full(
        (200, 200, 3),
        128,
        dtype=np.uint8,
    )

    blurred_image = cv2.GaussianBlur(
        image,
        (31, 31),
        0,
    )

    image_path = _save_test_image(
        "blurry_image.jpg",
        blurred_image,
    )

    result = engine.evaluate(image_path)

    assert result["is_acceptable"] is False
    assert "BLURRY_IMAGE" in result["issues"]
    assert result["recommendation"] == "RECAPTURE_REQUIRED"


def test_dark_image_is_rejected():
    engine = ImageQualityEngine(
        minimum_brightness=45.0,
    )

    image = np.full(
        (200, 200, 3),
        10,
        dtype=np.uint8,
    )

    image_path = _save_test_image(
        "dark_image.jpg",
        image,
    )

    result = engine.evaluate(image_path)

    assert result["is_acceptable"] is False
    assert "IMAGE_TOO_DARK" in result["issues"]


def test_bright_image_is_rejected():
    engine = ImageQualityEngine(
        maximum_brightness=220.0,
    )

    image = np.full(
        (200, 200, 3),
        250,
        dtype=np.uint8,
    )

    image_path = _save_test_image(
        "bright_image.jpg",
        image,
    )

    result = engine.evaluate(image_path)

    assert result["is_acceptable"] is False
    assert "IMAGE_TOO_BRIGHT" in result["issues"]


def test_low_contrast_image_is_rejected():
    engine = ImageQualityEngine(
        minimum_contrast=30.0,
    )

    image = np.full(
        (200, 200, 3),
        128,
        dtype=np.uint8,
    )

    image[50:150, 50:150] = 135

    image_path = _save_test_image(
        "low_contrast_image.jpg",
        image,
    )

    result = engine.evaluate(image_path)

    assert result["is_acceptable"] is False
    assert "LOW_CONTRAST" in result["issues"]


def test_missing_image_raises_file_not_found():
    engine = ImageQualityEngine()

    missing_path = (
        "test_images/"
        "this_image_does_not_exist.jpg"
    )

    try:
        engine.evaluate(missing_path)

        raise AssertionError(
            "FileNotFoundError was not raised."
        )

    except FileNotFoundError:
        pass