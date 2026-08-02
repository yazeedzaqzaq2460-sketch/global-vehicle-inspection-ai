import uuid
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, HTTPException, UploadFile

from src.pipelines.inspection_pipeline import InspectionPipeline
from src.schemas.inspection_response import (
    InspectionResponse,
    InspectionSummary,
    VehicleData,
    DamageItem,
    BoundingBox,
)

app = FastAPI(
    title="Vehicle Inspection AI",
    version="1.0.0",
)

pipeline = InspectionPipeline()


@app.get("/")
def root():
    return {
        "status": "running",
        "project": "Vehicle Inspection AI",
        "version": "1.0.0",
    }


@app.post(
    "/inspect/image",
    response_model=InspectionResponse,
)
async def inspect_image(
    file: UploadFile = File(...),
):
    allowed_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
    }

    original_name = file.filename or "uploaded_image.jpg"
    extension = Path(original_name).suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image format.",
        )

    temporary_path = None

    try:
        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty.",
            )

        with NamedTemporaryFile(
            delete=False,
            suffix=extension,
        ) as temporary_file:
            temporary_file.write(image_bytes)
            temporary_path = temporary_file.name

        result = pipeline.inspect_image(temporary_path)

        response = InspectionResponse(
            inspection_id=str(uuid.uuid4()),
            status="SUCCESS",
            image_name=original_name,
            decision=result["decision"],
            vehicle=VehicleData(
                license_plate=result["vehicle_data"]["license_plate"],
                vin=result["vehicle_data"]["vin"],
                engine_number=result["vehicle_data"]["engine_number"],
            ),
            damages=[
                DamageItem(
                    class_id=item["class_id"],
                    class_name=item["class_name"],
                    confidence=item["confidence"],
                    bbox=BoundingBox(**item["bbox"]),
                )
                for item in result["damages"]
            ],
            summary=InspectionSummary(
                damage_count=result["damage_count"],
                highest_confidence=result["highest_confidence"],
            ),
        )

        return response

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Inspection failed: {error}",
        )

    finally:
        if temporary_path and Path(temporary_path).exists():
            Path(temporary_path).unlink()