import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI(title="Bridge Damage Detection API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained YOLO model
MODEL_PATH = "best.pt"

model = YOLO(MODEL_PATH)

print(f"✓ Model loaded: {MODEL_PATH}")


# Detection thresholds
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5


# Class name mapping
CLASS_NAME_MAPPING = {
    "cracks": "Cracks",
    "spalling": "Spalling",
    "seepage": "Seepage",
    "honeycomb surface": "Honeycomb",
    "exposed rebar": "Exposed Reinforcement",
    "holes": "Holes",
}


# ==========================================
# ROOT ROUTE (para sa Render health check)
# ==========================================
@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "Bridge Damage Detection API is running",
        "model": MODEL_PATH
    }


# Health check
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_PATH
    }


# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Read uploaded image
    image_bytes = await file.read()

    # Convert image bytes to PIL image
    image = Image.open(io.BytesIO(image_bytes))

    # Get image dimensions
    img_width, img_height = image.size

    # Run YOLO detection
    results = model(
        image,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD
    )

    detections = []

    # Process detection results
    for result in results:

        for box in result.boxes:

            # Get class ID
            class_id = int(box.cls[0])

            # Get original class name
            original_name = result.names[class_id]

            # Convert to display name
            display_name = CLASS_NAME_MAPPING.get(
                original_name,
                original_name
            )

            # Get confidence
            confidence = float(box.conf[0])

            # Get bounding box
            bbox = box.xyxy[0].tolist()

            # Add detection
            detections.append({
                "class": display_name,
                "original_class": original_name,
                "confidence": round(confidence, 4),
                "bbox": [
                    round(v, 2)
                    for v in bbox
                ]
            })

    # Return result
    return {
        "detections": detections,
        "total_objects": len(detections),
        "image_width": img_width,
        "image_height": img_height
    }


# Run locally / Render
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
