from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io
import os

app = FastAPI(title="Bridge Damage Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ⭐ AUTO-DETECT MODEL PATH
POSSIBLE_PATHS = [
    "runs/train/bridge_damage/weights/best.pt",
    "runs/detect/runs/train/bridge_damage/weights/best.pt",
    "runs/detect/train/bridge_damage/weights/best.pt",
]

MODEL_PATH = None
for path in POSSIBLE_PATHS:
    if os.path.exists(path):
        MODEL_PATH = path
        print(f"✓ Model found: {path}")
        break

if MODEL_PATH is None:
    raise FileNotFoundError(
        "Hindi mahanap ang best.pt. I-check ang runs/ folder."
    )

model = YOLO(MODEL_PATH)

# ⭐ THRESHOLDS
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5

# ⭐ CLASS NAME MAPPING — GYU-DET (6 classes)
CLASS_NAME_MAPPING = {
    "cracks": "Cracks",
    "spalling": "Spalling",
    "seepage": "Seepage",
    "honeycomb surface": "Honeycomb",
    "exposed rebar": "Exposed Reinforcement",
    "holes": "Holes",
}

@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_PATH}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    img_width, img_height = image.size

    results = model(
        image,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD
    )

    detections = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            original_name = result.names[class_id]
            display_name = CLASS_NAME_MAPPING.get(original_name, original_name)
            confidence = float(box.conf[0])
            bbox = box.xyxy[0].tolist()

            detections.append({
                "class": display_name,
                "original_class": original_name,
                "confidence": round(confidence, 4),
                "bbox": [round(v, 2) for v in bbox]
            })

    return {
        "detections": detections,
        "total_objects": len(detections),
        "image_width": img_width,
        "image_height": img_height
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)