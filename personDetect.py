import io
import os
import base64
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from PIL import Image
import numpy as np

# ultralytics YOLOv8
from ultralytics import YOLO

# Load model once at startup
MODEL_PATH = "yolov8n.pt"  # change to your uploaded model if needed
print("Loading model from:", MODEL_PATH)
model = YOLO(MODEL_PATH)

app = FastAPI(title="YOLOv8 Person Detector", version="1.0")


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


class DetectResponse(BaseModel):
    persons: List[BBox]


def pil_from_bytes(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def normalize_box(box: List[float], w: int, h: int):
    # Optional helper to return normalized coords (0..1). Not used by default.
    x1, y1, x2, y2 = box
    return [x1 / w, y1 / h, x2 / w, y2 / h]


@app.post("/detect", response_model=DetectResponse)
async def detect_persons(frame: Optional[UploadFile] = File(None), payload: Optional[dict] = None):
    """
    Detect persons in an image frame.

    - Provide multipart/form-data with file field `frame` (image).
    - OR provide JSON body: {"b64": "<base64-encoded-image>"}.
    Returns bounding boxes as absolute pixel coords [x1,y1,x2,y2] and confidence.
    """
    # Get image bytes either from multipart or from JSON base64
    img_bytes = None
    if frame is not None:
        contents = await frame.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")
        img_bytes = contents
    else:
        # try reading base64 JSON
        try:
            json_body = await app.request.json()  # will fail in some clients, so fallback to payload param
        except Exception:
            json_body = payload
        if json_body and "b64" in json_body:
            try:
                img_bytes = base64.b64decode(json_body["b64"])
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid base64 in 'b64' field.")
        else:
            raise HTTPException(status_code=400, detail="No image provided. Send multipart `frame` or JSON with `b64`.")

    # create PIL image
    try:
        pil = pil_from_bytes(img_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image data.")

    # Convert to numpy array (H, W, C)
    img_np = np.asarray(pil)

    # Run inference (this returns an ultralytics Results object or list of Results)
    # we call model.predict (or model(img_np)) and get results
    # Use conf=0.25 by default; you can expose it as a query param later
    results = model.predict(source=img_np, imgsz=640, conf=0.25, save=False, verbose=False)

    # results is a list (one element per image). We'll take first result.
    if not results:
        return DetectResponse(persons=[])

    res = results[0]

    # res.boxes.xyxy, res.boxes.conf, res.boxes.cls, res.names
    boxes = []
    names = res.names  # mapping id->name

    if hasattr(res, "boxes") and len(res.boxes) > 0:
        xyxy = res.boxes.xyxy.cpu().numpy()  # shape (N,4)
        confs = res.boxes.conf.cpu().numpy()  # shape (N,)
        cls_ids = res.boxes.cls.cpu().numpy().astype(int)  # shape (N,)

        H, W = img_np.shape[:2]

        for b, c, cls in zip(xyxy, confs, cls_ids):
            label = names.get(cls, str(cls))
            if label.lower() == "person" or label == "person":
                x1, y1, x2, y2 = [float(v) for v in b]
                boxes.append(BBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=float(c)))

    return DetectResponse(persons=boxes)
