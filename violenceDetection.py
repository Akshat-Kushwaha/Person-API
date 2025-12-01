import io
import os
import base64
from typing import Optional, List, Tuple

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from pydantic import BaseModel
from PIL import Image

# Try to import tflite runtime (lighter). Fallback to full tensorflow if unavailable.
try:
    from tflite_runtime.interpreter import Interpreter
    print("Using tflite_runtime Interpreter")
except Exception:
    try:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
        print("Using tensorflow.lite Interpreter")
    except Exception as e:
        raise RuntimeError(
            "Neither tflite-runtime nor tensorflow available. Install one of them."
        )

# -----------------------
# Configuration
# -----------------------
MODEL_PATH = os.environ.get("TFLITE_MODEL_PATH", "model.tflite")
LABELS_PATH = os.environ.get("TFLITE_LABELS_PATH", "labels.txt")
IMG_SIZE = 224  # MobileNetV3 input size

# -----------------------
# Load labels
# -----------------------
def load_labels(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Labels file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    return lines

labels = []
try:
    labels = load_labels(LABELS_PATH)
    print(f"Loaded {len(labels)} labels from {LABELS_PATH}")
except FileNotFoundError:
    print("Warning: labels.txt not found. Predictions will return class indices.")


# -----------------------
# Load TFLite Interpreter
# -----------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"TFLite model not found at {MODEL_PATH}")

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

input_shape = input_details["shape"]  # e.g. [1,224,224,3]
input_dtype = input_details["dtype"]  # np.float32 or np.uint8 etc.
print("TFLite input shape:", input_shape, "dtype:", input_dtype)
print("TFLite output shape:", output_details["shape"], "dtype:", output_details["dtype"])


# -----------------------
# FastAPI app + schemas
# -----------------------
app = FastAPI(title="TFLite MobileNetV3 Classifier", version="1.0")


class PredictionItem(BaseModel):
    class_name: str
    class_id: int
    confidence: float


class PredictResponse(BaseModel):
    predictions: List[PredictionItem]


# -----------------------
# Helpers
# -----------------------
def load_image_from_bytes(img_bytes: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")


def preprocess_image(pil_img: Image.Image, size: int = IMG_SIZE, input_dtype=np.float32) -> np.ndarray:
    """
    Resize and preprocess image for MobileNetV3.
    Handles both float input (expects [-1,1]) and uint8 input (0-255).
    """
    img = pil_img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(img)

    if input_dtype == np.uint8:
        # Many quantized tflite models expect uint8 0..255
        input_data = np.expand_dims(arr.astype(np.uint8), axis=0)
    else:
        # assume float input expecting [-1, 1] as MobileNet preprocess:
        # scale from [0,255] to [-1,1]: (x / 127.5) - 1
        arr = arr.astype(np.float32)
        arr = (arr / 127.5) - 1.0
        input_data = np.expand_dims(arr, axis=0).astype(np.float32)

    return input_data


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum(axis=-1, keepdims=True)


def predict_top_k(interpreter, top_k: int = 1) -> List[Tuple[int, float]]:
    """
    Run interpreter (assumes input tensor already set), return list of (class_id, confidence)
    """
    interpreter.invoke()
    out_details = interpreter.get_output_details()
    # handle different outputs: logits or probabilities
    out = interpreter.get_tensor(out_details[0]["index"])
    # flatten to 1D
    out = np.squeeze(out)

    # If output is logits (float) - apply softmax
    if out.dtype == np.float32:
        probs = softmax(out)
    else:
        # If uint8 probabilities due to quantization, we may need to dequantize:
        # Check if quantization params present
        qparams = out_details[0].get("quantization_parameters", {})
        scale = out_details[0].get("quantization", (1.0, 0))[0] if "quantization" in out_details[0] else 1.0
        # Simple fallback: convert to float and softmax
        probs = out.astype(np.float32)
        try:
            probs = softmax(probs)
        except Exception:
            # final fallback: normalize to sum 1
            probs = probs / (np.sum(probs) + 1e-12)

    # top k
    if top_k == 1:
        idx = int(np.argmax(probs))
        return [(idx, float(probs[idx]))]
    else:
        top_idx = np.argsort(probs)[-top_k:][::-1]
        return [(int(i), float(probs[i])) for i in top_idx]


# -----------------------
# API Endpoint
# -----------------------
@app.post("/predict", response_model=PredictResponse)
async def predict(request: Request, frame: Optional[UploadFile] = File(None), top_k: int = 1):
    """
    POST multipart/form-data with file field `frame` (preferred),
    OR send JSON: {"b64": "<base64 image>"}.

    Query param: top_k (default 1) to return top-k predictions.
    """
    img_bytes = None

    if frame is not None:
        img_bytes = await frame.read()
        if not img_bytes:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")
    else:
        # try JSON body
        try:
            body = await request.json()
        except Exception:
            body = None
        if not body or "b64" not in body:
            raise HTTPException(status_code=400, detail="Send multipart 'frame' or JSON with 'b64'.")
        try:
            img_bytes = base64.b64decode(body["b64"])
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 provided.")

    pil_img = load_image_from_bytes(img_bytes)
    input_tensor = preprocess_image(pil_img, size=IMG_SIZE, input_dtype=input_dtype)

    # Set tensor
    interpreter.set_tensor(input_details["index"], input_tensor)
    preds = predict_top_k(interpreter, top_k=top_k)

    response_items = []
    for cid, conf in preds:
        name = labels[cid] if cid < len(labels) else str(cid)
        response_items.append(PredictionItem(class_name=name, class_id=cid, confidence=float(conf)))

    return PredictResponse(predictions=response_items)


# Root
@app.get("/")
def root():
    return {"status": "ok", "model": os.path.basename(MODEL_PATH), "labels": len(labels)}
