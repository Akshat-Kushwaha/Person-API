# violence_detector_fixed.py
import cv2
import numpy as np
import os
import time
from datetime import datetime

import tensorflow as tf
from keras.models import load_model
from keras.models import Model as KerasModel  # used when wrapping TFSMLayer
from keras.layers import TFSMLayer
from keras import Model
# ===========================
# CONFIG
# ===========================
MODEL_PATH = "model/model_hawkeye"     # your trained violence model (SavedModel folder or .keras/.h5)
VIDEO_PATH = "testingVideo/1 (2).webm"  # input video file
OUTPUT_DIR = "./output"                # folder where results will be saved

os.makedirs(OUTPUT_DIR, exist_ok=True)

FRAME_SIZE = (224, 224)
SEQ_LEN = 64
THRESHOLD = 0.75        # violence probability threshold
# Note: FPS will be inferred from the video file; this default is fallback
DEFAULT_FPS = 30

# ===========================
# MODEL LOADING (robust)
# ===========================
def load_inference_model(model_path):
    """
    Try keras.load_model (works for .keras / .h5).
    If it fails because the path is a SavedModel directory (Keras3), load via TFSMLayer.
    """
    # 1) try load_model - covers .keras and .h5
    try:
        print(f"Trying keras.models.load_model('{model_path}') ...")
        tfsmlayer = TFSMLayer(MODEL_PATH, call_endpoint='serving_default')

        # Wrap into a Keras Model for .predict()
        model = Model(inputs=tfsmlayer.inputs, outputs=tfsmlayer.outputs)
        print("✅ Model loaded with keras.models.load_model()")
        return model
    except Exception as e:
        print("⚠ keras.models.load_model failed:", str(e))

    # 2) If it's a SavedModel folder, try to inspect signatures
    try:
        print(f"Trying to read SavedModel signatures at '{model_path}' ...")
        loaded = tf.saved_model.load(model_path)
        sigs = list(loaded.signatures.keys())
        print("SavedModel signatures:", sigs)
        # pick a sensible default
        call_endpoint = sigs[0] if len(sigs) > 0 else "serving_default"
        print(f"Using call_endpoint = '{call_endpoint}' with TFSMLayer ...")
    except Exception as e:
        print("Failed to read saved_model signatures:", e)
        call_endpoint = "serving_default"

    # 3) Wrap with TFSMLayer
    try:
        tfsmlayer = TFSMLayer(model_path, call_endpoint=call_endpoint)
        model = KerasModel(inputs=tfsmlayer.inputs, outputs=tfsmlayer.outputs)
        print("✅ Model loaded via TFSMLayer and wrapped into Keras Model")
        return model
    except Exception as e:
        print("❌ Failed to load model via TFSMLayer:", e)
        raise RuntimeError("Model loading failed. Ensure MODEL_PATH points to a .keras/.h5 file or a SavedModel folder.") from e

# ===========================
# OPTICAL FLOW
# ===========================
def get_optical_flow(video):
    gray_frames = [cv2.cvtColor(f.astype('uint8'), cv2.COLOR_BGR2GRAY)
                   for f in video]

    flows = []
    for i in range(len(gray_frames) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            gray_frames[i], gray_frames[i+1], None,
            0.5, 3, 15, 3, 5, 1.2,
            cv2.OPTFLOW_FARNEBACK_GAUSSIAN
        )
        # remove mean (centering)
        flow[..., 0] -= np.mean(flow[..., 0])
        flow[..., 1] -= np.mean(flow[..., 1])

        # normalize to 0-255 range for each channel separately
        flow[..., 0] = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
        flow[..., 1] = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX)
        flows.append(flow.astype(np.float32))

    flows.append(np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 2), dtype=np.float32))  # pad last
    return np.array(flows, dtype=np.float32)

# ===========================
# NORMALIZATION
# ===========================
def normalize(x):
    mean = np.mean(x)
    std = np.std(x)
    return (x - mean) / (std + 1e-6)

# ===========================
# PREPROCESS 64 FRAMES
# ===========================
def preprocess_frames(frames):
    """
    frames: list/array of BGR frames (original pixel space)
    returns: np.array shaped (1, SEQ_LEN, 224, 224, 5)
    """
    frames_resized = []

    total = len(frames)
    if total < SEQ_LEN:
        # pad by repeating last frame if not enough - but normally your buffer size is SEQ_LEN
        frames = frames + [frames[-1]] * (SEQ_LEN - total)
        total = len(frames)

    # sample SEQ_LEN evenly spaced frames
    for i in range(SEQ_LEN):
        idx = int(total * i / SEQ_LEN)
        # boundary safety
        idx = min(idx, total - 1)
        f = cv2.resize(frames[idx], FRAME_SIZE)  # cv2.resize expects (w,h) as target?
        # cv2.resize uses (width, height) in tuple arg (cols, rows)
        # FRAME_SIZE is (224,224) we passed equally, so it's fine.
        f = cv2.cvtColor(f.astype('uint8'), cv2.COLOR_BGR2RGB)
        frames_resized.append(f)

    frames_resized = np.array(frames_resized, dtype=np.float32)
    flows = get_optical_flow(frames_resized)

    combined = np.zeros((SEQ_LEN, FRAME_SIZE[1], FRAME_SIZE[0], 5), dtype=np.float32)
    combined[..., :3] = frames_resized
    combined[..., 3:] = flows

    combined = normalize(combined)
    return combined.reshape(1, SEQ_LEN, FRAME_SIZE[1], FRAME_SIZE[0], 5)

# ===========================
# SAVE VIDEO CLIP
# ===========================
def save_clip(frames, event_id, fps):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"{OUTPUT_DIR}/violence_{event_id}_{now}.mp4"

    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    for f in frames:
        writer.write(f)

    writer.release()
    print(f"🚨 Saved violence clip → {out_path}")

# ===========================
# MAIN LOOP
# ===========================
def run_detection(model, video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Cannot open video:", video_path)
        return

    # read fps from file; fallback to DEFAULT_FPS if not available
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = DEFAULT_FPS
    print(f"Video FPS: {fps:.2f}")

    print("🎥 Processing video...\n")

    frame_buffer = []
    event_id = 0
    processed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_buffer.append(frame)
        processed_frames += 1

        # Only process once SEQ_LEN frames collected
        if len(frame_buffer) >= SEQ_LEN:
            # Preprocess (takes SEQ_LEN frames sampled evenly from buffer)
            batch = preprocess_frames(frame_buffer)

            # Predict
            try:
                pred_raw = model.predict(batch)
                # handle various output shapes
                pred = float(np.squeeze(pred_raw))
            except Exception as e:
                # if model outputs a dict (possible with TFSMLayer), try to extract
                print("Warning: model.predict raised:", e)
                try:
                    out = model(batch)
                    if isinstance(out, (list, tuple)):
                        out_arr = out[0].numpy()
                    elif isinstance(out, dict):
                        # pick first item
                        out_arr = list(out.values())[0].numpy()
                    else:
                        out_arr = out.numpy()
                    pred = float(np.squeeze(out_arr))
                except Exception as e2:
                    print("❌ Unable to interpret model output:", e2)
                    raise

            print(f"[frame {processed_frames}] Prediction: {pred:.4f}")

            if pred > THRESHOLD:
                print(f"⚠ Violence detected! (p={pred:.4f})")
                # Save the current SEQ_LEN window (last SEQ_LEN frames in buffer)
                clip_frames = frame_buffer[:SEQ_LEN]
                save_clip(clip_frames.copy(), event_id, int(round(fps)))
                event_id += 1

            # Slide window by 1 frame (pop first frame)
            frame_buffer.pop(0)

    cap.release()
    print("\n✅ Video processing completed.")

# ===========================
# ENTRYPOINT
# ===========================
if __name__ == "__main__":
    model = load_inference_model(MODEL_PATH)
    run_detection(model, VIDEO_PATH)
