"""
violence_detector_tflite.py
Simple violence detection on a video using a TFLite binary classifier.

Requirements:
 - tensorflow (for tflite runtime: `pip install tensorflow` or `pip install tflite-runtime`)
 - opencv-python
 - numpy

Change these variables below: model_path, video_path, output_path
"""

import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import time
from pathlib import Path

# ---------- USER CONFIG ----------
# Use absolute paths or Colab/Drive paths as needed
model_path = "model/violenceSmall.tflite"
video_path = "testingVideo/1 (3).mp4"   # <-- set your video file here
output_path = "output/violence_output.mp4"

# model input size (depends on your training) - you used 224x224
INPUT_SIZE = (224, 224)

# smoothing buffer (number of frames used to average predictions)
SMOOTHING_FRAMES = 7

# probability threshold above which we label 'VIOLENCE'
VIOLENCE_THRESHOLD = 0.5

# scale probabilities if model expects 0-1 floats; will be auto-detected for uint8 input
# ----------------------------------

def load_tflite_model(path: str):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    return interpreter, input_details, output_details

def preprocess_frame(frame, input_shape, input_details):
    # frame: BGR (from OpenCV)
    # convert to RGB, resize, and cast/scale depending on model input dtype
    h, w = input_shape
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (w, h))  # (width, height) order
    # add batch dim
    img = np.expand_dims(img, axis=0)
    dtype = input_details['dtype']
    if dtype == np.float32:
        # most Keras models expect float32 scaled to [0,1] or [-1,1]
        # Your MobileNetV3 used imagenet weights — usual preprocess: [-1,1] or [0,1]
        # We'll use 0-1 normalization as a safe choice; if your model requires different, change here.
        img = img.astype(np.float32) / 255.0
    elif dtype == np.uint8:
        img = img.astype(np.uint8)
    else:
        img = img.astype(dtype)
    return img

def run_inference(interpreter, input_details, output_details, preprocessed):
    interpreter.set_tensor(input_details['index'], preprocessed)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details['index'])
    # out shape: (1,1) or (1,) depending on model — handle both
    prob = float(np.squeeze(out))
    # If model outputs logits rather than probability, apply sigmoid:
    # (User likely saved with sigmoid activation, but if prob outside [0,1], apply sigmoid)
    if prob < 0.0 or prob > 1.0:
        prob = 1.0 / (1.0 + np.exp(-prob))
    return prob

def format_label(prob, threshold):
    label = "VIOLENCE" if prob >= threshold else "NO VIOLENCE"
    return f"{label}: {prob:.3f}"

def main():
    # sanity checks
    if not Path(model_path).exists():
        raise FileNotFoundError(f"TFLite model not found: {model_path}")
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    interpreter, input_details, output_details = load_tflite_model(model_path)
    # infer expected input shape (H,W)
    # input_details['shape'] might be (1,224,224,3) or dynamic; use shape_signature if provided
    shape = input_details.get('shape', input_details.get('shape_signature', None))
    if shape is None:
        in_h, in_w = INPUT_SIZE
    else:
        # shape can be (1,h,w,3) or (1,224,224,3)
        try:
            _, h, w, _ = shape
            in_h, in_w = int(h), int(w)
        except Exception:
            in_h, in_w = INPUT_SIZE

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_vid = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    prob_buffer = deque(maxlen=SMOOTHING_FRAMES)
    frame_count = 0
    start_time = time.time()

    print("Starting inference...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Preprocess & forward pass
        prepped = preprocess_frame(frame, (in_h, in_w), input_details)
        prob = run_inference(interpreter, input_details, output_details, prepped)
        prob_buffer.append(prob)
        smoothed_prob = float(np.mean(prob_buffer))

        # Overlay label
        label_text = format_label(smoothed_prob, VIOLENCE_THRESHOLD)
        # Choose color based on label
        if smoothed_prob >= VIOLENCE_THRESHOLD:
            color = (0, 0, 255)  # Red for violence (BGR)
        else:
            color = (0, 255, 0)  # Green for safe

        # Draw a semi-transparent rectangle as background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (360, 60), (0, 0, 0), -1)
        alpha = 0.45
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        cv2.putText(frame, label_text, (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

        # Optionally show a small probability bar
        bar_x, bar_y = 400, 20
        bar_w, bar_h = 240, 20
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), 1)
        fill_w = int(bar_w * smoothed_prob)
        cv2.rectangle(frame, (bar_x + 1, bar_y + 1), (bar_x + 1 + fill_w, bar_y + bar_h - 1), color, -1)

        out_vid.write(frame)

        # Optional: show live (comment out in headless Colab)
        # cv2.imshow("Violence Detection", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    duration = time.time() - start_time
    cap.release()
    out_vid.release()
    # cv2.destroyAllWindows()
    print(f"Done. Processed {frame_count} frames in {duration:.1f}s ({frame_count/duration:.2f} FPS).")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    main()
