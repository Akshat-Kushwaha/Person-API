#!/usr/bin/env python3
"""
Simple webcam person detector using Ultralytics YOLOv8 (yolov8n).
Draws bounding boxes around detected persons and shows confidence.

Usage (local):
    python webcam_person_detect.py
Or specify source (webcam index, video file, or stream URL):
    VIDEO_SOURCE=0 python webcam_person_detect.py
    VIDEO_SOURCE=/path/to/video.mp4 python webcam_person_detect.py
    VIDEO_SOURCE=rtsp://... python webcam_person_detect.py
"""

import os
import cv2
import time
import argparse
from ultralytics import YOLO
import numpy as np

# Config
CONF_THRESHOLD = 0.35       # detection confidence threshold
VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "0")  # default webcam
HEADLESS = os.getenv("HEADLESS", "0") == "1"   # if true, don't open GUI windows
MODEL_NAME = "yolov8n.pt"   # small, fast model—will auto-download if missing
DEVICE = os.getenv("YOLO_DEVICE", "cpu")  # navailable)

def str_to_source(s):
    # Convert environment string to numeric index if appropriate
    try:
        return int(s)
    except Exception:
        return s

def main():
    source = str_to_source(VIDEO_SOURCE)
    print(f"[INFO] Video source: {source}")
    print(f"[INFO] Loading model {MODEL_NAME} on device={DEVICE} ... (this may download weights)")

    model = YOLO(MODEL_NAME)  # load model

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {source}")

    fps_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of stream or cannot fetch frame.")
            break

        # YOLO expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run inference on single frame
        results = model.predict(
            source=rgb,
            conf=CONF_THRESHOLD,
            device=DEVICE,
            verbose=False,
            imgsz=640  # speed/accuracy tradeoff
        )

        # results is a list; we used single-frame input so take first
        r = results[0]

        # r.boxes contains xyxy, conf, cls
        boxes = getattr(r, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()        # (N,4)
            confs = boxes.conf.cpu().numpy()      # (N,)
            clss = boxes.cls.cpu().numpy().astype(int)  # (N,)

            for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, clss):
                # COCO class 0 == person
                if cls != 0:
                    continue

                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                label = f"person {conf:.2f}"

                # Draw box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                cv2.rectangle(frame, (x1, y1 - t_size[1] - 6), (x1 + t_size[0] + 6, y1), (0,200,0), -1)
                cv2.putText(frame, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        # FPS overlay
        cur_fps = 1.0 / (time.time() - fps_time + 1e-6)
        fps_time = time.time()
        cv2.putText(frame, f"FPS: {cur_fps:.1f}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        if not HEADLESS:
            cv2.imshow("Person Detector", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("[INFO] Quitting by user input.")
                break
        else:
            # In headless mode, you might want to write frames to disk or stream them.
            # For this simple demo we just continue processing.
            pass

    cap.release()
    if not HEADLESS:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
