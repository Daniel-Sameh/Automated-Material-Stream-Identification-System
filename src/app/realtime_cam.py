import cv2
import numpy as np
import joblib
import os
from collections import deque

from src.pipeline.inference import Predict
from src.pipeline.preprocess import preprocess_for_inference
from src.features.feature_extraction import FeatureExtractor
from src.config.settings import Settings

cap = cv2.VideoCapture(0)
frame_count = 0
prediction_interval = 15

last_label = "Place object in center"
last_confidence = ""

print("Ready! Place object in center of frame")

history = deque(maxlen=10)

classes = Settings().classes
feature_extractor = FeatureExtractor()

pipeline = None
threshold = None
unknown_label = -1


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    h, w = frame.shape[:2]

    crop_size = int(min(h, w) * 0.6)
    x1 = (w - crop_size) // 2
    y1 = (h - crop_size) // 2
    x2 = x1 + crop_size
    y2 = y1 + crop_size

    center_roi = frame[y1:y2, x1:x2]

    if frame_count % prediction_interval == 0:
            result = Predict(center_roi)
            if isinstance(result, tuple):
                last_label, confidence = result
                last_confidence = f" ({confidence:.1f}%)" if confidence else ""
            else:
                last_label = result
                last_confidence = ""

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    display_text = f"{last_label}"
    label_size, _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (20 + label_size[0], 50 + label_size[1]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(
        frame, display_text, (15, 40 + label_size[1]),
        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2
    )

    cv2.imshow("Material Classifier", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    if cv2.getWindowProperty("Material Classifier", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()