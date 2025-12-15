import cv2
import numpy as np
from src.pipeline.inference import Predict
from collections import deque

cap = cv2.VideoCapture(0)
frame_count = 0
prediction_interval = 15  # Predict every 15 frames for efficiency

last_label = "Place object in center"
last_confidence = ""

print("Ready! Place object in center of frame")

history = deque(maxlen=10)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    h, w = frame.shape[:2]
    
    # Define center region (60% of frame centered)
    crop_size = int(min(h, w) * 0.6)
    x1 = (w - crop_size) // 2
    y1 = (h - crop_size) // 2
    x2 = x1 + crop_size
    y2 = y1 + crop_size
    
    # Extract center region
    center_roi = frame[y1:y2, x1:x2]
    
    # Predict on center region periodically
    if frame_count % prediction_interval == 0:
        result, conf = Predict(center_roi)
        if isinstance(result, tuple):
            last_label, confidence = result
            last_confidence = f" ({confidence:.1f}%)" if confidence else ""
        else:
            last_label = result
            last_confidence = ""
    
    # Draw center region box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    
    # Display result at top
    display_text = f"{last_label}{last_confidence}"
    label_size, _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
    
    # Semi-transparent background for text
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (20 + label_size[0], 50 + label_size[1]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Draw text
    cv2.putText(frame, display_text, (15, 40 + label_size[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    
    cv2.imshow("Material Classifier", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty("Material Classifier", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()