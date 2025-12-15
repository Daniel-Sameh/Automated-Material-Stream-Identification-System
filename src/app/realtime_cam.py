# realtime_cam.py  (uses models/svm_open_set.pkl: pipeline + threshold)
import cv2
import numpy as np
import joblib
import os
from collections import deque

from src.pipeline.inference import Predict  # optional fallback
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

# -----------------------------
# Load open-set artifact (preferred)
# -----------------------------
artifact = None
pipeline = None
threshold = None
unknown_label = -1

if os.path.exists("models/svm_open_set.pkl"):
    artifact = joblib.load("models/svm_open_set.pkl")
    pipeline = artifact["pipeline"]
    threshold = artifact.get("threshold", None)
    unknown_label = artifact.get("unknown_label", -1)
    print("Loaded open-set model from models/svm_open_set.pkl")

else:
    # Fallback to older closed-set pipelines if you want
    for candidate in ("models/svm_pipeline.pkl", "models/svm.pkl"):
        if os.path.exists(candidate):
            try:
                pipeline = joblib.load(candidate)
                print(f"Loaded closed-set pipeline from {candidate}")
                break
            except Exception:
                pipeline = None

def max_margin_score(scores: np.ndarray) -> float:
    """Compute 'max score' compatible with binary/multiclass decision_function."""
    scores = np.asarray(scores)
    if scores.ndim == 1:
        return float(abs(scores[0]))
    return float(np.max(scores, axis=1)[0])

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
        if pipeline is not None:
            try:
                img_proc = preprocess_for_inference(center_roi)

                feats = feature_extractor.extract([img_proc])  # shape (1, D)

                # CLOSED-SET prediction from pipeline
                pred_closed = pipeline.predict(feats)[0]

                # Get confidence scores properly THROUGH the pipeline
                # (important: don't call clf.decision_function on raw feats)
                max_score = None
                if hasattr(pipeline, "decision_function"):
                    scores = pipeline.decision_function(feats)
                    max_score = max_margin_score(scores)

                # Apply open-set threshold if available
                pred = pred_closed
                if (threshold is not None) and (max_score is not None):
                    if max_score < threshold:
                        pred = unknown_label

                # Create a simple display confidence
                # (not a probability; just a normalized margin above threshold when possible)
                if max_score is None:
                    last_confidence = ""
                else:
                    if threshold is not None:
                        # 0% at threshold, grows as margin increases
                        conf = (max_score - threshold) / (abs(threshold) + 1e-6)
                        conf = float(np.clip(conf, 0.0, 1.0)) * 100.0
                    else:
                        # if no threshold, just show a bounded version of margin
                        conf = (max_score / (abs(max_score) + 1e-6)) * 100.0
                    last_confidence = f" ({conf:.1f}%)"

                if pred == unknown_label:
                    last_label = "Unknown"
                    last_confidence = ""  # or keep it if you want
                else:
                    last_label = classes[int(pred)]

            except Exception:
                # optional fallback to your existing Predict() function
                result = Predict(center_roi)
                if isinstance(result, tuple):
                    last_label, confidence = result
                    last_confidence = f" ({confidence:.1f}%)" if confidence else ""
                else:
                    last_label = result
                    last_confidence = ""
        else:
            # pipeline not found -> fallback
            result = Predict(center_roi)
            if isinstance(result, tuple):
                last_label, confidence = result
                last_confidence = f" ({confidence:.1f}%)" if confidence else ""
            else:
                last_label = result
                last_confidence = ""

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    display_text = f"{last_label}{last_confidence}"
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