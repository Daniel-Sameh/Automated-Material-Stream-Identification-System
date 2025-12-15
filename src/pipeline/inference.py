# src/pipeline/inference.py
import os
import joblib
import numpy as np

from src.models.SVM.train_svm import SvmModel
from src.models.KNN.train_knn import KnnModel
from src.features.feature_extraction import FeatureExtractor
from src.config.settings import Settings
from src.pipeline.preprocess import preprocess_for_inference


classes = Settings().classes
feature_extractor = FeatureExtractor()

# -----------------------------
# Preferred: open-set artifact (pipeline + threshold)
# -----------------------------
artifact = None
pipeline = None
pipeline_threshold = None
unknown_label = -1

open_set_path = "models/svm_open_set.pkl"
if os.path.exists(open_set_path):
    try:
        artifact = joblib.load(open_set_path)
        # expected keys: pipeline, threshold, unknown_label
        pipeline = artifact.get("pipeline", None)
        pipeline_threshold = artifact.get("threshold", None)
        unknown_label = artifact.get("unknown_label", -1)
        print(f"Loaded open-set artifact from {open_set_path} (threshold={pipeline_threshold}, unknown_label={unknown_label})")
    except Exception:
        artifact = None
        pipeline = None
        pipeline_threshold = None

# -----------------------------
# Secondary: closed-set pipeline (no threshold)
# -----------------------------
if pipeline is None:
    pipeline_path = None
    for candidate in ("models/svm_pipeline.pkl", "models/svm.pkl"):
        if os.path.exists(candidate):
            pipeline_path = candidate
            break

    if pipeline_path:
        try:
            loaded = joblib.load(pipeline_path)
            # Sometimes older code may have saved dicts {'pipeline':..., 'threshold':...}
            if isinstance(loaded, dict) and "pipeline" in loaded:
                pipeline = loaded["pipeline"]
                pipeline_threshold = loaded.get("threshold", None)
                unknown_label = loaded.get("unknown_label", -1)
                print(f"Loaded pipeline artifact from {pipeline_path} (threshold={pipeline_threshold})")
            else:
                pipeline = loaded
                pipeline_threshold = None
                print(f"Loaded pipeline from {pipeline_path}")
        except Exception:
            pipeline = None
            pipeline_threshold = None

# -----------------------------
# Legacy fallback: scaler+pca+best.pkl with SvmModel/KnnModel
# -----------------------------
scaler = None
pca = None
model = None
best_model_name = None
svm_wrapper = None

if pipeline is None:
    if (
        os.path.exists("models/scaler.pkl")
        and os.path.exists("models/pca_reducer.pkl")
        and os.path.exists("models/best.pkl")
    ):
        scaler = joblib.load("models/scaler.pkl")
        pca = joblib.load("models/pca_reducer.pkl")["pca"]
        best_model = joblib.load("models/best.pkl")
        best_model_name = best_model.get("best")

        if best_model_name == "svm":
            model = SvmModel()
            model.load("models/svm.pkl")
            svm_wrapper = model
            print("Loaded legacy SvmModel.")
        else:
            model = KnnModel()
            model.load("models/knn.pkl")
            print("Loaded legacy KNN model.")


def _max_margin(scores: np.ndarray) -> float:
    """max score compatible with binary/multiclass SVM decision_function outputs."""
    scores = np.asarray(scores)
    if scores.ndim == 1:
        return float(np.abs(scores)[0])
    return float(np.max(scores, axis=1)[0])


def Predict(image):
    image = preprocess_for_inference(image)

    # 1) Feature extraction
    feats = feature_extractor.extract([image])  # (1, D)

    # 2) Pipeline path (preferred)
    if pipeline is not None:
        pred_closed = pipeline.predict(feats)[0]

        raw_score = None
        try:
            if hasattr(pipeline, "decision_function"):
                scores = pipeline.decision_function(feats)
                raw_score = _max_margin(scores)
        except Exception:
            raw_score = None

        # Apply stored threshold if available (open-set behavior)
        if pipeline_threshold is not None and raw_score is not None:
            if raw_score < float(pipeline_threshold):
                return "unknown", 0.0

        # If model itself outputs unknown label (in case you ever use a rejecting clf)
        if pred_closed == unknown_label:
            return "unknown", 0.0

        label = classes[int(pred_closed)]

        # confidence: simple, monotonic with margin; NOT a probability
        if raw_score is None:
            confidence = 0.0
        elif pipeline_threshold is not None:
            # 0% at threshold, increases above it, capped at 100
            confidence = (raw_score - float(pipeline_threshold)) / (abs(float(pipeline_threshold)) + 1e-6)
            confidence = float(np.clip(confidence, 0.0, 1.0) * 100.0)
        else:
            confidence = float((raw_score / (abs(raw_score) + 1e-6)) * 100.0)

        return label, float(confidence)

    # 3) Legacy wrapper path
    if scaler is None or pca is None or model is None:
        raise RuntimeError("No model available for inference. Train and save models/svm_open_set.pkl or a pipeline first.")

    feats = scaler.transform(feats)
    feats = pca.transform(feats)

    if best_model_name == "svm" and svm_wrapper is not None:
        # svm_wrapper.predict returns -1 for unknown if it has its own threshold
        pred = svm_wrapper.predict(feats)[0]

        if pred == -1:
            return "unknown", 0.0

        # decision-function based confidence (not calibrated)
        try:
            scores = svm_wrapper.model.decision_function(feats)
            raw_score = _max_margin(scores)
            confidence = float((raw_score / (abs(raw_score) + 1e-6)) * 100.0)
        except Exception:
            confidence = 0.0

        return classes[int(pred)], confidence

    # KNN legacy
    pred = model.predict(feats)[0]
    if pred == -1:
        return "unknown", 0.0
    return classes[int(pred)], 0.0


if __name__ == "__main__":
    import cv2

    test_dir = "test/"
    if os.path.isdir(test_dir):
        for filename in os.listdir(test_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(test_dir, filename)
                image = cv2.imread(image_path)
                print(f"Image: {filename}")
                print(Predict(image))