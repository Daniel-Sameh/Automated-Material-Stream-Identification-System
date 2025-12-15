import joblib
import os
from src.models.SVM.train_svm import SvmModel
from src.models.KNN.train_knn import KnnModel
from src.features.feature_extraction import FeatureExtractor
import numpy as np
from src.config.settings import Settings
from src.pipeline.preprocess import preprocess_for_inference


classes = Settings().classes
feature_extractor = FeatureExtractor()

# Prefer a saved sklearn Pipeline (contains scaler+pca+clf)
pipeline_path = None
for candidate in ("models/svm_pipeline.pkl", "models/svm.pkl"):
    if os.path.exists(candidate):
        pipeline_path = candidate
        break

use_pipeline = False
pipeline = None
pipeline_threshold = None
svm_wrapper = None
if pipeline_path:
    try:
        loaded = joblib.load(pipeline_path)
        # If loaded is a SvmModel wrapper, prefer that
        if isinstance(loaded, SvmModel):
            svm_wrapper = loaded
            print(f"Loaded SvmModel wrapper from {pipeline_path}")
        # artifact may be a dict {'pipeline': Pipeline, 'threshold': float}
        elif isinstance(loaded, dict) and 'pipeline' in loaded:
            pipeline = loaded['pipeline']
            pipeline_threshold = loaded.get('threshold', None)
            use_pipeline = True
            print(f"Loaded pipeline artifact from {pipeline_path} (threshold={pipeline_threshold})")
        else:
            # assume it's a raw sklearn Pipeline
            pipeline = loaded
            pipeline_threshold = None
            use_pipeline = True
            print(f"Loaded pipeline from {pipeline_path}")
    except Exception:
        pipeline = None
        pipeline_threshold = None
        use_pipeline = False

# Fallback: legacy files
scaler = None
pca = None
model = None
best_model_name = None
if not use_pipeline:
    # legacy loading (scaler+pca+model)
    if os.path.exists("models/scaler.pkl") and os.path.exists("models/pca_reducer.pkl") and os.path.exists("models/best.pkl"):
        scaler = joblib.load("models/scaler.pkl")
        pca = joblib.load("models/pca_reducer.pkl")["pca"]
        best_model = joblib.load("models/best.pkl")
        best_model_name = best_model.get("best")

        if best_model_name == "svm":
            model = SvmModel()
            model.load("models/svm.pkl")
            print("Loaded legacy SVM model.")
        else:
            model = KnnModel()
            model.load("models/knn.pkl")
            print("Loaded legacy KNN model.")


def Predict(image):
    image = preprocess_for_inference(image)

    # 1. Feature extraction
    features = feature_extractor.extract([image])

    # If we loaded a SvmModel wrapper, use its scaler/pca then its predict API
    if svm_wrapper is not None:
        # expect scaler and pca attached to the wrapper
        try:
            X = features
            if hasattr(svm_wrapper, 'scaler') and svm_wrapper.scaler is not None:
                X = svm_wrapper.scaler.transform(X)
            if hasattr(svm_wrapper, 'pca') and svm_wrapper.pca is not None:
                X = svm_wrapper.pca.transform(X)

            preds = svm_wrapper.predict_with_confidence(X)
            # predict_with_confidence returns (preds, scores)
            if isinstance(preds, tuple) and len(preds) == 2:
                pred_arr, scores = preds
                pred = pred_arr[0]
                score = scores[0]
            else:
                pred = preds[0]
                score = 0.0

            if pred == -1:
                return "unknown", 0.0

            label = classes[int(pred)]
            # convert raw score to percentage-like value
            confidence = 100 * (abs(score) / (abs(score) + 1e-6)) if score is not None else 0.0
            return label, float(confidence)
        except Exception:
            # fallback to pipeline path
            pass

    # If we loaded an sklearn pipeline, feed the extracted features into it
    if use_pipeline and pipeline is not None:
        # pipeline expects a 2D array
        preds = pipeline.predict(features)

        # Compute decision-function score via pipeline (delegates to classifier)
        confidence = 0.0
        raw_score = None
        try:
            # pipeline.decision_function will call the estimator's decision_function
            scores = pipeline.decision_function(features)
            if scores is not None:
                if np.ndim(scores) == 1:
                    raw_score = np.abs(scores)[0]
                else:
                    raw_score = np.max(scores, axis=1)[0]
                confidence = 100 * (raw_score / (np.abs(raw_score) + 1e-6))
        except Exception:
            raw_score = None
            confidence = 0.0

        pred = preds[0]

        # If a threshold was stored with the pipeline, use it to reject unknowns
        if pipeline_threshold is not None and raw_score is not None:
            try:
                if raw_score < pipeline_threshold:
                    return "unknown", 0.0
            except Exception:
                pass

        if pred == -1:
            return "unknown", 0.0

        label = classes[int(pred)]
        return label, float(abs(confidence))

    # Legacy path: apply scaler + pca + model
    if scaler is None or pca is None or model is None:
        raise RuntimeError("No model available for inference. Train and save a pipeline first.")

    features = scaler.transform(features)
    X_reduced = pca.transform(features)

    if best_model_name == "svm":
        pred = model.predict(X_reduced)[0]

        # confidence from decision function
        scores = model.model.decision_function(X_reduced)
        confidence = np.max(scores)

        if pred == -1:
            return "unknown", 0.0

        label = classes[pred]
        confidence = 100 * (confidence / (np.abs(confidence) + 1e-6))

    else:  # KNN
        pred = model.predict(X_reduced)[0]
        confidence = 0.0

        if pred == -1:
            return "unknown", 0.0

        label = classes[pred]

    return label, float(abs(confidence))


if __name__ == "__main__":
    import cv2
    import os
    # Predict all images in test/
    test_dir = "test/"
    for filename in os.listdir(test_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(test_dir, filename)
            image = cv2.imread(image_path)
            print(f"Image: {filename}")
            print(Predict(image))