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

pipeline = None
pipeline_threshold = None
unknown_label = -1

model_name = None

if os.path.exists("models/bestModel.json"):
    bestModel = joblib.load("models/bestModel.json")
    model_name = bestModel["best"]
else:
    print("Please Train before running this file.")

if model_name=="svm":
    from src.models.SVM.train_svm import SvmModel

    model = SvmModel()
    svm_path = "models/svm.pkl"
    if os.path.exists(svm_path):
        model.load(svm_path)
        pipeline = model
        threshold = model.threshold
        unknown_label = model.unknown_label
        print(f"Loaded SVM model from {svm_path} with threshold {threshold}")
    else:
        print(f"SVM model file not found at {svm_path}")
elif model_name=="knn":
    from src.models.KNN.train_knn import KnnModel

    model = KnnModel()
    knn_path = "models/knn.pkl"
    if os.path.exists(knn_path):
        model.load(knn_path)
        pipeline = model
        print(f"Loaded KNN model from {knn_path}")
    else:
        print(f"KNN model file not found at {knn_path}")


# scaler = joblib.load("models/scaler.pkl")
# pca = joblib.load("models/pca.pkl")["pca"]

def Predict(image):
    image = preprocess_for_inference(image)

    # Feature extraction
    feats = feature_extractor.extract([image])  # (1, D)

    # feats = scaler.transform(feats)
    # feats = pca.transform(feats)

    pred, conf = model.predict(feats)
    if pred == -1:
        return "unknown", conf[0]
    return classes[pred[0]], conf[0]


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