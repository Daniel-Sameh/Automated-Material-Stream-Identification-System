import joblib
from src.models.SVM.train_svm import SvmModel
from src.models.KNN.train_knn import KnnModel
from src.features.feature_extraction import FeatureExtractor
from src.features.pca_reducer import PCAFeatureReducer
import numpy as np
from src.config.settings import Settings
from src.pipeline.preprocess import preprocess_for_inference

classes = Settings().classes
feature_extractor = FeatureExtractor()
scaler = joblib.load("models/scaler.pkl")
pca = joblib.load("models/pca_reducer.pkl")['pca']

best_model = joblib.load("models/best.pkl")
best_model_name = best_model["best"]

if best_model_name=="svm":
    model = SvmModel()
    model.load("models/svm.pkl")
    print("Loaded SVM model.")
else:
    model = KnnModel()
    model.load("models/knn.pkl")
    print("Loaded KNN model.")


def Predict(image):
    image = preprocess_for_inference(image)
    
    # 1. Feature extraction
    features = feature_extractor.extract([image])
    features = scaler.transform(features)
    X_reduced = pca.transform(features)

    # 2. Predict
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