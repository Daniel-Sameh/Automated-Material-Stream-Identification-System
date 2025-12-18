import os
import cv2
import joblib
import numpy as np
from src.features.feature_extraction import FeatureExtractor
from src.pipeline.preprocess import preprocess_for_inference
from src.models.SVM.train_svm import SvmModel
from src.models.KNN.train_knn import KnnModel
from src.config.settings import Settings


def predict(dataFilePath, bestModelPath):
    feature_extractor = FeatureExtractor()
    
    # Load images from folder
    image_files = []
    for filename in sorted(os.listdir(dataFilePath)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(os.path.join(dataFilePath, filename))
    
    if len(image_files) == 0:
        return []
    
    # Load model
    model_data = joblib.load(bestModelPath)
    
    if "best" in model_data:
        model_type = model_data["best"]
        if model_type == "svm":
            actual_model_path = os.path.join(os.path.dirname(bestModelPath), "svm.pkl")
            model = SvmModel()
            model.load(actual_model_path)
        else:
            actual_model_path = os.path.join(os.path.dirname(bestModelPath), "knn.pkl")
            model = KnnModel()
            model.load(actual_model_path)
    else:
        if "pipeline" in model_data:
            if hasattr(model_data["pipeline"].named_steps.get("clf"), "support_vectors_"):
                model = SvmModel()
                print("Loaded SVM.")
            else:
                model = KnnModel()
                print("Loaded KNN.")
            model.load(bestModelPath)
    
    # Process images and predict
    predictions = []
    for img_path in image_files:
        img = cv2.imread(img_path)
        if img is None:
            predictions.append(-1)
            continue
        
        # Preprocess
        img = preprocess_for_inference(img)
        
        # Extract features
        features = feature_extractor.extract([img])
        
        # Predict
        pred, _ = model.predict(features)
        predictions.append(int(pred[0]))
    
    return predictions


if __name__ == "__main__":
    dataFilePath = "test/"
    bestModelPath = "models/bestModel.json"
    predictions = predict(dataFilePath, bestModelPath)
    print(predictions)
    classes = Settings().classes
    # Print prediction with image name:
    for i, filename in enumerate(sorted(os.listdir(dataFilePath))):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            label = "unknown" if predictions[i] == -1 else classes[predictions[i]]
            print(f"{filename}: {label}")
