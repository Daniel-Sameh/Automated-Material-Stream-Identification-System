import cv2
import joblib
from src.models.SVM.train_svm import SvmModel
from src.features.feature_extraction import FeatureExtractor
from src.features.pca_reducer import PCAFeatureReducer
import numpy as np
from src.config.settings import Settings


classes = Settings().classes
svm = joblib.load("models/svm.pkl")
pca = joblib.load("models/pca_reducer.pkl")['pca']
feature_extractor = FeatureExtractor()

cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
frame_count = 0
while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1
    if frame_count % 7 != 0:
        continue
    # frame = cv2.resize(frame, (224, 224))
    print(type(frame))
    features = feature_extractor.extract([frame])   # IMPORTANT: list input
    # print(f"Extracted features shape: {features.shape}")
    features = np.array(features)
    # print(f"Extracted features shape: {features.shape}")
    X_reduced = pca.transform(features)
    # print(f"Reduced features shape: {X_reduced.shape}")
    
    model = svm['model']
    prediction = model.predict(X_reduced)[0]
    label = classes[prediction] if prediction != -1 else "other"
    print(f"Predicted label: {label}")
    
    cv2.putText(frame, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Material Classifier", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty("Material Classifier", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()  # Assume this function is defined elsewhere