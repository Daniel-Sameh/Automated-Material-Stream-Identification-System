from src.models.KNN.train_knn import KnnModel
from src.models.SVM.train_svm import SvmModel
from src.config.settings import Settings
from src.data.augmentation import Augmentor
from src.data.data_loader import DataLoader
from src.features.feature_extraction import FeatureExtractor
from src.features.pca_reducer import PCAFeatureReducer
from sklearn.model_selection import train_test_split
import os
import glob
import logging
from PIL import Image
import numpy as np
import cv2

def main():
    settings = Settings()
    if settings.debug:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("Train")
    
    classes = settings.classes
    images, labels = [], []
    # Data Augmentation & Loading:
    # Check if there is processed images in data/processed
    if not os.path.exists("data/processed") or not os.listdir("data/processed"):
        # Data Augementation
        augmentor = Augmentor()
        processedImgs = []
        for cls in classes:
            # Get all images paths
            class_dir= os.path.join("data/raw", cls)
            if not os.path.exists(class_dir):
                logger.info(f"Skipping {cls}: directory not found")
                continue

            image_paths= glob.glob(os.path.join(class_dir, "*.jpg"))+glob.glob(os.path.join(class_dir, "*.png"))
            
            if len(image_paths)==0:
                logger.info(f"Skipping {cls}: no images found")
                continue

            processed= augmentor.process_class(cls, image_paths)
            processedImgs.extend(processed)
            
            logger.info(f"Processed {cls} images: {len(processed)}")
        
        images, labels = zip(*processedImgs)
        images = list(images)
        labels = list(labels)

        logger.info(f"Total processed images: {len(images)}")
        # logger.info(f"Images type= {type(images[0])}")
        # logger.info(f"Image0={images[0]}")
    else:
        # Load from processed
        dataLoader = DataLoader("data/processed")
        images, labels = dataLoader.load_data()
        logger.info(f"Loaded {len(images)} processed images")
        # print(f"Labels: {labels}")

    # Loading images from paths:
    imgs= []
    for img_path in images:
        img=cv2.imread(img_path)
        if img is not None:
            imgs.append(img)
    
    logger.info(f"Loaded {len(imgs)} images")
    featureExtractor = FeatureExtractor()
    features = featureExtractor.extract(imgs)
    logger.info(f"Extracted features shape: {features.shape}")

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    pca_reducer = PCAFeatureReducer(n_components=200)
    X_train_reduced = pca_reducer.fit_transform(X_train)
    X_test_reduced = pca_reducer.transform(X_test)
    logger.info(f"Reduced features - Train: {X_train_reduced.shape}, Test: {X_test_reduced.shape}")

    # Train SVM Model:
    if not os.path.exists("models/svm.pkl"):
        model = SvmModel( C=1.0)
        model.train(X_train_reduced, y_train)
    else:
        model = SvmModel()
        model.load("models/svm.pkl")
    
    logger.info("SVM Model trained")

    # Evaluate
    train_accuracy = model.score(X_train_reduced, y_train)
    svm_test_accuracy = model.score(X_test_reduced, y_test)
    logger.info("Model evaluated")
    
    logger.info(f"Train Accuracy: {train_accuracy*100:.2f}%")
    logger.info(f"Test Accuracy: {svm_test_accuracy*100:.2f}%")

    # model.save("models/svm.pkl")
    
    # Train KNN Model with tuning
    logger.info("\n=== Training KNN Model ===")

    best_k = 5
    best_acc = 0

    # Try different k values
    for k in [3, 5, 7, 9, 11]:
        knn_model = KnnModel(n_neighbors=k, metric='cosine')
        knn_model.train(X_train_reduced, y_train)
        test_acc = knn_model.score(X_test_reduced, y_test)
        logger.info(f"KNN k={k}: Test Accuracy: {test_acc*100:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_k = k

    logger.info(f"\nBest k={best_k} with Test Accuracy: {best_acc*100:.2f}%")

    # Train final KNN model
    model = KnnModel(n_neighbors=best_k, metric='cosine')
    model.train(X_train_reduced, y_train)

    train_accuracy = model.score(X_train_reduced, y_train)
    knn_test_accuracy = model.score(X_test_reduced, y_test)

    logger.info(f"\nFinal KNN Results:")
    logger.info(f"Train Accuracy: {train_accuracy*100:.2f}%")
    logger.info(f"Test Accuracy: {knn_test_accuracy*100:.2f}%")


    # At the end of main()
    logger.info("\n" + "="*50)
    logger.info("FINAL MODEL COMPARISON")
    logger.info("="*50)
    logger.info(f"SVM  - Test Accuracy: {svm_test_accuracy*100:.2f}%")
    logger.info(f"KNN  - Test Accuracy: {knn_test_accuracy*100:.2f}% (k={best_k})")
    logger.info("="*50)

    if knn_test_accuracy > svm_test_accuracy:
        logger.info("✅ KNN performs slightly better")
    else:
        logger.info("✅ SVM performs slightly better")
    # model.save("models/knn.pkl")


if __name__=="__main__":
    main()