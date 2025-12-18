from src.config.settings import Settings
from src.data.augmentation import Augmentor
from src.data.data_loader import DataLoader
from src.features.feature_extraction import FeatureExtractor

from sklearn.model_selection import train_test_split
from src.models.SVM.train_svm import SvmModel
from src.models.KNN.train_knn import KnnModel

import os
import glob
import logging
import joblib
import numpy as np
import cv2


def main():
    settings = Settings()
    logging.basicConfig(level=logging.INFO if settings.debug else logging.WARNING)
    logger = logging.getLogger("Train")

    classes = settings.classes
    images, labels = [], []

    # Load/augment data and get (path, label idx)
    processed_root = "data/processed"
    if not os.path.exists(processed_root) or not os.listdir(processed_root):
        augmentor = Augmentor()
        processedImgs = []

        for cls in classes:
            class_dir = os.path.join("data/raw", cls)
            if not os.path.exists(class_dir):
                logger.info(f"Skipping {cls}: directory not found")
                continue

            image_paths = glob.glob(os.path.join(class_dir, "*.jpg")) + glob.glob(os.path.join(class_dir, "*.png"))
            if len(image_paths) == 0:
                logger.info(f"Skipping {cls}: no images found")
                continue

            processed = augmentor.process_class(cls, image_paths)
            processedImgs.extend(processed)
            logger.info(f"Processed {cls} images: {len(processed)}")

        images, labels = zip(*processedImgs)
        images, labels = list(images), list(labels)
        logger.info(f"Total processed images: {len(images)}")
    else:
        dataLoader = DataLoader(processed_root)
        images, labels = dataLoader.load_data()
        logger.info(f"Loaded {len(images)} processed images")

    # Read images + extract features
    imgs = []
    filtered_labels = []
    filtered_paths = []

    for i, img_path in enumerate(images):
        img = cv2.imread(img_path)
        if img is not None:
            imgs.append(img)
            filtered_labels.append(labels[i])
            filtered_paths.append(img_path)

    labels = filtered_labels
    images = filtered_paths

    logger.info(f"Loaded {len(imgs)} images for feature extraction")

    featureExtractor = FeatureExtractor()
    features = featureExtractor.extract(imgs)
    logger.info(f"Extracted features shape: {features.shape}")

    # Split original & augmented features
    org_X, org_y = [], []
    aug_X, aug_y = [], []

    for i, img_path in enumerate(images):
        if os.path.basename(img_path).startswith("aug_"):
            aug_X.append(features[i])
            aug_y.append(labels[i])
        else:
            org_X.append(features[i])
            org_y.append(labels[i])

    logger.info(f"Original images: {len(org_X)}")
    logger.info(f"Augmented images: {len(aug_X)}")

    org_X = np.asarray(org_X)
    org_y = np.asarray(org_y)
    aug_X = np.asarray(aug_X) if len(aug_X) else np.empty((0, features.shape[1]), dtype=features.dtype)
    aug_y = np.asarray(aug_y) if len(aug_y) else np.empty((0,), dtype=org_y.dtype)

    # Create 3 splits using originals:
    # - test: final evaluation (original distribution)
    # - valid: choose rejection threshold (original distribution)
    # - fit: used for training (we will add AUGMENTED here)
    X_train_org, X_test, y_train_org, y_test = train_test_split(
        org_X, org_y, test_size=0.2, random_state=42, stratify=org_y
    )

    X_fit_org, X_valid, y_fit_org, y_valid = train_test_split(
        X_train_org, y_train_org, test_size=0.2, random_state=42, stratify=y_train_org
    )

    svm = SvmModel()
    knn = KnnModel()

    # Add augmented images to the fit split (not to valid/test)
    if len(aug_X) > 0:
        X_fit = np.vstack([X_fit_org, aug_X])
        y_fit = np.concatenate([y_fit_org, aug_y])
    else:
        X_fit, y_fit = X_fit_org, y_fit_org

    logger.info(f"Fit set size (with aug): {len(X_fit)}")
    logger.info(f"validation set size (original only): {len(X_valid)}")
    logger.info(f"Test set size (original only): {len(X_test)}")

    # Train the models
    svm.train(X_fit, y_fit, X_valid)
    knn.train(X_fit, y_fit, X_valid)
    svm.save("models/svm.pkl")
    knn.save("models/knn.pkl")

    # Evaluate the models & Compare between them
    acc_svm, acc_svm_with_unknown, _= svm.evaluate(X_test, y_test)
    acc_knn, acc_knn_with_unknown, _= knn.evaluate(X_test, y_test)

    logger.info(f"SVM accuracy {acc_svm*100:.4f} (no unknowns)")
    logger.info(f"KNN accuracy {acc_knn*100:.4f} (no unknowns)")
    logger.info(f"SVM accuracy {acc_svm_with_unknown*100:.4f} (with unknowns)")
    logger.info(f"KNN accuracy {acc_knn_with_unknown*100:.4f} (with unknowns)")

    if acc_svm_with_unknown >= acc_knn_with_unknown:
        logger.info("SVM model is better")
        joblib.dump({"best":"svm"}, "models/bestModel.json")
    else:
        logger.info("KNN model is better")
        joblib.dump({"best":"knn"}, "models/bestModel.json")


if __name__ == "__main__":
    main()