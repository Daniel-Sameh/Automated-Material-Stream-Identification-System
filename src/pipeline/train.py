# train.py  (Fix A: closed-set training + separate calibration threshold)
from src.config.settings import Settings
from src.data.augmentation import Augmentor
from src.data.data_loader import DataLoader
from src.features.feature_extraction import FeatureExtractor

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC  # IMPORTANT: sklearn SVC (NO rejection here)
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

    # -----------------------------
    # 1) Load / augment data (to disk) and get (path, label_idx)
    # -----------------------------
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

    # -----------------------------
    # 2) Read images + extract features
    # -----------------------------
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

    # -----------------------------
    # 3) Split ORIGINAL vs AUGMENTED features
    # -----------------------------
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

    # -----------------------------
    # 4) Create 3 splits using ORIGINALS ONLY:
    #    - test: final evaluation (original distribution)
    #    - calib: choose rejection threshold (original distribution)
    #    - fit: used for training (we will add AUGMENTED here)
    # -----------------------------
    X_train_org, X_test, y_train_org, y_test = train_test_split(
        org_X, org_y, test_size=0.2, random_state=42, stratify=org_y
    )

    X_fit_org, X_calib, y_fit_org, y_calib = train_test_split(
        X_train_org, y_train_org, test_size=0.2, random_state=42, stratify=y_train_org
    )

    svm = SvmModel()
    knn = KnnModel()
    # Add augmented images ONLY to the fit split (not to calib/test)
    if len(aug_X) > 0:
        X_fit = np.vstack([X_fit_org, aug_X])
        y_fit = np.concatenate([y_fit_org, aug_y])
        svm.train(X_fit, y_fit, X_calib, X_test, y_test)
        knn.train(X_fit, y_fit, X_calib, X_test, y_test)
        svm.save("models/svm_open_set.pkl")
        knn.save("models/knn_open_set.pkl")
    else:
        X_fit, y_fit = X_fit_org, y_fit_org
        svm.train(X_fit, y_fit, X_calib, X_test, y_test)
        knn.train(X_fit, y_fit, X_calib, X_test, y_test)
        svm.save("models/svm_open_set.pkl")
        knn.save("models/knn_open_set.pkl")

    logger.info(f"Fit set size (with aug): {len(X_fit)}")
    logger.info(f"Calibration set size (original only): {len(X_calib)}")
    logger.info(f"Test set size (original only): {len(X_test)}")


if __name__ == "__main__":
    main()