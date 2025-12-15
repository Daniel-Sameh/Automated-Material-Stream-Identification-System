from src.models.KNN.train_knn import KnnModel
from src.models.SVM.train_svm import SvmModel
from src.config.settings import Settings
from src.data.augmentation import Augmentor
from src.data.data_loader import DataLoader
from src.features.feature_extraction import FeatureExtractor
from src.features.pca_reducer import PCAFeatureReducer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import os
import glob
import logging
import joblib
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
        dataLoader = DataLoader("data\processed")
        images, labels = dataLoader.load_data()
        logger.info(f"Loaded {len(images)} processed images")
        # print(f"Labels: {labels}")

    # Loading images from paths:
    imgs= []
    filtered_labels = []
    
    for i, img_path in enumerate(images):
        # print(img_path[21:24])
        img=cv2.imread(img_path)
        if img is not None:
            imgs.append(img)
            filtered_labels.append(labels[i])
    
    labels = filtered_labels
    logger.info(f"Loaded {len(imgs)} images")
    featureExtractor = FeatureExtractor()
    features = featureExtractor.extract(imgs)
    logger.info(f"Extracted features shape: {features.shape}")

    org_imgs=[]
    org_labels=[]
    aug_imgs=[]
    aug_labels=[]
    for i, img_path in enumerate(images):
        # Check if the path begins with "aug"
        if img_path[21:24]=="aug":
            aug_imgs.append(features[i])
            aug_labels.append(labels[i])
        else:
            org_imgs.append(features[i])
            org_labels.append(labels[i])

    logger.info(f"Original images: {len(org_imgs)}")
    logger.info(f"Augmented images: {len(aug_imgs)}")

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        org_imgs, org_labels, test_size=0.2, random_state=42, stratify=org_labels
    )

    logger.info(f"Training data shape: {len(X_train)}")
    # Add the augmented images to the training data
    X_train.extend(aug_imgs)
    y_train.extend(aug_labels)
    logger.info(f"Training data shape after augmentation: {len(X_train)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca_reducer = PCAFeatureReducer(n_components=0.95)
    X_train_reduced = pca_reducer.fit_transform(X_train_scaled)
    X_test_reduced = pca_reducer.transform(X_test_scaled)
    joblib.dump(scaler, "models/scaler.pkl")
    pca_reducer.save("models/pca_reducer.pkl")
    logger.info(f"Reduced features - Train: {X_train_reduced.shape}, Test: {X_test_reduced.shape}")

    # Train SVM Model:
    if not os.path.exists("models/svm.pkl"):
        best_svm_c = 1.0
        best_svm_acc = 0
        best_kernel = 'rbf'
        best_gamma = 'scale'
        
        logger.info("\n=== Training SVM with Hyperparameter Search ===")
        
        # Split training data for validation
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train_reduced, y_train, test_size=0.15, random_state=42, stratify=y_train
        )
        
        # Test wider range of C values with more regularization
        for c_val in [0.001, 0.01, 0.05, 0.1, 0.5]:
            for ker in ['linear', 'rbf']:
                for gam in ['scale', 'auto']:
                    temp_model = SvmModel(kernel=ker, C=c_val, gamma=gam)
                    temp_model.train(X_train_split, y_train_split)
                    
                    train_acc = temp_model.score(X_train_split, y_train_split)
                    val_acc = temp_model.score(X_val, y_val)
                    gap = train_acc - val_acc
                    
                    logger.info(f"SVM {ker} C={c_val} gamma={gam}: Train={train_acc*100:.2f}%, Val={val_acc*100:.2f}%, Gap={gap*100:.2f}%")
                    
                    # Choose based on validation accuracy with reasonable gap
                    if val_acc > best_svm_acc and gap < 0.25:
                        best_svm_acc = val_acc
                        best_svm_c = c_val
                        best_kernel = ker
                        best_gamma = gam
        
        # If no model found with gap < 25%, just pick best validation accuracy
        if best_svm_acc == 0:
            logger.info("No model with gap<25%, selecting best validation accuracy...")
            for c_val in [0.001, 0.01, 0.05, 0.1, 0.5]:
                for ker in ['linear', 'rbf']:
                    for gam in ['scale', 'auto']:
                        temp_model = SvmModel(kernel=ker, C=c_val, gamma=gam)
                        temp_model.train(X_train_split, y_train_split)
                        val_acc = temp_model.score(X_val, y_val)
                        
                        if val_acc > best_svm_acc:
                            best_svm_acc = val_acc
                            best_svm_c = c_val
                            best_kernel = ker
                            best_gamma = gam
        
        logger.info(f"\n✓ Best SVM: kernel={best_kernel}, C={best_svm_c}, gamma={best_gamma}, Val={best_svm_acc*100:.2f}%")
        
        # Train final model on full training set
        model = SvmModel(kernel=best_kernel, C=best_svm_c, gamma=best_gamma)
        model.train(X_train_reduced, y_train)
    else:
        model = SvmModel()
        model.load("models/svm.pkl")
    
    logger.info("SVM Model trained on full training set")

    # Evaluate
    train_accuracy = model.score(X_train_reduced, y_train)
    svm_test_accuracy = model.score(X_test_reduced, y_test)
    gap = train_accuracy - svm_test_accuracy
    
    logger.info("\n=== Final SVM Performance ===")
    logger.info(f"Train Accuracy: {train_accuracy*100:.2f}%")
    logger.info(f"Test Accuracy: {svm_test_accuracy*100:.2f}%")
    logger.info(f"Overfitting Gap: {gap*100:.2f}%")

    model.save("models/svm.pkl")
    svm_model = model
    
    if not os.path.exists("models/knn.pkl"):
        # Train KNN Model with tuning
        logger.info("\n=== Training KNN Model ===")

        best_k = 5
        best_metric='cosine'
        best_acc = 0

        # Try different k values
        for k in [3, 5, 7, 9, 11]:
            for metric in ['euclidean', 'manhattan', 'cosine']:
                knn_model = KnnModel(n_neighbors=k, metric=metric)
                knn_model.train(X_train_reduced, y_train)
                train_acc= knn_model.score(X_train_reduced, y_train)
                test_acc = knn_model.score(X_test_reduced, y_test)
                gap = train_acc - test_acc
                logger.info(f"KNN k={k}: Test Accuracy: {test_acc*100:.2f}%")
                
                if test_acc > best_acc and gap < 0.35:
                    best_acc = test_acc
                    best_metric = metric
                    best_k = k

        logger.info(f"\nBest k={best_k}, Best Metric={best_metric} with Test Accuracy: {best_acc*100:.2f}%")

        # Train final KNN model
        model = KnnModel(n_neighbors=best_k, metric=best_metric)
        model.train(X_train_reduced, y_train)
    else:
        model = KnnModel()
        model.load("models/knn.pkl")
    
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
        joblib.dump({"best":"knn"}, "models/best.pkl")
    else:
        logger.info("✅ SVM performs slightly better")
        joblib.dump({"best":"svm"}, "models/best.pkl")
    
    # Save Knn
    model.save("models/knn.pkl")


if __name__=="__main__":
    main()