from sklearn import svm
from sklearn.svm import OneClassSVM
import joblib
import numpy as np
from src.config.settings import Settings
import logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
import os


class SvmModel:
    def __init__(self, kernel='rbf', C=10, gamma='scale', nu=0.1, pipeline=None, unknown_rate=0.01, unknown_label=-1):
        # self.kernel = kernel
        # self.C = C
        # self.model = None
        self.threshold = None
        # self.gamma = gamma
        # self.nu = nu
        self.logger = logging.getLogger("SvmModel")
        self.pipeline = pipeline
        self.unknown_rate = unknown_rate
        self.unknown_label = unknown_label

    def train(self, X_fit, y_fit, X_valid):
        # Closed-set pipeline + GridSearchCV (No rejection here)
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA()),
            ("clf", SVC(class_weight="balanced")) # closed-set classifier
        ])

        param_grid = {
            "pca__n_components": [0.95],
            "clf__kernel": ["rbf"],
            "clf__C": [10],
            "clf__gamma": ["scale"],
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid = GridSearchCV(self.pipeline, param_grid, cv=cv, n_jobs=-1, scoring="accuracy", verbose=2)
        grid.fit(X_fit, y_fit)

        best_pipeline = grid.best_estimator_
        self.pipeline = best_pipeline
        self.logger.info(f"Best params: {grid.best_params_}")
        self.logger.info(f"CV best score: {grid.best_score_}")

        # Calibrate Unknown threshold on validation set (original distribution)
        # threshold is chosen to reject only a small % of KNOWN samples.
        calib_scores = best_pipeline.decision_function(X_valid)
        if calib_scores.ndim == 1:
            calib_max = np.abs(calib_scores)
        else:
            calib_max = np.max(calib_scores, axis=1)

        self.unknown_rate = 0.01  # reject ~1% of known validation samples
        self.threshold = np.percentile(calib_max, self.unknown_rate * 100.0)

        self.logger.info(f"Chosen threshold: {self.threshold:.6f}")
        self.logger.info(f"validation unknown rate: {(calib_max < self.threshold).mean():.4f}")


    def predict(self, X):
        """
        Predict class labels. Returns -1 for unknown/outlier samples.
        """
        if self.pipeline is None:
            raise RuntimeError("Model not loaded/trained: self.pipeline is None")

        predictions = self.pipeline.predict(X)

        if self.threshold is None:
            return predictions

        scores = self.pipeline.decision_function(X)
        scores = np.asarray(scores)
        if scores.ndim == 1:
            max_score = np.abs(scores)
        else:
            max_score = np.max(scores, axis=1)

        # Normalize to percentage (0-100%)
        # Higher score = more confident
        if self.threshold is not None:
            # Scale relative to threshold: threshold=50%, higher=100%
            confidence_percentage = np.clip(
                (max_score / (self.threshold * 2)) * 100, 
                0, 
                100
            )
            
            # Apply unknown rejection
            predictions = np.where(
                max_score < self.threshold, 
                self.unknown_label, 
                predictions
            )
            
            # Set unknown confidence to 0%
            confidence_percentage = np.where(
                predictions == self.unknown_label,
                0.0,
                confidence_percentage
            )
        else:
            # No threshold: just scale scores to 0-100
            score_min = max_score.min()
            score_max = max_score.max()
            confidence_percentage = ((max_score - score_min) / (score_max - score_min + 1e-8)) * 100
        
        return predictions, confidence_percentage
    
    def save(self, path):
        os.makedirs("models", exist_ok=True)
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "threshold": float(self.threshold),
                "unknown_label": -1,
                "unknown_rate": float(self.unknown_rate),
            },
            path,
        )

    def load(self, path):
        data = joblib.load(path)
        # self.model = data['model']
        # self.kernel = data['kernel']
        # self.C = data['C']
        self.pipeline = data.get('pipeline', None)
        self.unknown_rate = data.get('unknown_rate', 0.01)
        self.unknown_label = data.get('unknown_label', -1)
        self.threshold = data.get('threshold', None)
    
    def score(self, X, y):
        """
        Calculate accuracy on known classes only.
        Unknown predictions (-1) are not counted as errors.
        """
        predictions = self.predict(X)
        y = np.array(y)
        
        # Separate known and unknown predictions
        # known_mask = predictions != -1
        # unknown_count = np.sum(~known_mask)
        
        # if self.logger:
            # self.logger.info(f"Unknown rejections: {unknown_count}/{len(predictions)} ({unknown_count/len(predictions)*100:.1f}%)")
        
        # Calculate accuracy only on samples the model classified (not rejected)
        # if np.sum(known_mask) == 0:
        #     if self.logger:
        #         self.logger.warning("All predictions marked as unknown!")
        #     return 0.0
        
        accuracy = np.mean(predictions == y)
        return accuracy
    
    def predict_with_confidence(self, X):
        """
        Returns predictions and their confidence scores (max decision function value).
        Useful for inference to see how confident the model is.
        """
        decision_scores = self.model.decision_function(X)
        predictions = self.model.predict(X)
        
        if decision_scores.ndim == 1:
            confidence_scores = np.abs(decision_scores)
        else:
            confidence_scores = np.max(decision_scores, axis=1)
        
        # Mark unknowns
        if self.threshold is not None:
            predictions = np.where(confidence_scores < self.threshold, -1, predictions)
        
        return predictions, confidence_scores
    
    def evaluate(self, X_test, y_test):
        # Evaluate on test
        test_preds = self.pipeline.predict(X_test)
        accuracy = np.mean(test_preds == y_test)

        test_scores = self.pipeline.decision_function(X_test)
        if test_scores.ndim == 1:
            test_max = np.abs(test_scores)
        else:
            test_max = np.max(test_scores, axis=1)

        test_preds_with_unknown = np.where(test_max < self.threshold, self.unknown_label, test_preds)

        unknown_rate = np.mean(test_preds_with_unknown == self.unknown_label)
        acc_with_unknown = np.mean(test_preds_with_unknown == y_test)  # on known-only test, unknown counts as wrong

        self.logger.info("\n=== Final Performance ===")
        self.logger.info(f"Test Accuracy(Without Unknown): {accuracy*100:.2f}%")
        self.logger.info(f"Test Accuracy (unknown counts wrong): {acc_with_unknown*100:.2f}%")
        self.logger.info(f"Test Unknown Rate: {unknown_rate*100:.2f}%")

        return accuracy, acc_with_unknown, unknown_rate