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
    def __init__(self, kernel='rbf', C=10, gamma='scale', nu=0.1, pipeline=None, false_reject_rate=0.01, unknown_label=-1):
        # self.kernel = kernel
        # self.C = C
        # self.model = None
        self.threshold = None
        # self.gamma = gamma
        # self.nu = nu
        self.logger = logging.getLogger("SvmModel")
        self.pipeline = pipeline
        self.false_reject_rate = false_reject_rate
        self.unknown_label = unknown_label

    def train(self, X_fit, y_fit, X_calib, X_test, y_test):
        # -----------------------------
        # Closed-set pipeline + GridSearchCV (NO rejection here)
        # -----------------------------
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA()),
            ("clf", SVC(class_weight="balanced"))  # closed-set classifier
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

        # -----------------------------
        # Calibrate Unknown threshold on calibration set (original distribution)
        #    threshold is chosen to reject only a small % of KNOWN samples.
        # -----------------------------
        calib_scores = best_pipeline.decision_function(X_calib)
        if calib_scores.ndim == 1:
            calib_max = np.abs(calib_scores)
        else:
            calib_max = np.max(calib_scores, axis=1)

        self.false_reject_rate = 0.01  # reject ~1% of known calibration samples
        self.threshold = np.percentile(calib_max, self.false_reject_rate * 100.0)

        self.logger.info(f"Chosen threshold: {self.threshold:.6f}")
        self.logger.info(f"Calibration unknown rate: {(calib_max < self.threshold).mean():.4f}")

        # -----------------------------
        # Evaluate on test (closed-set and open-set)
        # -----------------------------
        test_preds_closed = best_pipeline.predict(X_test)
        acc_closed = np.mean(test_preds_closed == y_test)

        test_scores = best_pipeline.decision_function(X_test)
        if test_scores.ndim == 1:
            test_max = np.abs(test_scores)
        else:
            test_max = np.max(test_scores, axis=1)

        test_preds_open = np.where(test_max < self.threshold, -1, test_preds_closed)

        unknown_rate = np.mean(test_preds_open == -1)
        acc_open = np.mean(test_preds_open == y_test)  # on known-only test, unknown counts as wrong

        self.logger.info("\n=== Final Performance ===")
        self.logger.info(f"Closed-set Test Accuracy: {acc_closed*100:.2f}%")
        self.logger.info(f"Open-set  Test Accuracy (unknown counts wrong): {acc_open*100:.2f}%")
        self.logger.info(f"Test Unknown Rate: {unknown_rate*100:.2f}%")

        # Optional: accuracy on non-rejected samples only (useful diagnostic)
        known_mask = test_preds_open != -1
        if np.any(known_mask):
            acc_when_not_rejected = np.mean(test_preds_open[known_mask] == y_test[known_mask])
            self.logger.info(f"Accuracy on non-rejected test samples: {acc_when_not_rejected*100:.2f}%")
        else:
            self.logger.info("All test samples were rejected as Unknown.")

    def predict(self, X):
        """
        Predict class labels. Returns -1 for unknown/outlier samples.
        """
        if self.pipeline is None:
            raise RuntimeError("Model not loaded/trained: self.pipeline is None")

        pred_closed = self.pipeline.predict(X)

        if self.threshold is None:
            return pred_closed

        scores = self.pipeline.decision_function(X)
        scores = np.asarray(scores)
        if scores.ndim == 1:
            max_score = np.abs(scores)
        else:
            max_score = np.max(scores, axis=1)

        pred_open = np.where(max_score < self.threshold, self.unknown_label, pred_closed)
        return pred_open
    
    def save(self, path):
        os.makedirs("models", exist_ok=True)
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "threshold": float(self.threshold),
                "unknown_label": -1,
                "false_reject_rate": float(self.false_reject_rate),
            },
            path,
        )

    def load(self, path):
        data = joblib.load(path)
        # self.model = data['model']
        # self.kernel = data['kernel']
        # self.C = data['C']
        self.pipeline = data.get('pipeline', None)
        self.false_reject_rate = data.get('false_reject_rate', 0.01)
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
            max_scores = np.abs(decision_scores)
        else:
            max_scores = np.max(decision_scores, axis=1)
        
        # Mark unknowns
        if self.threshold is not None:
            predictions = np.where(max_scores < self.threshold, -1, predictions)
        
        return predictions, max_scores