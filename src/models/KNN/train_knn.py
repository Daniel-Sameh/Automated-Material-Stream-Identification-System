import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
import logging
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


class KnnModel:
    def __init__(self, unknown_rate=0.01, unknown_label=-1, pipeline=None):
        self.pipeline = pipeline
        self.threshold = None
        self.unknown_rate = unknown_rate
        self.unknown_label = unknown_label
        self.logger = logging.getLogger("KnnModel")

    def train(self, X_fit, y_fit, X_valid):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA()),
            ("clf", KNeighborsClassifier(weights="distance"))
        ])

        param_grid = {
            "pca__n_components": [0.95],
            "clf__n_neighbors": [3, 5, 7, 9],
            "clf__metric": ["cosine", "euclidean", "manhattan"],
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid = GridSearchCV(pipe, param_grid, cv=cv, n_jobs=-1, scoring="accuracy", verbose=2)
        grid.fit(X_fit, y_fit)

        self.pipeline = grid.best_estimator_
        self.logger.info(f"Best params: {grid.best_params_}")
        self.logger.info(f"CV best score: {grid.best_score_}")

        # Calibrate threshold on known calibration set
        Xc = self.pipeline[:-1].transform(X_valid) # scaled+pca features
        knn = self.pipeline.named_steps["clf"]
        distances, _ = knn.kneighbors(Xc, n_neighbors=knn.n_neighbors, return_distance=True)

        avg_dist = distances.mean(axis=1)

        # We reject if avg_dist > threshold.
        self.threshold = float(np.percentile(avg_dist, (1.0 - self.unknown_rate) * 100.0)) 

        self.logger.info(f"Chosen distance threshold: {self.threshold:.6f}")
        self.logger.info(f"Calibration reject rate: {(avg_dist > self.threshold).mean():.4f}")

    def predict(self, X):
        if self.pipeline is None:
            raise RuntimeError("Model not loaded/trained: self.pipeline is None")

        predictions = self.pipeline.predict(X)

        if self.threshold is None:
            return predictions

        Xt = self.pipeline[:-1].transform(X)
        knn = self.pipeline.named_steps["clf"]
        distances, _ = knn.kneighbors(Xt, n_neighbors=knn.n_neighbors, return_distance=True)
        avg_dist = distances.mean(axis=1)

        predictions = np.where(avg_dist > self.threshold, self.unknown_label, predictions)
        return predictions, avg_dist

    def save(self, path):
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "threshold": None if self.threshold is None else float(self.threshold),
                "unknown_label": int(self.unknown_label),
                "unknown_rate": float(self.unknown_rate),
            },
            path,
        )

    def load(self, path):
        data = joblib.load(path)
        self.pipeline = data.get("pipeline", None)
        self.threshold = data.get("threshold", None)
        self.unknown_label = data.get("unknown_label", -1)
        self.unknown_rate = data.get("unknown_rate", 0.01)

    def evaluate(self, X_test, y_test):
        # Evaluate on test
        test_preds = self.pipeline.predict(X_test)
        accuracy = float(np.mean(test_preds == y_test))

        # reject if avg_dist > threshold
        knn = self.pipeline.named_steps["clf"]
        Xt = self.pipeline[:-1].transform(X_test)
        test_dist, _ = knn.kneighbors(
            Xt,
            n_neighbors=knn.n_neighbors,
            return_distance=True
        )
        test_avg_dist = test_dist.mean(axis=1)

        test_preds_with_unknown = np.where(test_avg_dist > self.threshold, self.unknown_label, test_preds)

        unknown_rate = float(np.mean(test_preds_with_unknown == self.unknown_label))
        acc_with_unknwon = float(np.mean(test_preds_with_unknown == y_test))  # on known-only test, unknown counts as wrong

        self.logger.info("\n=== Final Performance (KNN) ===")
        self.logger.info(f"Test Accuracy(no unknowns): {accuracy*100:.2f}%")
        self.logger.info(f"Test Accuracy (unknown counts wrong): {acc_with_unknwon*100:.2f}%")
        self.logger.info(f"Test Unknown Rate: {unknown_rate*100:.2f}%")

        # known_mask = (test_preds_with_unknown != self.unknown_label)
        # if np.any(known_mask):
        #     acc_when_not_rejected = float(np.mean(test_preds_with_unknown[known_mask] == y_test[known_mask]))
        #     self.logger.info(f"Accuracy on non-rejected test samples: {acc_when_not_rejected*100:.2f}%")
        # else:
        #     self.logger.info("All test samples were rejected as Unknown.")

        return accuracy, acc_with_unknwon, unknown_rate