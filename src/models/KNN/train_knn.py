import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
import logging


class KnnModel:
    def __init__(self, n_neighbors=5,threshold=None, metric='euclidean'):
        self.n_neighbors=n_neighbors
        self.threshold=threshold
        self.metric=metric
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, weights='distance')
        self.nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
        self.logger = logging.getLogger("KnnModel")

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.nn.fit(X_train)

        distances, _=self.nn.kneighbors(X_train, n_neighbors=min(self.n_neighbors, X_train.shape[0]))
        avg_distances=distances.mean(axis=1)
        self.threshold=float(np.percentile(avg_distances,95))
        self.logger.info(f"KNN training completed. threshold= {self.threshold:.5f}")

    def predict(self, X):
        distances, _=self.nn.kneighbors(X, n_neighbors=min(self.n_neighbors, max(1, self.nn._fit_X.shape[0])))
        predictions=self.model.predict(X)
        
        # Reject samples that are too far from training data as 'unknown'
        if self.threshold is not None:
            avg_distance = distances.mean(axis=1)
            predictions = np.where(avg_distance > self.threshold, -1, predictions)
        
        return predictions


    def save(self, path):
        joblib.dump({
            'model': self.model,
            'nn': self.nn,
            'threshold': self.threshold,
            'n_neighbors': self.n_neighbors,
            'metric': self.metric
            }, path)

    def load(self, path):
        data=joblib.load(path)
        self.model = data['model']
        self.nn = data.get('nn', None)
        self.threshold = data.get('threshold', None)
        self.n_neighbors = data.get('n_neighbors', self.n_neighbors)
        self.metric = data.get('metric', self.metric)
        self.logger.info(f"KnnModel loaded from {path}")

    def score(self, X, y):
        predictions = self.predict(X)
        
        # Filter out unknown predictions for accuracy calculation
        # mask = predictions != -1
        # if np.sum(mask) == 0:
        #     return 0.0
        
        # Calculate accuracy only on non-rejected predictions
        return float(np.mean(predictions == np.array(y)))
