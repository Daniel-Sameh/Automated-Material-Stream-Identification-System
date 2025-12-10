import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors


class KnnModel:
    def __init__(self, n_neighbors=5,threshold=None, metric='manhatten'):
        self.n_neighbors=n_neighbors
        self.threshold=threshold
        self.metric=metric
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, weights='distance')
        self.nn = NearestNeighbors(n_neighbors=n_neighbors)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.nn.fit(X_train)

        distances, _=self.nn.kneighbors(X_train)
        avg_distances=distances.mean(axis=1)
        self.threshold=np.percentile(avg_distances,95)

    def predict(self, X):
        distances, _=self.nn.kneighbors(X)
        predictions=self.model.predict(X)
        if self.threshold is not None:
            avg_distance = distances.mean(axis=1)
            predictions = np.where(avg_distance > self.threshold, -1, predictions)
        
        return predictions


    def save(self, path):
        joblib.dump({'model': self.model, 'threshold': self.threshold}, path)

    def load(self, path):
        data=joblib.load(path)
        self.model=data['model']
        self.threshold=data.get('threshold')

    def score(self, X, y):
        predictions = self.predict(X)
        
        # Filter out unknown predictions
        mask = predictions != -1
        if np.sum(mask) == 0:
            return 0.0
        
        return np.mean(predictions[mask] == np.array(y)[mask])
