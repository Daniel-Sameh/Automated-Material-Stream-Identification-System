import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors


class KnnModel:
    def __init__(self, n_neighbors=5,threshold=None):
        self.n_neighbors=n_neighbors
        self.threshold=threshold
        self.model=None

    def train(self, X_train, y_train):
        self.model=KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.model.fit(X_train, y_train)
        nn=NearestNeighbors(n_neighbors=self.n_neighbors)
        nn.fit(self.model._fit_X)
        distances=nn.kneighbors(X_train)
        avg_distances=distances.mean(axis=1)
        self.threshold=np.percentile(avg_distances,95)

    def predict(self, X):
        distances, neighbors=self.model.kneighbors(X, n_neighbors=self.n_neighbors)
        predictions=self.model.predict(X)
        if self.threshold is None:
            return predictions
        else:
            avg_distance=distances.mean(axis=1)
            predictions_with_unknown=[]
            for i, pred in enumerate(predictions):
                if avg_distance[i] > self.threshold:
                    predictions_with_unknown.append("unknown")
                else:
                    predictions_with_unknown.append(pred)
            return np.array(predictions_with_unknown)


    def save(self, path):
        joblib.dump({'model': self.model, 'threshold': self.threshold}, path)

    def load(self, path):
        data=joblib.load(path)
        self.model=data['model']
        self.threshold=data.get('threshold')

    def score(self, X, y):
        return np.mean(self.predict(X) == y)
