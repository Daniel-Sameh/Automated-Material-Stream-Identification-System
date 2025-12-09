from sklearn import svm
import joblib
import numpy as np

class SvmModel:
    def __init__(self, kernel='rbf', C=10):
        self.kernel = kernel
        self.C = C
        self.model = None
        self.probs = None
        self.threshold = None

    def train(self, X_train, y_train):
        self.model = svm.SVC(kernel=self.kernel, C=self.C, probability=True)
        self.model.fit(X_train, y_train)
        self.probs = self.model.predict_proba(X_train)
        max_probs = np.max(self.probs, axis=1)
        mean_p = np.mean(max_probs)
        std_p = np.std(max_probs)
        self.threshold = max(0, mean_p - 2 * std_p)
        

    def predict(self, X):
        if self.threshold is not None:
            probs = self.model.predict_proba(X)
            max_probs = np.max(probs, axis=1)
            predictions = np.array([pred if prob >= self.threshold else "other"
                                    for pred, prob in zip(predictions, max_probs)])
            return predictions
        else:
            return self.model.predict(X)
    
    def save(self, path):
        joblib.dump({'model': self.model, 'kernel': self.kernel, 'C': self.C, 'threshold': self.threshold}, path)

    def load(self, path):
        data = joblib.load(path)
        self.model = data['model']
        self.kernel = data['kernel']
        self.C = data['C']
        self.threshold = data.get('threshold', None)
    
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)