from sklearn import svm
import joblib
import numpy as np
from src.config.settings import Settings
import logging

class SvmModel:
    def __init__(self, kernel='rbf', C=10):
        self.kernel = kernel
        self.C = C
        self.model = None
        self.probs = None
        self.threshold = None
        self.logger =  logging.getLogger("SvmModel") if Settings().debug else None


    def train(self, X_train, y_train):
        if self.logger:
            self.logger.info(f"Training SVM with kernel={self.kernel}, C={self.C}")
        self.model = svm.SVC(kernel=self.kernel, C=self.C, probability=True, gamma='scale')
        if self.logger:
            self.logger.info("Fitting the model...")
        self.model.fit(X_train, y_train)
        if self.logger:
            self.logger.info("Model training completed.")
        self.probs = self.model.predict_proba(X_train)
        if self.logger:
            self.logger.info("Calculating threshold for 'other' class...")
        max_probs = np.max(self.probs, axis=1)
        mean_p = np.mean(max_probs)
        std_p = np.std(max_probs)
        self.threshold = np.percentile(max_probs, 5)
        # if self.logger:
        #     self.logger.info(f"SVM trained with kernel={self.kernel}, C={self.C}")
        #     self.logger.info(f"Calculated threshold for 'other' class: {self.threshold:.4f}")
        

    def predict(self, X):
        predictions = self.model.predict(X)
        if self.threshold is not None:
            probs = self.model.predict_proba(X)
            max_probs = np.max(probs, axis=1)
            predictions = np.where(max_probs < self.threshold, -1, predictions)
          
        return predictions
    
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
        mask = predictions != -1
        
        if np.sum(mask) == 0:
            if self.logger:
                self.logger.warning("All predictions marked as unknown!")
            return 0.0
        
        # Calculate accuracy
        accuracy = np.mean(predictions[mask] == np.array(y)[mask])
        
        if self.logger:
            unknown_count = np.sum(~mask)
            total = len(predictions)
            self.logger.info(f"Unknown rejections: {unknown_count}/{total} ({unknown_count/total*100:.1f}%)")
        
        return accuracy