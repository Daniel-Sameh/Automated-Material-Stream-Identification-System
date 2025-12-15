from sklearn import svm
from sklearn.svm import OneClassSVM
import joblib
import numpy as np
from src.config.settings import Settings
import logging

class SvmModel:
    def __init__(self, kernel='rbf', C=10, gamma='scale', nu=0.1):
        self.kernel = kernel
        self.C = C
        self.model = None
        # self.threshold = None
        self.gamma = gamma
        self.nu = nu
        self.logger = logging.getLogger("SvmModel")

    def train(self, X_train, y_train):
        # if self.logger:
        #     self.logger.info(f"Training SVM with kernel={self.kernel}, C={self.C}")
        
        # Train multi-class SVM (one-vs-one by default)
        self.model = svm.SVC(
            kernel=self.kernel, 
            C=self.C, 
            decision_function_shape='ovr',  # One-vs-Rest for better outlier detection
            gamma=self.gamma, 
            class_weight='balanced'
        )
        
        self.model.fit(X_train, y_train)
        
        
        # Get decision function scores for training data
        decision_scores = self.model.decision_function(X_train)
        
        # For each sample, get the maximum decision score (confidence in best class)
        if decision_scores.ndim == 1:  # Binary classification
            max_scores = np.abs(decision_scores)
        else:  # Multi-class
            max_scores = np.max(decision_scores, axis=1)
        
        # Set threshold at 5th percentile - samples below this are "far" from all classes
        self.threshold = np.percentile(max_scores, 5)
        
        if self.logger:
            self.logger.info(f"SVM trained with kernel={self.kernel}, C={self.C}")
            self.logger.info(f"Decision function threshold: {self.threshold:.4f}")
            self.logger.info(f"Min train score: {np.min(max_scores):.4f}, Max: {np.max(max_scores):.4f}")

    def predict(self, X):
        """
        Predict class labels. Returns -1 for unknown/outlier samples.
        """
        # Get decision function scores
        decision_scores = self.model.decision_function(X)
        
        # Get standard predictions
        predictions = self.model.predict(X)
        
        # Calculate max decision score for each sample
        if decision_scores.ndim == 1:  # Binary
            max_scores = np.abs(decision_scores)
        else:  # Multi-class
            max_scores = np.max(decision_scores, axis=1)
        
        # Mark samples with low decision scores as unknown (-1)
        if self.threshold is not None:
            predictions = np.where(max_scores < self.threshold, -1, predictions)
        
        return predictions
    
    def save(self, path):
        joblib.dump({
            'model': self.model, 
            'kernel': self.kernel, 
            'C': self.C, 
            'threshold': self.threshold
        }, path)

    def load(self, path):
        data = joblib.load(path)
        self.model = data['model']
        self.kernel = data['kernel']
        self.C = data['C']
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