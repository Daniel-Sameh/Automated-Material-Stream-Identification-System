import os
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class PCAFeatureReducer:
    def __init__(self, n_components=128, whiten=False, random_state=42):
        self.n_components = n_components
        self.whiten = whiten
        self.random_state = random_state

        self.scaler = None
        self.pca = None
        self.fitted = False

    def fit(self, features):
        """
        Fit the StandardScaler and PCA on the training features.
        """
        features = np.array(features)

        # Standardization
        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(features)

        # PCA fitting
        self.pca = PCA(
            n_components=self.n_components,
            whiten=self.whiten,
            random_state=self.random_state
        )
        self.pca.fit(scaled)

        self.fitted = True

        print(f"PCA fitted: reduced {scaled.shape[1]} → {self.n_components} dimensions")
        return self

    def transform(self, features):
        """
        Transform features using the fitted scaler + PCA.
        """
        if not self.fitted:
            raise RuntimeError("PCAFeatureReducer must be fitted before calling transform().")

        features = np.array(features)
        scaled = self.scaler.transform(features)
        reduced = self.pca.transform(scaled)
        return reduced

    def fit_transform(self, features):
        self.fit(features)
        return self.transform(features)

    def save(self, path):
        """
        Save scaler and PCA objects to disk.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            "scaler": self.scaler,
            "pca": self.pca,
            "n_components": self.n_components
        }, path)

        print(f"PCA reducer saved to {path}")

    def load(self, path):
        """
        Load scaler + PCA from file.
        """
        obj = joblib.load(path)
        self.scaler = obj["scaler"]
        self.pca = obj["pca"]
        self.n_components = obj["n_components"]
        self.fitted = True

        print(f"PCA reducer loaded from {path}")
        return self
    
    def explained_variance(self):
        """
        Return variance explained per component.
        """
        if not self.fitted:
            raise RuntimeError("Fit the PCA first.")
        return self.pca.explained_variance_ratio_
