import os
import joblib
import numpy as np
from sklearn.decomposition import PCA

class PCAFeatureReducer:
    def __init__(self, n_components=128, whiten=True, random_state=42):
        self.n_components = n_components
        self.whiten = whiten
        self.random_state = random_state

        self.pca = None
        self.fitted = False

    def fit(self, features):
        """
        Fit PCA on the training features.
        """
        features = np.array(features)

        # PCA fitting with whitening (no separate scaler needed)
        self.pca = PCA(
            n_components=self.n_components,
            whiten=self.whiten,
            random_state=self.random_state
        )
        self.pca.fit(features)

        self.fitted = True

        print(f"PCA fitted: reduced {features.shape[1]} → {self.n_components} dimensions")
        return self

    def transform(self, features):
        """
        Transform features using the fitted PCA.
        """
        if not self.fitted:
            raise RuntimeError("PCAFeatureReducer must be fitted before calling transform().")

        features = np.array(features)
        reduced = self.pca.transform(features)
        return reduced

    def fit_transform(self, features):
        self.fit(features)
        return self.transform(features)

    def save(self, path):
        """
        Save PCA object to disk.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            "pca": self.pca,
            "n_components": self.n_components
        }, path)

        print(f"PCA reducer saved to {path}")

    def load(self, path):
        """
        Load PCA from file.
        """
        obj = joblib.load(path)
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
