import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern


class FeatureExtractor:
    def __init__(self):
        return
    
    def extract(self, images):
        features = []

        if isinstance(images, np.ndarray) and images.ndim == 3:
            images = [images]

        for img in images:
            feat = self.extract_features_from_image(img)
            features.append(feat)

        return np.array(features)
    
    def extract_features_from_image(self, img):
        img = cv2.resize(img, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hogFeature = self.hog_extractor(img)
        lbpFeature = self.lbp_extractor(img)
        
        return np.concatenate([hogFeature, lbpFeature])
    
    def hog_extractor(self, img):
        hog_features = hog(
            img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            transform_sqrt=True,
            feature_vector=True
        )
        return hog_features
    
    def lbp_extractor(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        lbp = local_binary_pattern(gray, 16, 2, method='uniform')
        lbp_hist, _ = np.histogram(
            lbp.ravel(),
            bins=np.arange(0, 16+3),
            range=(0, 16+2)
        )
        lbp_hist = lbp_hist.astype("float")
        lbp_hist /= (lbp_hist.sum() + 1e-6)

        return lbp_hist

