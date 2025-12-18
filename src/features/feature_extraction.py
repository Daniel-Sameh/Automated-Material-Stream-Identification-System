import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from tqdm import tqdm
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

class FeatureExtractor:
    def __init__(self, image_size=128):
        self.image_size = image_size
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.resnet = torch.nn.Sequential(*list(self.resnet.children())[:-1])
        self.resnet.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.resnet = self.resnet.to(self.device)
        print(f"Model loaded on {self.device}")

        # ImageNet normalization
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225]),
        ])
    
    def extract(self, images):
        features = []

        if isinstance(images, np.ndarray) and images.ndim == 3:
            images = [images]

        pbar = tqdm(images, desc="Extracting features", leave=False)
        for img in images:
            feat = self.extract_features_from_image(img)
            features.append(feat)
            pbar.update(1)
        pbar.close()

        return np.array(features)
    
    def extract_features_from_image(self, img):
        # img = cv2.resize(img, (self.image_size, self.image_size))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        deep_feat = self.extract_deep_features(img_rgb)
        
        img = cv2.resize(img, (self.image_size, self.image_size))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Extract comprehensive features
        hog_features = self.hog_extractor(img_rgb)
        lbp_features = self.lbp_extractor(img_rgb)
        color_features = self.color_histogram(img_rgb)
        # texture_features = self.glcm_features(img_rgb)
        # edge_features = self.edge_features(img_rgb)
        # shape_features = self.shape_features(img_rgb)
        
        # specular_features = self.specular_features(img_rgb)  
        surface_features = self.surface_properties(img_rgb) 
        # frequency_features = self.frequency_features(img_rgb)
        # fractal_features = self.fractal_dimension(img_rgb)

        # Concatenate all features
        combined = np.concatenate([
            deep_feat,
            hog_features,
            lbp_features,
            color_features,
            # texture_features,
            # edge_features,
            # shape_features,
            # specular_features,
            surface_features,
            # frequency_features,
            # fractal_features
        ])
        
        return combined
    
    def extract_deep_features(self, img):
        """Extract features using pretrained ResNet50"""
        
        # Convert to PIL Image for transforms
        img_pil = Image.fromarray(img)
        
        # Apply transforms
        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.resnet(img_tensor).squeeze().cpu().numpy()
        
        return features  # 2048 features from ResNet50
    
    def hog_extractor(self, img):
        """HOG features - shape and edge patterns"""
        hog_features = hog(
            img,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            transform_sqrt=True,
            feature_vector=True,
            channel_axis=-1
        )
        return hog_features
    
    def lbp_extractor(self, img):
        """Multi-scale LBP for texture - fast and effective"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        features = []
        # Two scales only for speed
        for radius in [1, 3]:
            n_points = 8 * radius
            lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
            n_bins = n_points + 2
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
            lbp_hist = lbp_hist.astype("float") / (lbp_hist.sum() + 1e-6)
            features.append(lbp_hist)
        
        return np.concatenate(features)
    
    def color_histogram(self, img):
        """Enhanced color features"""
        features = []
        
        # RGB histogram
        for i in range(3):
            hist = cv2.calcHist([img], [i], None, [16], [0, 256])
            hist = hist.flatten() / (hist.sum() + 1e-6)
            features.append(hist)
        
        # HSV histogram - critical for materials
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        for i in range(3):
            hist = cv2.calcHist([hsv], [i], None, [16], [0, 256])
            hist = hist.flatten() / (hist.sum() + 1e-6)
            features.append(hist)
        
        # Lab color space - better color perception
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        for i in range(3):
            hist = cv2.calcHist([lab], [i], None, [16], [0, 256])
            hist = hist.flatten() / (hist.sum() + 1e-6)
            features.append(hist)
        
        # Color moments - RGB, HSV, Lab
        for color_img in [img, hsv, lab]:
            for channel in cv2.split(color_img):
                features.append([
                    np.mean(channel),
                    np.std(channel),
                    # Skewness (3rd moment)
                    np.mean(((channel - np.mean(channel)) / (np.std(channel) + 1e-6)) ** 3)
                ])
        
        return np.concatenate(features)
    
    def glcm_features(self, img):
        """GLCM texture features - simplified for speed"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (64, 64))
        gray = (gray / 32).astype(np.uint8)
        
        # Multiple distances for better texture capture
        glcm = graycomatrix(gray, [1, 2], [0, np.pi/2], levels=8, symmetric=True, normed=True)
        
        # More comprehensive texture properties
        features = []
        for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
            features.append(graycoprops(glcm, prop).flatten())
        
        return np.concatenate(features)
    
    def edge_features(self, img):
        """Edge and gradient features - materials have different edge patterns"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Canny edges
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Sobel gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobelx**2 + sobely**2)
        
        # Gradient direction histogram
        gradient_dir = np.arctan2(sobely, sobelx)
        dir_hist, _ = np.histogram(gradient_dir, bins=8, range=(-np.pi, np.pi))
        dir_hist = dir_hist.astype(float) / (dir_hist.sum() + 1e-6)
        
        features = [
            edge_density,
            np.mean(gradient_mag),
            np.std(gradient_mag),
            np.percentile(gradient_mag, 75),  # 75th percentile
            np.percentile(gradient_mag, 90),  # 90th percentile
        ]
        features.extend(dir_hist)
        
        return np.array(features)
    
    def shape_features(self, img):
        """Shape and structural features"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Thresholding for shape analysis
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        features = []
        
        if len(contours) > 0:
            # Largest contour features
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            # Shape descriptors
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
            else:
                circularity = 0
            
            # Hu moments (rotation invariant)
            moments = cv2.moments(largest_contour)
            if moments['m00'] != 0:
                hu_moments = cv2.HuMoments(moments).flatten()
                # Log transform for better scaling
                hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
            else:
                hu_moments = np.zeros(7)
            
            features = [
                area / (self.image_size ** 2),  # Normalized area
                perimeter / (self.image_size * 4),  # Normalized perimeter
                circularity,
                len(contours),  # Number of objects
            ]
            features.extend(hu_moments[:5])  # First 5 Hu moments
        else:
            # No contours found
            features = np.zeros(9)
        
        return np.array(features)
        
    def surface_properties(self, img):
        """Estimate surface smoothness - glass/metal smooth, cardboard rough"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Laplacian variance - smoothness indicator
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_variance = laplacian.var()
        
        # Local standard deviation distribution
        kernel = np.ones((9, 9)) / 81
        local_std = cv2.filter2D((gray - cv2.filter2D(gray.astype(float), -1, kernel))**2, -1, kernel)
        
        # Smooth areas have low local std
        smooth_ratio = np.sum(local_std < 100) / local_std.size
        
        # Surface uniformity
        uniformity = 1.0 / (1.0 + lap_variance)
        
        return np.array([
            lap_variance,
            smooth_ratio,
            uniformity,
            np.mean(local_std),
            np.percentile(local_std, 90)
        ])

