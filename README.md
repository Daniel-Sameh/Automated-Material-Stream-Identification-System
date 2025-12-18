# Automated Material Stream Identification System

A real-time material classification system that uses machine learning to identify and sort materials through a webcam. The system can distinguish between cardboard, glass, metal, paper, plastic, and trash with 93% accuracy.

## What This Does

Point your webcam at a piece of material, and the system tells you what it is. We built this to help automate waste sorting, making recycling more efficient. The classifier runs in real-time on your laptop and can handle different lighting conditions, camera angles, and material appearances.

We trained two models (SVM and KNN) and automatically select whichever performs better. The system also rejects items it's unsure about rather than guessing, which is crucial for real-world sorting applications.

## Features

- **Real-time classification** through webcam feed
- **93% accuracy** on test set with SVM classifier
- **Unknown rejection**, won't guess when uncertain
- **Robust to lighting variations** with preprocessing pipeline
- **Deep learning + traditional features** - combines ResNet50 with HOG, LBP, and color histograms
- **Automatic data augmentation** to balance training data

## Setup

### Requirements

You'll need Python 3.8+ and a webcam. We recommend using a virtual environment.

```bash
# Clone the repo
git clone <your-repo-url>
cd Automated-Material-Stream-Identification-System

# Create virtual environment (recommended)
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### GPU Support (Optional but Recommended)

Feature extraction runs much faster on GPU. If you have an NVIDIA GPU:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Otherwise, CPU works fine but feature extraction will be slower.

## Quick Start

### Option 1: Use Pre-trained Models

There is already trained models in the `models/` folder, you can jump straight to real-time classification:

```bash
python -m src.app.realtime_cam
```

Point your webcam at objects and you'll see predictions overlaid on the video feed. Press `q` to quit.

### Option 2: Train From Scratch

If you want to train your own models or retrain with new data:

#### 1. Prepare The Data

Organize raw images in this structure:

```
data/raw/
├── cardboard/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```

#### 2. Train the Models

```bash
python -m src.pipeline.train
```

This will:
- Augment your images to 500 per class (saved in `data/processed/`)
- Extract features using ResNet50 + traditional computer vision techniques
- Train both SVM and KNN classifiers with grid search
- Evaluate on test set and pick the best one
- Save models to `models/` folder

**Note:** The first run processes and augments all images. Subsequent runs will use cached processed images unless you delete the `data/processed/` folder.

#### 3. Run Real-time Classification

```bash
python -m src.app.realtime_cam
```

## Testing on Image Folder

If you want to test the classifier on a folder of images without the webcam:

```bash
# Put test images in test/ folder
# Then run:
python test.py
```

This will print predictions for each image. 

You can also test individual images programmatically:

```python
from test import predict

predictions = predict("path/to/images/", "models/bestModel.json")
print(predictions)  # List of class indices (-1 for unknown)
```

## Project Structure

```
.
├── data/
│   ├── raw/              # Original images
│   └── processed/        # Augmented images (generated automatically)
├── models/               # Trained models
│   ├── svm.pkl
│   ├── knn.pkl
│   └── bestModel.json
├── src/
│   ├── app/
│   │   └── realtime_cam.py    # Webcam interface
│   ├── data/
│   │   ├── augmentation.py    # Image augmentation
│   │   └── data_loader.py
│   ├── features/
│   │   └── feature_extraction.py  # ResNet50 + HOG + LBP + color
│   ├── models/
│   │   ├── SVM/
│   │   │   └── train_svm.py
│   │   └── KNN/
│   │       └── train_knn.py
│   ├── pipeline/
│   │   ├── train.py           # Main training script
│   │   ├── inference.py       # Prediction logic
│   │   └── preprocess.py      # White balance + CLAHE (for realtime camera)
│   └── config/
│       └── settings.py        # Class names and settings
├── test.py
└── requirements.txt
```

## How It Works

### Feature Extraction

We extract about 2,900 features from each image:
- **ResNet50 features (2048):** High-level patterns learned from ImageNet
- **HOG features:** Edge directions and shapes
- **LBP features:** Texture patterns at multiple scales
- **Color histograms:** RGB, HSV, and LAB color distributions
- **Surface properties:** Smoothness, roughness, uniformity

### Data Augmentation

Each class is augmented to 500 images with:
- Random cropping and resizing
- Rotation (±15°)
- Perspective transforms
- Color jittering (brightness, contrast, saturation, hue)
- Gaussian blur
- JPEG compression artifacts
- Camera sensor noise

This helps the model handle real-world webcam variations.

### Training Pipeline

1. Load raw images from `data/raw/`
2. Augment to 500 images per class
3. Extract features from all images
4. Split into training (64%), validation (16%), and test (20%)
5. Only original images go into validation/test - augmented ones only in training
6. Train SVM and KNN with 5-fold cross-validation
7. Calibrate rejection thresholds on validation set
8. Evaluate both models and pick the winner
9. Save models and best model selection

### Preprocessing (Real-time)

Before classification, each webcam frame goes through:
- **Gray world white balance:** Corrects color cast from lighting
- **CLAHE on L channel:** Stabilizes brightness variations

This makes the system work under different lighting conditions.

### Unknown Rejection

Both models can reject samples they're uncertain about. The threshold is calibrated to reject ~1% of known validation samples, which helps prevent wild guesses on out-of-distribution items.


## Model Performance

Current results with our dataset:

| Model | Accuracy | Accuracy (with unknowns) | Rejection Rate |
|-------|----------------------|---------------------------|----------------|
| SVM   | 93.03%              | 92.49%                    | 1.61%          |
| KNN   | 88.47%              | 87.40%                    | 1.07%          |

**SVM wins** and is selected as the deployment model.

## Future Improvements

- Add more material categories (aluminum vs steel, different plastics)
- Deploy to edge devices (Raspberry Pi, Jetson Nano)
- Integration with robotic sorting arms
- Active learning to improve model with user corrections
- Ensemble predictions for higher confidence

## Contributors
[Daniel Sameh](https://github.com/Daniel-Sameh)

[Abanob Essam](https://github.com/AbanobEssam19)

[Marcelino Maximos](https://github.com/Marcelino-10)

[Youssef Ehab](https://github.com/YoussefEhab2)

[Daniel Raafat](https://github.com/DanialRaafat)