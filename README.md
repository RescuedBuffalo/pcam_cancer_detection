# Histopathologic Cancer Detection

## Problem Statement

Identify metastatic cancer tissue in histopathology images using deep learning models. This project uses the PatchCamelyon (PCam) dataset from the Kaggle competition: https://www.kaggle.com/c/histopathologic-cancer-detection/overview

**Dataset**: 
- 262,144 training samples (96×96×3 RGB images)
- 32,768 validation samples
- 32,768 test samples
- Binary classification: tumor vs normal tissue

## Project Structure

```
├── EDA.ipynb                 # Comprehensive exploratory data analysis
├── modeling.ipynb            # Model training and evaluation
├── requirements.txt          # Python dependencies
├── MODELING_GUIDE.md         # Detailed modeling reference
├── RESIZE_GUIDE.md           # Image resizing trade-offs (96×96 vs 224×224)
├── TRAINING_STRATEGY.md      # Efficient training approach (quick mode + focused training)
├── cancer-detection-eda-plan.plan.md  # Project plan
└── data/                     # Dataset files (HDF5 and CSV)
```

## Setup

1. Create virtual environment:
```bash
pyenv virtualenv 3.12.5 cancer-env
pyenv local cancer-env
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Register Jupyter kernel:
```bash
python -m ipykernel install --user --name=cancer-env
```

## Methodology

### 1. Exploratory Data Analysis (EDA.ipynb)

**Completed comprehensive analysis including:**

1. **Class Distribution**: 50/50 balanced split (no class imbalance)
2. **Image Visualization**: Sample images from both classes
3. **Statistical Analysis**: RGB channel distributions, mean/std per class
4. **Data Quality Checks** (9 comprehensive checks):
   - Missing values: None found
   - Dimensional consistency: All 96×96×3
   - Pixel value ranges: 0-255 (uint8)
   - Outlier detection: <0.1% outliers
   - Label-metadata consistency: 99.76% match
   - Low variance detection: 1% of samples
   - Pixel saturation: 0.5% highly saturated
   - Train/validation distribution comparison: KS test
   - Duplicate detection: 4.23% similar fingerprints

5. **Data Characteristics**:
   - 216 unique WSI (Whole Slide Image) sources
   - Visible staining variation across sources
   - Spatial distribution analysis

6. **Center Region Analysis**: 99.5% of tumor patches have centered tumor
7. **Data Augmentation Preview**: Tested rotation, flip, zoom, brightness

**Key EDA Findings:**
- Clean, high-quality dataset - no preprocessing needed
- Balanced classes - no need for class weights
- Centered tumors - attention mechanisms could help
- Staining variation - strong augmentation recommended

### 2. Model Architecture (modeling.ipynb)

**Implemented Models:**

1. **Simple CNN Baseline**
   - 4 convolutional blocks with batch norm and dropout
   - Establishes performance lower bound
   - ~522K trainable parameters

2. **EfficientNet-Inspired Model**
   - Custom architecture inspired by EfficientNet
   - Optimized for 96×96 input size
   - ~522K parameters

3. **EfficientNet + CBAM Attention**
   - EfficientNet architecture with CBAM attention modules
   - Channel + Spatial attention mechanisms
   - Focuses on tumor regions (99.5% are centered)
   - ~528K trainable parameters

**Architecture Rationale:**
- **Lightweight models**: All models ~500K parameters for fast training on CPU/M3
- **Training from scratch**: 96×96 images optimized for histopathology
- **Attention mechanisms**: CBAM leverages centered tumor finding from EDA
- **Strong augmentation**: Compensates for training from scratch and handles staining variation

**Data Cleaning:**
1. Duplicate Removal: Remove images with identical statistical fingerprints (~4-5%)
2. Extreme Outlier Removal:
   - Very white images: Remove top 0.1 percentile (likely artifacts)
   - Very black images: Remove beyond 3-sigma (likely corrupted)
3. Final cleaned dataset: ~95% of original (still 250K+ images)

**Data Preprocessing:**
1. Optional Resizing: 96×96 → 224×224 (enables ImageNet transfer learning)
2. Normalization: Simple [0,1] scaling
3. Data Augmentation:
   - Rotation: 180° (rotation-invariant tissue)
   - Horizontal/vertical flips
   - Zoom: ±10%
   - Brightness: ±10% (simulates staining variation)
   - Width/height shift: ±10%
4. No class balancing needed (already balanced after cleaning)

**Training Configuration:**
- Batch size: 64 (96×96) or 32 (224×224) - auto-adjusted
- Optimizer: Adam
- Loss: Binary cross-entropy
- Primary metric: AUC-ROC (Kaggle metric)
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
