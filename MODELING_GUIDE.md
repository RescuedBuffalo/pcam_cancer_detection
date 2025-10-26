# Modeling Guide - Histopathologic Cancer Detection

## Overview

This document provides a quick reference for the modeling approach based on EDA insights.

## EDA Key Findings

1. **Class Balance**: 50/50 split (perfectly balanced)
   - **Implication**: No need for class weights or oversampling

2. **Image Characteristics**: 96×96×3 RGB, mean intensity ~164
   - **Implication**: Simple [0,1] normalization sufficient; ImageNet stats for transfer learning

3. **Data Quality**: High-quality dataset, but some issues found
   - **Duplicate images**: 4.23% have similar statistical fingerprints
   - **Extreme outliers**: <0.1% very white/black images (artifacts)
   - **Implication**: Clean duplicates and extreme outliers before training

4. **Spatial Structure**: 99.5% of tumor patches have centered tumor
   - **Implication**: Could benefit from attention mechanisms or center-weighted architectures

5. **Data Diversity**: 216 unique WSI sources with visible staining variation
   - **Implication**: Strong augmentation needed, transfer learning recommended

## Model Architecture Strategy

### 1. Baseline Model: Simple CNN
- **Purpose**: Establish lower bound performance
- **Architecture**: 4 conv blocks with batch norm and dropout
- **Expected AUC**: 0.75-0.85

### 2. Transfer Learning: ResNet50
- **Purpose**: Leverage ImageNet pre-trained features
- **Strategy**: 
  - Phase 1: Freeze base, train classifier (higher LR: 1e-3)
  - Phase 2: Fine-tune last 30 layers (lower LR: 1e-5)
- **Expected AUC**: 0.88-0.93

### 3. Transfer Learning: EfficientNetB0
- **Purpose**: Compare with more efficient architecture
- **Strategy**: Similar two-phase approach as ResNet
- **Expected AUC**: 0.89-0.94

### 4. Novel Approach: VLM Zero-Shot (Optional)
- **Purpose**: Demonstrate multi-modal approach
- **Model**: CLIP or BiomedCLIP
- **Strategy**: Zero-shot classification using text prompts
- **Expected AUC**: 0.60-0.75 (no training)

### 5. Ensemble
- **Purpose**: Improve robustness
- **Strategy**: Weighted average of best CNN + VLM
- **Expected AUC**: Best individual + 1-2%

## Data Cleaning Strategy

Before training, apply the following cleaning steps:

### 1. Duplicate Detection and Removal
```python
# Create statistical fingerprints: mean + std for each RGB channel
fingerprint = (R_mean, R_std, G_mean, G_std, B_mean, B_std)

# Remove duplicate fingerprints (keep first occurrence)
# Expected to remove ~4-5% of training data (~10-13K images)
```

### 2. Extreme Outlier Removal
```python
# Remove very white images (artifacts/overexposure)
white_threshold = 99.9th percentile of mean intensity
# Expected to remove ~260 images

# Remove very black images (corrupted/underexposure)
black_threshold = mean - 3*std of mean intensity  
# Expected to remove ~200 images
```

### 3. Impact
- Final cleaned dataset: ~95% of original (~249K training images)
- Still maintains large dataset size for deep learning
- Improves data quality by removing noise and redundancy
- Class balance maintained after cleaning

## Data Augmentation Strategy

Based on histopathology image properties:

```python
- rotation_range=180  # Rotation invariant
- horizontal_flip=True
- vertical_flip=True
- zoom_range=0.1
- brightness_range=[0.9, 1.1]  # Simulates staining variation
- width_shift_range=0.1
- height_shift_range=0.1
```

## Training Configuration

### Hyperparameters
- **Batch size**: 64 (balance between speed and memory)
- **Learning rate**: 
  - CNN baseline: 1e-3
  - Transfer learning (frozen): 1e-3
  - Transfer learning (fine-tuning): 1e-5
- **Optimizer**: Adam (adaptive learning rate)
- **Loss**: Binary cross-entropy
- **Metrics**: Accuracy, AUC-ROC (primary)

### Callbacks
1. **EarlyStopping**: Patience=10, monitor val_auc
2. **ReduceLROnPlateau**: Factor=0.5, patience=5
3. **ModelCheckpoint**: Save best val_auc model

## Evaluation Metrics

1. **Primary**: AUC-ROC (Kaggle leaderboard metric)
2. **Secondary**: 
   - Accuracy
   - Precision/Recall
   - F1-Score
   - Confusion Matrix

## Expected Timeline

1. **Simple CNN**: ~30-40 min on GPU, ~2-3 hours on CPU
2. **ResNet50** (frozen): ~20-30 min on GPU
3. **ResNet50** (fine-tuning): ~30-40 min on GPU
4. **EfficientNet**: ~25-35 min on GPU
5. **VLM** (if implemented): ~10-15 min (no training)

## Experiment Tracking

Track the following for each experiment:
- Model architecture and parameters
- Training time per epoch
- Best validation AUC and epoch
- Final test AUC
- Training curves (loss, accuracy, AUC)

## Next Steps After Modeling

1. **Error Analysis**:
   - Visualize false positives/negatives
   - Identify patterns in misclassifications
   - Check if errors correlate with specific WSI sources

2. **Test Predictions**:
   - Load test data
   - Generate predictions with best model
   - Create submission CSV

3. **Kaggle Submission**:
   - Submit predictions
   - Capture leaderboard screenshot
   - Document final AUC score

4. **Documentation**:
   - Update README with results
   - Include training curves
   - Summarize key learnings

## Potential Improvements

If time permits:
1. **Attention Mechanisms**: Focus on center region (99.5% centered tumors)
2. **Vision Transformers**: ViT or Swin Transformer
3. **Multi-scale Analysis**: Use multiple patch sizes
4. **Stain Normalization**: Reduce color variation impact
5. **K-Fold Cross-Validation**: More robust performance estimate
6. **Test-Time Augmentation**: Average predictions over augmented versions

## Architecture Decision Rationale

### Why Transfer Learning?
- Medical images share features with natural images (edges, textures, patterns)
- Limited training time for competition
- Proven effectiveness in medical imaging tasks
- Pre-trained weights provide excellent initialization

### Why ResNet50 and EfficientNet?
- **ResNet50**: Well-established, residual connections prevent vanishing gradients
- **EfficientNet**: Better efficiency, compound scaling, similar/better performance
- Both have strong track records in medical imaging

### Why VLM Approach?
- Novel contribution to project
- Demonstrates understanding of multi-modal models
- Zero-shot capability shows model's general knowledge
- Can be ensembled with task-specific models

## Common Issues and Solutions

### Issue 1: Overfitting
- **Symptoms**: High train accuracy, low validation accuracy
- **Solutions**: Increase dropout, more augmentation, reduce model complexity

### Issue 2: Slow Training
- **Symptoms**: Long epoch times
- **Solutions**: Reduce batch size, use smaller model, check GPU utilization

### Issue 3: Low AUC
- **Symptoms**: AUC < 0.80
- **Solutions**: Check data normalization, increase model capacity, adjust learning rate

### Issue 4: Unstable Training
- **Symptoms**: Loss spikes, NaN values
- **Solutions**: Reduce learning rate, check input normalization, add gradient clipping

## File Structure

```
pcam_cancer_detection/
├── EDA.ipynb                 # Exploratory Data Analysis
├── modeling.ipynb            # Model training and evaluation
├── requirements.txt          # Python dependencies
├── cancer-detection-eda-plan.plan.md  # Project plan
├── MODELING_GUIDE.md         # This file
├── data/                     # Dataset files
│   ├── *.h5                  # Image data
│   └── *.csv                 # Metadata
└── results/                  # Model checkpoints and predictions
    ├── *.h5                  # Saved models
    └── submission.csv        # Kaggle submission
```

