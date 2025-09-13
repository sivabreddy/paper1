# Medical Image Segmentation Framework - Project Summary

## Overview
This project implements a comprehensive medical image analysis framework that combines multiple deep learning architectures for medical image segmentation and classification. The system focuses on encoder-decoder architectures and hybrid optimization techniques to improve the accuracy of medical image analysis.

## Key Features

### 1. Modified SegNet Architecture
- Encoder-decoder structure with skip connections
- Batch normalization layers for stable training
- Custom hybrid loss function combining weighted cross-entropy and Dice coefficient
- Pretrained weights available for immediate use

### 2. Hybrid Feature Guided Swarm Optimization (HFGSO)
- Combines swarm intelligence with feature guidance
- Optimizes model weights during training
- Implemented in `Proposed_HFGSO_DRN/HFGSO.py`

### 3. Evaluation Metrics
- Intersection over Union (IoU)
- Dice coefficient
- Precision, Recall, Accuracy

## Project Structure

### Core Models
- `Main/Proposed_SegNet.py`: Primary segmentation model with:
  - 13 convolutional encoder blocks
  - 13 transposed convolutional decoder blocks
  - HFGSO optimization integration

- `Proposed_HFGSO_DRN/`: Hybrid optimization implementation
  - `HFGSO.py`: Optimization algorithm
  - `DRN.py`: Deep Residual Network components

### Supporting Architectures
- `DCNN/`: Custom deep convolutional neural network implementation
- `ResNet/`: Standard Residual Network implementation
- `Focal_Net/`: Simplified CNN with LeakyReLU activation
- `Panoptic_model/`: ResNet50-based model with SMOTE for class imbalance

### Data Processing Pipeline
- `Main/prepare_data.py`: Dataset preparation and ground truth conversion
- `Main/Augmentation.py`: Medical image augmentation (rotation, cropping)
- `Main/Pre_processing.py`: Complete image processing workflow
- `Main/read.py`: Feature and label data reading utilities

## Data Flow and Processing

### 1. Data Structure
The project uses a structured approach to handle medical image data:
- `Database/`: Raw input medical images
- `Database_gt/`: Ground truth annotation images
- `Main/data/im/`: Processed input images (101 PNG files)
- `Main/data/gt/`: Processed ground truth binary masks (101 PNG files)
- `Output/`: Intermediate processing results (roi, t2fcs, segmented, rotation, cropping)

### 2. Processing Pipeline
1. **Data Preparation** (`Main/prepare_data.py`):
   - Converts RGB ground truth images to binary masks
   - Resizes images to 128x128 pixels
   - Organizes data in standardized format

2. **Preprocessing** (`Main/Pre_processing.py`):
   - ROI extraction
   - T2FCS filtering (Two-Threshold Fuzzy Contrast Stretching)
   - Initial segmentation using SegNet
   - Feature extraction through histogram analysis

3. **Data Augmentation** (`Main/Augmentation.py`):
   - Geometric transformations (rotation, cropping)
   - Feature vector generation for augmented images
   - CSV export of features and labels

### 3. Ground Truth Role
Ground truth images are binary annotation masks that:
- Define regions of interest in medical images (white pixels = target, black pixels = background)
- Provide supervision for training segmentation models
- Enable performance evaluation through IoU, Dice coefficient, precision, and recall
- Guide intelligent feature extraction by distinguishing target tissue from background

## Convolutional Network Implementations

### 1. Proposed SegNet (Encoder-Decoder CNN)
- Purpose: Image segmentation to delineate regions of interest
- 13 encoder blocks (Conv2D → BatchNormalization → ReLU → MaxPooling)
- 13 decoder blocks (UpSampling2D → Conv2DTranspose → BatchNormalization → ReLU)
- Custom loss function optimizing both pixel accuracy and region overlap

### 2. DCNN (Custom Deep CNN)
- Purpose: Feature classification with hand-crafted implementation
- Custom NumPy-based convolution layer implementation
- Manual gradient computation for backpropagation
- Suitable for smaller feature vectors rather than raw images

### 3. Focal-Net (Simple CNN)
- Purpose: Baseline classification approach
- Single Conv2D layer with LeakyReLU activation
- MaxPooling for dimensionality reduction
- Simple architecture for baseline comparison

### 4. ResNet (Residual CNN)
- Purpose: Deep network training without degradation
- Residual blocks with skip connections
- Identity shortcuts that bypass layers
- Addresses vanishing gradient problem in deep networks

### 5. Panoptic Model (ResNet50-based CNN)
- Purpose: Transfer learning approach for medical image classification
- Pre-built ResNet50 backbone with pretrained capabilities
- Global average pooling instead of fully connected layers
- Leverages proven architecture from natural image domains

## User Interface
- `Main/GUI.py`: Tkinter-based GUI for selecting datasets, training parameters, and visualizing results
- `Image_GUI.py`: Separate GUI for processing individual images through the pipeline

## Getting Started

### Prerequisites
```bash
pip install tensorflow keras opencv-python scikit-image
```

### Run Segmentation
```python
from Main.Proposed_SegNet import Segnet_Segmentation
segmented = Segnet_Segmentation(input_image, original_image)
```

### Train New Model
```bash
python Main/Proposed_SegNet.py
```

## Model Architecture

### SegNet Encoder
- Stack of 13 convolutional blocks
- Each block: Conv → BatchNorm → ReLU → MaxPool

### SegNet Decoder
- Symmetric transposed convolutional blocks
- Each block: Upsample → ConvTranspose → BatchNorm → ReLU

### Optimization
- HFGSO applied to model weights
- Custom loss function combining cross-entropy and Dice coefficient

## Technical Stack
- TensorFlow/Keras for neural network implementations
- OpenCV for image processing
- NumPy for numerical computations
- Matplotlib for visualization
- Scikit-learn for data splitting and metrics
- Requires Python 3.x with specific package versions

This framework enables comprehensive medical image analysis, from precise segmentation to reliable classification, demonstrating the versatility and power of deep learning in healthcare applications.