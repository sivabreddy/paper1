# Medical Image Segmentation Framework

This project implements multiple deep learning architectures for medical image segmentation, with a focus on encoder-decoder architectures and hybrid optimization techniques.

## Key Features

- **Modified SegNet Architecture** with:
  - Encoder-decoder structure with skip connections
  - Batch normalization layers
  - Custom hybrid loss function
  - Pretrained weights available

- **Hybrid Feature Guided Swarm Optimization (HFGSO)**:
  - Combines swarm intelligence with feature guidance
  - Optimizes model weights during training
  - Implemented in `Proposed_HFGSO_DRN/HFGSO.py`

- **Custom Loss Function**:
  - Combines weighted cross-entropy (β=0.75) 
  - Negative log of Dice coefficient
  - Defined in `Main/Proposed_SegNet.py`

- **Evaluation Metrics**:
  - Intersection over Union (IoU)
  - Dice coefficient
  - Precision, Recall, Accuracy

## Project Structure

### Core Models
- `Main/Proposed_SegNet.py`: Main segmentation model with:
  - 13 convolutional encoder blocks
  - 13 transposed convolutional decoder blocks 
  - HFGSO optimization integration

- `Proposed_HFGSO_DRN/`: Hybrid optimization implementation
  - `HFGSO.py`: Optimization algorithm
  - `DRN.py`: Deep Residual Network components

### Supporting Architectures
- `DCNN/`: Deep Convolutional Neural Network
- `ResNet/`: Residual Network 
- `Focal_Net/`: Focal Network
- `Panoptic_model/`: Panoptic segmentation

### Data Processing
- `Main/prepare_data.py`: Dataset preparation
- `Main/Augmentation.py`: Medical image augmentation
- `Main/Pre_processing.py`: Image normalization

## Getting Started

1. **Prerequisites**:
```bash
pip install tensorflow keras opencv-python scikit-image
```

2. **Download pretrained weights**:
```bash
wget [URL_TO_SEGNET_100.H5] -O segnet_100.h5
```

3. **Run segmentation**:
```python
from Main.Proposed_SegNet import Segnet_Segmentation
segmented = Segnet_Segmentation(input_image, original_image)
```

4. **Train new model**:
```bash
python Main/Proposed_SegNet.py
```

## Model Architecture

![SegNet Architecture](architecture.png)

1. **Encoder**:
   - Stack of 13 convolutional blocks
   - Each block: Conv -> BatchNorm -> ReLU -> MaxPool

2. **Decoder**:
   - Symmetric transposed convolutional blocks  
   - Each block: Upsample -> ConvTranspose -> BatchNorm -> ReLU

3. **Optimization**:
   - HFGSO applied to model weights
   - Custom loss function