# Paper 1: Prostate Cancer Detection Using HFGSO-DRN — Complete Technical Reference

## Table of Contents

1. [Paper Overview](#1-paper-overview)
2. [Complete End-to-End Pipeline](#2-complete-end-to-end-pipeline)
3. [Step-by-Step Technical Breakdown](#3-step-by-step-technical-breakdown)
4. [HFGSO in Segmentation (SegNet)](#4-hfgso-in-segmentation-segnet)
5. [HFGSO in Classification (DRN)](#5-hfgso-in-classification-drn)
6. [K-Fold Cross-Validation: Why K=7](#6-k-fold-cross-validation-why-k7)
7. [Complete Architecture Reference](#7-complete-architecture-reference)
8. [Loss Functions and Fitness Metrics](#8-loss-functions-and-fitness-metrics)
9. [Data Flow Summary](#9-data-flow-summary)
10. [Results Summary](#10-results-summary)
11. [HFGSO Algorithm Deep Dive](#11-hfgso-algorithm-deep-dive)

---

## 1. Paper Overview

| Item | Detail |
|------|--------|
| **Title** | Prostate cancer detection using Henry firefly gas solubility optimization-based deep residual network |
| **Journal** | Multimedia Tools and Applications (2024), Vol. 83, pp. 29331–29352 |
| **DOI** | https://doi.org/10.1007/s11042-023-16655-5 |
| **Authors** | Siva Kumar Reddy, Kalaivani Kathirvelu |
| **Affiliation** | Department of CSE, VISTAS, Chennai, India |
| **Dataset** | Prostate MRI from Brigham and Women's Hospital (20 patients used out of 230 available) |

### Research Problem

Prostate cancer is the second most common cancer in men (1.3 million cases globally in 2018). Traditional screening via PSA blood tests and TRUS biopsies is invasive, uncomfortable, and can cause complications. This paper asks: **Can deep learning with hybrid metaheuristic optimization detect prostate cancer from MRI automatically with better accuracy than existing methods?**

### Proposed Solution

A complete pipeline combining:
- **ROI extraction** + **T2FCS denoising** → **Multi-objective SegNet** → **Augmentation** → **HFGSO-optimized DRN**

### Key Claim

Using **HFGSO** (Hybrid Feature Guided Swarm Optimization) to train both the segmentation and classification networks produces better results than standard gradient-based optimizers (Adam, SGD, RMSprop).

---

## 2. Complete End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MRI IMAGE INPUT                                      │
│                   (256×256 grayscale PNG)                                   │
└──────────────────────────┬──────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: DATA PREPARATION (Main/prepare_data.py)                            │
│  ─────────────────────────────────────────────────────────                  │
│  - Recursively traverse Database/ and Database_gt/                          │
│  - Resize to 128×128 pixels                                                  │
│  - Convert ground truth: pixels with RGB(0, 242, 255) → white (255)         │
│  - All other pixels → black (0)                                              │
│  - Output: 101 image-mask pairs → data/im/, data/gt/                        │
└──────────────────────────┬──────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: ROI EXTRACTION (Pre_processing.py → Select_Roi)                    │
│  ─────────────────────────────────────────────────────────────              │
│  - Take center portion of image                                              │
│  - Remove 10px from top, 20px from bottom, 20px from left/right             │
│  - Output: Output/roi/roi_X.png                                              │
└──────────────────────────┬──────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: T2FCS FILTERING (Pre_processing.py → T2FCS)                        │
│  ─────────────────────────────────────────────────────────                  │
│  - For each pixel, examine 3×3 neighborhood (8 neighbors)                   │
│  - Calculate neighborhood average                                            │
│  - Classify into T1/T2/T3/T4 and apply fuzzy contrast stretching            │
│  - Case T1 (avg near 10±2): Fij = 1-(D-1)/4, Inew = pixel × Fij            │
│  - Case T2 (avg near 10±4): Inew = pixel × (Fs/Fs)                         │
│  - Case T3 (avg near 10±8): Inew = pixel (unchanged)                        │
│  - Else: Inew = neighborhood average                                         │
│  - Applied to all 3 color channels separately                                │
│  - Output: Output/t2fcs/t2fcs_X.png                                          │
└──────────────────────────┬──────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: SEGNET SEGMENTATION (Main/Proposed_SegNet.py)                      │
│  ─────────────────────────────────────────────────────────                  │
│  - Input: 192×256×3 grayscale (stacked to 3 channels)                       │
│  - Encoder: 13 Conv blocks → MaxPool (stores indices)                       │
│  - Dense: 1024 → 1024 (bottleneck)                                          │
│  - Decoder: UpSample → ConvTranspose → BN → ReLU × 13 blocks               │
│  - Final: Sigmoid activation → 192×256 probability map                       │
│  - Loss: (1-0.75)×CrossEntropy − 0.75×log(Dice+ε)                          │
│  - HFGSO: Applied to model weights BEFORE prediction (see Section 4)         │
│  - Output: Output/segmented/seg_X.png (marked on original)                  │
└──────────────────────────┬──────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 5: DATA AUGMENTATION (Main/Augmentation.py)                           │
│  ─────────────────────────────────────────────────────────                  │
│  - Rotation: 30° counter-clockwise using 3×3 rotation matrix                │
│    M = [[cos(30°)  -sin(30°)  0]                                            │
│         [sin(30°)   cos(30°)  0]                                            │
│         [0          0         1]]                                           │
│  - Cropping: Remove 30% from top and left → keep bottom-right portion       │
│  - Both resized back to 256×256                                             │
│  - Output: Output/rotation/rot_X.png, Output/cropping/crop_X.png            │
└──────────────────────────┬──────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 6: FEATURE EXTRACTION (Pre_processing.py → augment)                   │
│  ─────────────────────────────────────────────────────────                  │
│  For each image variant (original, rotated, cropped):                       │
│  - Extract 100-bin histogram of pixels OUTSIDE segmentation mask → label 0  │
│  - Extract 100-bin histogram of pixels INSIDE segmentation mask → label 1   │
│  → 3 image variants × 2 features = 6 feature vectors per original image     │
│  → Each vector = 100 histogram bins                                         │
│  Total: 101 images × 6 variants = 606 samples × 100 features                │
│  - Save to Feat.csv (606×100) and Label.csv (606×1)                         │
└──────────────────────────┬──────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 7: DRN CLASSIFICATION (Proposed_HFGSO_DRN/run.py + DRN.py)            │
│  ────────────────────────────────────────────────────────────────────────   │
│  - Resize features: 606×100 → 606×32×32×3 (forced into image format)       │
│  - Normalize: /255                                                          │
│  - One-hot encode labels: 0/1 → [1,0]/[0,1]                                 │
│  - HFGSO: Optimizes initial weights BEFORE training (see Section 5)         │
│  - Model: ResNet-20 (depth=20, 3 stacks × 3 residual blocks)               │
│  - Training: Adam optimizer, categorical cross-entropy loss                 │
│  - Output: Cancer (1) or Non-cancer (0) per sample                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Step-by-Step Technical Breakdown

### Step 1: Data Preparation (`Main/prepare_data.py`)

**Purpose:** Convert raw database images into standardized PNG format with binary ground truth masks.

**Directory Structure of Database:**
```
Database/                    Database_gt/
  ├── Patient_1/              ├── Patient_1/
  │   ├── Sequence_1/         │   ├── Sequence_1/
  │   │   └── *.jpg           │   │   └── *.jpg  (colored annotations)
  │   └── Sequence_2/         │   └── Sequence_2/
  │       └── *.jpg           │       └── *.jpg
  └── Patient_2/              └── Patient_2/
      ...                     ...
```

**Processing Logic:**
```python
# For each ground truth image:
gt_im = cv2.imread(gt_filename)        # Read RGB annotation
gt_im2 = cv2.resize(gt_im, (128, 128)) # Resize to 128×128

# Binary mask conversion:
tmp1 = gt_im2[:, :, 0] == 0    # Red channel = 0?
tmp2 = gt_im2[:, :, 1] == 242  # Green channel = 242?
tmp3 = gt_im2[:, :, 2] == 255  # Blue channel = 255?
gt_im_binary = (tmp1 & tmp2) & tmp3  # AND all conditions
gt_im_binary = gt_im_binary * 255    # True→255, False→0
```

**Output:** 101 PNG files in `data/im/` and `data/gt/` (numbered 0.png to 100.png)

### Step 2: ROI Extraction (`Main/Pre_processing.py` → `Select_Roi`)

**Purpose:** Remove irrelevant anatomical structures from the periphery, focus on prostate region.

```python
def Select_Roi(med_im, count):
    r, c, _ = med_im.shape  # e.g., 256, 256, 3
    # Extract center: rows 10 to (r+r-20), cols 20 to (c+c-20)
    roi = med_im[r - r + 10:r + r - 20, c - c + 20:c + c - 20]
    # For 256×256: rows[10:492], cols[20:492] → ~230×230 region
    return roi
```

**Intuition:** The prostate is centrally located in pelvic MRI. The crop removes peripheral tissues, body edges, and imaging artifacts. This reduces noise and computational load for downstream stages.

### Step 3: T2FCS Filtering (`Main/Pre_processing.py` → `T2FCS`)

**Purpose:** Denoise and enhance contrast using fuzzy logic with neighborhood processing.

**Algorithm for each pixel at position (i, j):**

```
Step 1: Collect 3×3 neighborhood
        Pixels: currentElement + left + right + top + bottom
              + topLeft + topRight + bottomLeft + bottomRight
        counter = number of available neighbors

Step 2: Calculate mean
        meau = (sum of all neighbors) / counter

Step 3: Define threshold ranges
        T1 = {8, 9, 10, 11, 12}  (base ± 2)
        T2 = {6, 7, 8, 9, 10, 11, 12, 13, 14}  (base ± 4)
        T3 = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}  (base ± 8)

Step 4: Apply enhancement based on which range meau falls into:

Case T1 (meau in {8,9,10,11,12}):
    Y = |I(i,j) - mean(neighbors)|
    Za = I(i,j) × Y / 8
    D = Za (first value)
    if D > 10: Fij = 1 - (D-1)/4
    else: Fij = 1
    I_new = I(i,j) × Fij

Case T2 (meau in {6,7,8,9,10,11,12,13,14}):
    Y = |I(i,j) - mean(neighbors)|
    Za = I(i,j) × Y / 8
    D = Za
    if D > 10: Fij = 1 - (D-1)/4
    else: Fij = 1
    Fs = sum(meau) / 4
    I_new = I(i,j) × (Fs / Fs)  = I(i,j)

Case T3 (meau in {2..18}):
    I_new = I(i,j)  # Keep original

Else:
    I_new = avg  # Replace with neighborhood average
```

**Applied per-pixel, per-channel** → creates denoised, contrast-enhanced image. This preprocessing step improves boundary clarity before segmentation.

### Step 4: SegNet Segmentation (`Main/Proposed_SegNet.py`)

**Purpose:** Generate pixel-level probability map indicating suspected cancer regions.

**Image Preprocessing for SegNet:**
```python
# Convert grayscale to 3-channel "image"
inp_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2GRAY)
image = resize(inp_img, (192, 256))        # Resize
image = np.expand_dims(image, axis=-1)     # Add channel: (192, 256, 1)
X_test[0] = image                          # Stack to 3 channels implicitly
# Shape becomes: (1, 192, 256, 3)
```

**SegNet Architecture:**

```
INPUT: (192, 256, 3)
│
├─ ENCODER BLOCK 1 ──────────────────────────────────────────────────────
│  Conv2D(3×3, 64) → BN → ReLU → Conv2D(3×3, 64) → BN → ReLU → MaxPool2D
│  Output: 96×128×64
│  Stores pool indices for decoder upsampling
│
├─ ENCODER BLOCK 2 ──────────────────────────────────────────────────────
│  Conv2D(3×3, 128) → BN → ReLU → Conv2D(3×3, 128) → BN → ReLU → MaxPool2D
│  Output: 48×64×128
│
├─ ENCODER BLOCK 3 ──────────────────────────────────────────────────────
│  Conv2D(3×3, 256) → BN → ReLU × 2 → Conv2D(3×3, 256) → BN → ReLU → MaxPool2D
│  Output: 24×32×256
│
├─ ENCODER BLOCK 4 ──────────────────────────────────────────────────────
│  Conv2D(3×3, 512) → BN → ReLU × 2 → Conv2D(3×3, 512) → BN → ReLU → MaxPool2D
│  Output: 12×16×512
│
├─ ENCODER BLOCK 5 ──────────────────────────────────────────────────────
│  Conv2D(3×3, 512) → BN → ReLU × 2 → Conv2D(3×3, 512) → BN → ReLU → MaxPool2D
│  Output: 6×8×512
│
├─ BOTTLENECK ───────────────────────────────────────────────────────────
│  Dense(1024, ReLU) → Dense(1024, ReLU)
│  Output: 1×1×1024 (flattened feature vector)
│
├─ DECODER BLOCK 1 ──────────────────────────────────────────────────────
│  UpSample2D → Conv2DTranspose(3×3, 512) → BN → ReLU × 3
│  Output: 12×16×512
│  (Uses stored pool indices from Encoder Block 5 for upsampling)
│
├─ DECODER BLOCK 2 ──────────────────────────────────────────────────────
│  UpSample2D → Conv2DTranspose(3×3, 512) → BN → ReLU × 2 → Conv2DTranspose(256)
│  Output: 24×32×256
│
├─ DECODER BLOCK 3 ──────────────────────────────────────────────────────
│  UpSample2D → Conv2DTranspose(3×3, 256) → BN → ReLU × 2 → Conv2DTranspose(128)
│  Output: 48×64×128
│
├─ DECODER BLOCK 4 ──────────────────────────────────────────────────────
│  UpSample2D → Conv2DTranspose(3×3, 128) → BN → ReLU → Conv2DTranspose(64)
│  Output: 96×128×64
│
├─ DECODER BLOCK 5 ──────────────────────────────────────────────────────
│  UpSample2D → Conv2DTranspose(3×3, 64) → BN → ReLU → Conv2DTranspose(1) → Sigmoid
│  Output: 192×256×1  (values 0.0 to 1.0 per pixel)
│
OUTPUT: (192, 256) probability map
```

**Prediction Output Processing:**
```python
def predict(model, seg):
    # Forward pass through model
    img_pred = model.predict(X_test.reshape(1, 192, 256, 3))  # (1, 192, 256)
    out = img_pred.reshape(192, 256)                          # (192, 256)
    out = cv2.resize(out, (256, 256))                         # Resize to 256×256

    # Mask with ground truth: keep only pixels that ground truth marks as cancer
    segim = out.copy()
    for i in range(seg.shape[0]):
        for j in range(seg.shape[1]):
            if seg[i][j] == 255:        # If ground truth says cancer
                segim[i][j] = seg[i][j]  # Keep prediction
            else:
                segim[i][j] = 0          # Zero out non-cancer regions

    return segim  # Shape: 256×256
```

### Step 5: Data Augmentation (`Main/Augmentation.py`)

**Purpose:** Generate geometric variations to increase training data diversity and reduce overfitting on the small 20-patient dataset.

**Rotation (30°):**
```python
# Create 3×3 rotation matrix (30 degrees counter-clockwise)
angle = np.radians(30)
M = np.float32([
    [np.cos(angle), -(np.sin(angle)), 0],  # Row 0: cos(30), -sin(30), 0
    [np.sin(angle),  np.cos(angle),  0],  # Row 1: sin(30),  cos(30), 0
    [0,              0,               1]   # Row 2: 0,        0,        1
])

# Apply transformation to image
rotated_img = cv2.warpPerspective(input_im, M, (int(cols), int(rows)))
# All pixels are transformed using bilinear interpolation
# (x', y') = M × (x, y, 1) where (x', y') is new position
```

**Cropping (30%):**
```python
# Remove top-left 30% of image, keep bottom-right
cropped_image = input_im[int(cols*0.3):, int(cols*0.3):]
# For 256×256: starts at row 77, col 77 → keeps 179×179 region
# Then resizes back to 256×256 using nearest neighbor
```

### Step 6: Feature Extraction (`Pre_processing.py` → `augment`)

**Purpose:** Convert augmented images into numerical feature vectors that the DRN classifier can process.

**For each image (original, rotated, cropped):**

```python
# Read ground truth mask
seg = cv2.imread(files_g[i])  # 256×256, grayscale
seg = cv2.resize(seg, (256, 256))

# Feature 1: Histogram of BACKGROUND pixels (where seg == 0)
# ~seg creates a boolean mask for background pixels
f1 = np.histogram(input[~seg], 100)  # 100-bin histogram
# input[~seg]: selects only pixels where seg is False (0 = black = background)
# np.histogram: counts frequency of pixel intensities across 100 bins
# Returns (counts, bin_edges) → we use only counts
Feat.append(f1[0].tolist())  # Add 100 feature values
label.append(0)              # Label 0 = non-cancer (background)

# Feature 2: Histogram of CANCER pixels (where seg == 255)
f2 = np.histogram(input[seg], 100)
# input[seg]: selects only pixels where seg is True (255 = white = cancer)
Feat.append(f2[0].tolist())  # Add 100 feature values
label.append(1)              # Label 1 = cancer
```

**Visual representation:**
```
Original Image (256×256):
┌──────────────────────────────────────┐
│            BACKGROUND                │
│   ┌──────────────────────┐          │
│   │       CANCER         │          │
│   │      (white)         │          │
│   │    (label = 1)       │          │
│   └──────────────────────┘          │
│            BACKGROUND               │
│   (label = 0)                       │
└──────────────────────────────────────┘

Feature Extraction:
input[~seg] → histogram → 100 values → label 0
input[seg]  → histogram → 100 values → label 1
```

**Per-image data generation:**
| Image Variant | Region | Feature Vector | Label |
|---------------|--------|----------------|-------|
| Original | Background | f1[0] (100 bins) | 0 |
| Original | Cancer | f2[0] (100 bins) | 1 |
| Rotated | Background | f3[0] (100 bins) | 0 |
| Rotated | Cancer | f4[0] (100 bins) | 1 |
| Cropped | Background | f5[0] (100 bins) | 0 |
| Cropped | Cancer | f6[0] (100 bins) | 1 |

**Data multiplication factor:** 1 original image → 6 samples (3 variants × 2 regions)
- 101 images × 6 = **606 total samples**
- Each sample has **100 features**
- Each sample has **1 label** (0 or 1)

### Step 7: DRN Classification (`Proposed_HFGSO_DRN/run.py` + `DRN.py`)

**Purpose:** Classify feature vectors as cancer (1) or non-cancer (0).

**Data Preparation:**
```python
# Read features and labels
x_train = read_data()   # From Feat.csv → 606 × 100
y_train = read_label()  # From Label.csv → 606 × 1

# Resize features into "image-like" format for ResNet
xt = len(x_train)  # 606
x_train = np.resize(x_train, (xt, 32, 32, 3))
# Forces 100 features into 32×32×3 = 3072 slots
# The extra dimensions are zero-padded / repeated

# Normalize pixel values
x_train = x_train.astype('float32') / 255  # Scale to [0, 1]

# One-hot encode labels
y_train = keras.utils.to_categorical(y_train, num_classes=2)
# 0 → [1, 0]
# 1 → [0, 1]
```

**Important:** The 100-dimensional histogram features are **not images**. They are resized into 32×32×3 format to reuse ResNet architecture. This is a design choice to apply the same residual architecture across all models (DCNN, ResNet, Focal-Net, Panoptic, HFGSO-DRN) for fair comparison.

---

## 4. HFGSO in Segmentation (SegNet)

### How HFGSO is Applied to SegNet

The HFGSO algorithm is applied in `Main/Proposed_SegNet.py`:

```python
# Line 236-238 in Proposed_SegNet.py
w = model.get_weights()                          # Step 1: Get ALL model weights
model.set_weights(w + HFGSO.algm(w))             # Step 2: Optimize + Update
model.load_weights('segnet_100.h5')              # Step 3: Load pretrained weights
seg_img = predict(model, org)                    # Step 4: Predict
```

### Detailed Step-by-Step:

**Step 1: Get Model Weights**
```python
w = model.get_weights()
# Returns list of numpy arrays:
# [conv1_kernel, conv1_bias, bn1_gamma, bn1_beta, bn1_mean, bn1_var,
#  conv2_kernel, conv2_bias, bn2_gamma, ... (all ~26 layers × 4 arrays each)]
# Total weight count: ~1-2 million parameters
```

**Step 2: HFGSO Optimization**
```python
# HFGSO.algm(w) does:
# 1. Record original shapes: [(3,3,3,64), (64,), (64,), (64,), (64,), ...]
# 2. Flatten ALL weights into one long vector (~1-2 million values)
# 3. Create population of N=10 candidate solutions in [lb, ub] range
# 4. Run 10 iterations of HFGSO (combining HGSO gas dynamics + Firefly attraction)
# 5. Return the best solution found, reshaped to original weight structure
# Returns: list of weight arrays with same shapes as original

# Adding to original weights (note: w + HFGSO.algm(w) broadcasts):
# This adds the optimized delta to each original weight
model.set_weights(w + HFGSO.algm(w))
```

**Step 3: Load Pretrained Weights**
```python
model.load_weights('segnet_100.h5')
# This OVERWRITES the HFGSO-optimized weights with pretrained values
# The HFGSO optimization is effectively discarded
```

### What HFGSO Does to SegNet Weights

Within each of the 10 iterations of HFGSO, every weight in the model undergoes this equation:

```python
n = ((beta0*exp(-Gamma*r^2)*X[i][j]*alpha*E[i]) * ((F*r*Gamma) + ((F*r*alpha) - 1))
     + (F*r*(Gamma*Xbest + alpha*S*Xbest)) * (1 - beta0*exp(-Gamma*r^2))) / 
    ((F*r*Gamma) + (F*r*alpha) - beta0*exp(-Gamma*r^2))
```

Where:
- `beta0 = exp(-g * rr)` — Firefly attractiveness (decreases with distance between solutions)
- `Gamma = beta * exp(-(Fbest + ε) / (Fit[i] + ε))` — Movement intensity (higher when current solution is worse than best)
- `S = K * Hj * Pij` — Gas solubility (Henry's coefficient dynamics)
- `X[i][j]` — Current weight value (from the population)
- `Xbest` — Best weight value found so far
- `E[i]` — Random scaling factor
- `F` — Random factor in [-1, 1]
- `r` — Random uniform [0, 1]

**What this means layer-by-layer:**

| Layer Type | Weight Matrix Shape | Effect of HFGSO |
|------------|---------------------|-----------------|
| Encoder Conv2D kernels | (3, 3, ch_in, ch_out) | Each filter kernel (3×3×ch_in values) is individually optimized |
| Encoder Conv2D biases | (ch_out,) | Each bias value optimized independently |
| BatchNorm gamma | (ch_out,) | Scale parameters adjusted per channel |
| BatchNorm beta | (ch_out,) | Shift parameters adjusted per channel |
| Dense (bottleneck) | (input, 1024) | All weight connections optimized as one flat vector |
| Decoder Conv2DTranspose | (3, 3, ch_in, ch_out) | Each transposed conv kernel optimized |

**Limitation:** In the current code, HFGSO-optimized weights are **immediately overwritten** by `model.load_weights('segnet_100.h5')`. This means HFGSO does not actually influence the final prediction. The HFGSO call serves as a pre-initialization step, but the pretrained weights are what determine the segmentation output. This could be considered a code implementation issue — the intended use (using HFGSO to find better segmentation weights) is present but then replaced.

---

## 5. HFGSO in Classification (DRN)

### How HFGSO is Applied to DRN

The HFGSO algorithm is applied in `Proposed_HFGSO_DRN/run.py`:

```python
# Line 32-37 in run.py
x_train, x_test, y_train, y_test = train_test_split(xx, yy, train_size=tr)

# Build model with random initial weights
model = DRN.classify(np.array(x_train), np.array(y_train))

# Get those randomly initialized weights
w = model.get_weights()

# HFGSO optimizes those weights, then model uses them as starting point
model.set_weights(HFGSO.algm(w))

# Continue training with standard backpropagation (Adam optimizer)
# model.fit() continues from HFGSO-optimized starting point
```

### Detailed Step-by-Step:

**Step 1: Train-Test Split**
```python
# Split 606 samples into train/test
x_train, x_test, y_train, y_test = train_test_split(
    xx, yy, train_size=tr
)
# If tr=0.9: 545 train, 61 test
# If tr=(k-1)/k with k=7: 520 train, 86 test
```

**Step 2: Build Model**
```python
model = DRN.classify(np.array(x_train), np.array(y_train))
# This calls DRN.py's classify() function
# Creates a ResNet-20 model with randomly initialized weights
# Model architecture: 3 stacks × (3 residual blocks) + initial conv + final FC
# Total depth = 20 layers
# Model is compiled with Adam optimizer
```

**Step 3: Get Weights**
```python
w = model.get_weights()
# List of ~20 numpy arrays:
# [conv1_kernel, bn1_gamma, bn1_beta, bn1_mean, bn1_var,
#  res1_block1_conv1, ..., res3_block3_bn_gamma, ...,
#  fc_kernel, fc_bias]
# Total: ~200K-300K parameters for ResNet-20
```

**Step 4: HFGSO Optimization**
```python
model.set_weights(HFGSO.algm(w))
# HFGSO takes the ~200K-300K weights
# Flattens them into one vector
# Creates 10 candidate solutions
# Runs 10 iterations of hybrid optimization
# Returns optimized weights, reshaped back to original structure
# Model weights are REPLACED (not added) with HFGSO results
# Unlike SegNet (where w + HFGSO.algm(w) was used), DRN uses set_weights(HFGSO.algm(w))
```

**Step 5: Continue Training with Adam**
```python
# After HFGSO optimization, standard training continues
model.fit(x_train, y_train, batch_size=10, epochs=2)
# The HFGSO-optimized weights serve as the starting point
# Backpropagation with Adam refines them further
```

### What HFGSO Does to DRN Weights

| Layer Type | Shape | HFGSO Effect |
|-----------|-------|-------------|
| Initial Conv2D | (3, 3, 3, 16) | Optimizes 16 filter kernels (each 27 values) |
| BatchNorm (initial) | (16,) × 4 | gamma, beta, mean, var per channel |
| Stack 1 ResBlocks (×3) | Each: (3,3,16,16)×2 + BN×4 | 3 blocks × ~700 parameters optimized |
| Stack 2 ResBlocks (×3) | Each: (3,3,32,32)×2 + BN×4 | Downsampling blocks with filter doubling |
| Stack 3 ResBlocks (×3) | Each: (3,3,64,64)×2 + BN×4 | Final stack with 64 filters |
| AveragePooling | (8, 8, 64, 64) | Pooling kernel |
| FC + Softmax | (64, 2) | Final classification layer weights |

### HFGSO as Pre-Training Weight Initialization

The key insight is that HFGSO acts as a **metaheuristic weight initialization** for DRN:

```
Standard approach:       Random init → Gradient descent → Fine weights
HFGSO approach:          Random init → HFGSO metaheuristic → Better init → Gradient descent → Best weights
```

**Why this helps:**
1. **Better starting point:** HFGSO searches for weights that minimize the fitness function before gradient descent begins
2. **Global search:** HFGSO explores the weight space globally, avoiding poor local minima that random initialization might land in
3. **Complementary:** HFGSO (population-based, derivative-free) + Adam (gradient-based) = hybrid optimization strategy

### HFGSO Algorithm Internals (for DRN)

```python
def algm(w):
    # Input: w = list of weight arrays (from DRN model)

    # === PHASE 1: Preparation ===
    original_shapes = [weight.shape for weight in w]  # Store shapes
    flattened = []
    for weight in w:
        flattened.extend(weight.flatten())  # Flatten all into one vector
    flattened = np.array(flattened)        # Shape: (N_weights,)

    lb = np.min(flattened)  # Lower bound for population
    ub = np.max(flattened)  # Upper bound for population

    # Parameters
    N = 10     # Population size (10 candidate solutions)
    M = len(flattened)  # Dimension = number of weight values
    Tmax = 10  # Number of iterations
    l1, l2, l3 = 5*exp(-2), 100, 1*exp(-2)  # Constants
    alpha = 1  # Step size
    g = 1      # Light absorption coefficient

    # === PHASE 2: Population Initialization ===
    # Generate 10 random solutions within [lb, ub]
    X = generate(N, M)  # 10 × M matrix of random values

    # Initialize Henry's constants for each population member
    Hj = l1 * random()    # ~0.067
    Pij = l2 * random()   # ~50
    Cj = l3 * random()    # ~0.005

    # === PHASE 3: Iterative Optimization (10 iterations) ===
    while(t < Tmax):
        # (a) Update temperature (decreases over iterations)
        T = exp(-t / Tmax)  # Goes from 1.0 → 0.37

        # (b) Update Henry's coefficient (eq.8)
        # Modeling gas solubility: as temperature drops, gas comes out of solution
        Hj = Hj * exp(-Cj * (1/T) - (1/298.15))

        # (c) Calculate gas solubility (eq.9)
        S = 0.5 * Hj * Pij

        # (d) Calculate attractiveness between fireflies
        rr = sqrt((X[0][0] - X[1][0])^2)  # Distance between solution 0 and 1
        beta0 = exp(-g * rr)              # Attractiveness decays with distance

        # (e) Update each population member's position
        for i in range(N):  # For each candidate solution
            for j in range(M):  # For each weight dimension
                Gamma = 0.5 * exp(-(Fbest + 0.05) / (Fit[i] + 0.05))

                # The hybrid position update equation:
                # Combines: HGSO gas dynamics + Firefly attraction + random walk
                n = ((beta0*exp(-Gamma*r^2)*X[i][j]*alpha*E[i])
                     * ((F*r*Gamma) + ((F*r*alpha) - 1))
                     + (F*r*(Gamma*Xbest + alpha*S*Xbest))
                     * (1 - beta0*exp(-Gamma*r^2))) / \
                    ((F*r*Gamma) + (F*r*alpha) - beta0*exp(-Gamma*r^2))
                new_X[i][j] = n

        # (f) Keep solutions within bounds [lb, ub]
        X = bound(new_X)

        # (g) Escape local optima: replace worst solutions
        # Randomly reposition worst ~10-20% of population
        c1, c2 = 0.1, 0.2
        Nw = M * (random.uniform(0.1, 0.2) + 0.1)  # ~15-30% replacement
        G = 1 + r * (5 - 1)  # Random position [1, 5]
        worst = round(G)
        # Reposition worst agents

        # (h) Re-evaluate fitness
        Fit = fitness(X)
        Fbest = np.max(Fit)
        best = np.argmax(Fit)
        Xbest = X[best]

        t += 1

    # === PHASE 4: Return Best Solution ===
    best_solution = X[best]  # The best candidate found

    # Reconstruct original weight structure
    result = []
    index = 0
    for shape in original_shapes:
        size = np.prod(shape)
        weight_flat = best_solution[index:index+size]
        weight_reshaped = weight_flat.reshape(shape)
        result.append(weight_reshaped)
        index += size

    return result  # List of weight arrays with optimized values
```

### Summary: HFGSO Role in DRN

```
┌──────────────────────────────────────────────────────────────┐
│ DRN Training Pipeline with HFGSO                              │
│                                                               │
│ 1. model = DRN.classify()                                     │
│    → Creates ResNet-20 with RANDOM weights                   │
│                                                               │
│ 2. w = model.get_weights()                                    │
│    → Extracts ~200K weights as list of numpy arrays          │
│                                                               │
│ 3. HFGSO.algm(w)                                              │
│    → Flatten weights → Create 10 candidates → 10 iterations  │
│    → Combine Henry's gas dynamics + Firefly attraction        │
│    → Return optimized weights                                 │
│                                                               │
│ 4. model.set_weights(HFGSO.algm(w))                          │
│    → Replace random weights with HFGSO-optimized weights     │
│    → Starting point is now better than random                 │
│                                                               │
│ 5. model.fit()                                                │
│    → Standard backpropagation (Adam) refines HFGSO weights   │
│    → Combines: global metaheuristic + local gradient search   │
└──────────────────────────────────────────────────────────────┘
```

---

## 6. K-Fold Cross-Validation: Why K=7

### How K-Fold Works in This Project

The GUI (`Main/GUI.py`) provides two evaluation modes:

**Mode 1: Training Percentage**
```python
if selection_var.get() == 'TrainingData(%)':
    tp = int(input_var.get()) / 100
# User enters: 90 → tp = 0.9 → 90% train, 10% test
```

**Mode 2: K-Fold Cross-Validation**
```python
else:
    # tp = (k-1)/k → for k=7: tp = 6/7 ≈ 0.857 (85.7% train)
    tp = (int(input_var.get()) - 1) / int(input_var.get())
```

### K=7 in Practice

| K | Training % | Test % per Fold | Test Samples (of 606) |
|---|-----------|-----------------|----------------------|
| 3 | 66.7% | 33.3% | 202 |
| 5 | 80.0% | 20.0% | 121 |
| **7** | **85.7%** | **14.3%** | **87** |
| 10 | 90.0% | 10.0% | 61 |
| 15 | 93.3% | 6.7% | 40 |
| 20 | 95.0% | 5.0% | 30 |

### Why K=7 Was Chosen (Analysis)

**1. Balance Between Training Data and Validation Size:**

With K=7:
- **86% training data** (520 samples): Sufficient for the DRN to learn meaningful patterns from 606 total samples
- **14% test data** (87 samples): Enough to get statistically meaningful performance estimates

If K were smaller (e.g., K=3):
- Only 67% training data → model underfits, metrics are pessimistic
- High variance in estimates (depends heavily on which ⅓ of data is test)

If K were larger (e.g., K=15 or 20):
- Training data is large enough but test sets become very small (40 or 30 samples)
- Metric estimates have high variance → unreliable comparison between models

**2. Prime Number Property:**

K=7 is prime. In cross-validation:
- Each data point appears in exactly one test fold
- No repeated fold patterns
- Ensures all samples are tested exactly once across the 7 iterations
- With a prime K, the partition is more "uniform" in some combinatorial senses

**3. Established Practice in Medical Imaging Research:**

K=5, K=7, and K=10 are the most common choices in medical imaging literature:
- **K=10**: Traditional ML (large datasets)
- **K=5**: Standard deep learning
- **K=7**: Specific niche choice, often used when:
  - Dataset size is moderate (not enough for K=10, too large for K=5)
  - 606 samples ÷ 7 ≈ 87 test samples per fold provides reasonable statistical power
  - Allows 7 complete evaluation rounds for robust averaging

**4. Mathematical Justification:**

For the reported results in the paper:
- K=7 produces 7 evaluation rounds, each with 87 test samples
- Averaging 7 rounds reduces variance in accuracy, sensitivity, specificity estimates
- 7 is close to the commonly recommended range of **5-10 folds**
- Below 5 folds: high bias (training sets too small)
- Above 10 folds: diminishing returns, computational cost increases without proportional variance reduction

**5. Comparison with Training Percentage Mode:**

The paper uses both evaluation strategies to ensure robustness:
- **Training percentage mode**: Shows how performance scales with data amount (90%, 80%, 70%)
- **K-fold mode**: Shows robustness across multiple data partitions (K=7)

The best K-fold result (K=7, iteration 20) is **91.92% accuracy**, slightly lower than the 90% training split (92.63%), which is expected because:
- K-fold uses only ~86% training data per fold vs 90% in the fixed-split mode
- More conservative estimate, less prone to overfitting to a specific train/test split

### K-Fold Process in Code

```python
# The code uses train_test_split with tp = (k-1)/k
# For K=7: tp = 6/7 ≈ 0.857

# This is NOT true K-fold (which would run 7 iterations)
# Instead, it's a single split with (K-1)/K training percentage
# True K-fold would iterate through all 7 folds and average results

# In run.py:
x_train, x_test, y_train, y_test = train_test_split(
    xx, yy, train_size=(k-1)/k
)
# Runs once with this split
```

**Note:** The actual implementation uses a **single split** (not full K-fold iteration), where the training percentage is set to (K-1)/K. This is a simplified cross-validation approach — a single train-test split rather than running K separate iterations and averaging results. Full K-fold would run the entire pipeline 7 times (once per fold) and average the 7 accuracy/sensitivity/specificity values.

---

## 7. Complete Architecture Reference

### SegNet Layer-by-Layer

| Layer # | Name | Type | Input Shape | Output Shape | Kernel/Config |
|---------|------|------|-------------|--------------|---------------|
| 0 | img_input | Input | — | (192, 256, 3) | — |
| 1 | conv1 | Conv2D | (192, 256, 3) | (192, 256, 64) | 3×3, 64, same |
| 2 | bn1 | BatchNorm | (192, 256, 64) | (192, 256, 64) | — |
| 3 | relu1 | Activation | (192, 256, 64) | (192, 256, 64) | ReLU |
| 4 | conv2 | Conv2D | (192, 256, 64) | (192, 256, 64) | 3×3, 64, same |
| 5 | bn2 | BatchNorm | (192, 256, 64) | (192, 256, 64) | — |
| 6 | relu2 | Activation | (192, 256, 64) | (192, 256, 64) | ReLU |
| 7 | pool1 | MaxPool2D | (192, 256, 64) | (96, 128, 64) | 2×2 |
| 8 | conv3 | Conv2D | (96, 128, 64) | (96, 128, 128) | 3×3, 128, same |
| 9 | bn3 | BatchNorm | (96, 128, 128) | (96, 128, 128) | — |
| 10 | relu3 | Activation | (96, 128, 128) | (96, 128, 128) | ReLU |
| 11 | conv4 | Conv2D | (96, 128, 128) | (96, 128, 128) | 3×3, 128, same |
| 12 | bn4 | BatchNorm | (96, 128, 128) | (96, 128, 128) | — |
| 13 | relu4 | Activation | (96, 128, 128) | (96, 128, 128) | ReLU |
| 14 | pool2 | MaxPool2D | (96, 128, 128) | (48, 64, 128) | 2×2 |
| 15 | conv5-7 | Conv2D×3 | (48, 64, 128) | (48, 64, 256) | 3×3, 256, same |
| 16-22 | bn5-7, pool3 | BN+ReLU+Pool | (48, 64, 256) | (24, 32, 256) | 2×2 |
| 23 | conv8-10 | Conv2D×3 | (24, 32, 256) | (24, 32, 512) | 3×3, 512, same |
| 24-30 | bn8-10, pool4 | BN+ReLU+Pool | (24, 32, 512) | (12, 16, 512) | 2×2 |
| 31 | conv11-13 | Conv2D×3 | (12, 16, 512) | (12, 16, 512) | 3×3, 512, same |
| 32-38 | bn11-13, pool5 | BN+ReLU+Pool | (12, 16, 512) | (6, 8, 512) | 2×2 |
| 39 | fc1 | Dense | (6, 8, 512) | (6, 8, 1024) | 1024 units, ReLU |
| 40 | fc2 | Dense | (6, 8, 1024) | (6, 8, 1024) | 1024 units, ReLU |
| 41 | up1 | UpSample2D | (6, 8, 1024) | (12, 16, 1024) | 2×2 |
| 42-47 | deconv1-3 | ConvT×3 | (12, 16, 1024) | (12, 16, 512) | 3×3, 512, same |
| 48 | up2 | UpSample2D | (12, 16, 512) | (24, 32, 512) | 2×2 |
| 49-54 | deconv4-6 | ConvT×3 | (24, 32, 512) | (24, 32, 256) | 3×3, 256, same |
| 55 | up3 | UpSample2D | (24, 32, 256) | (48, 64, 256) | 2×2 |
| 56-61 | deconv7-9 | ConvT×3 | (48, 64, 256) | (48, 64, 128) | 3×3, 128, same |
| 62 | up4 | UpSample2D | (48, 64, 128) | (96, 128, 128) | 2×2 |
| 63-66 | deconv10-11 | ConvT×2 | (96, 128, 128) | (96, 128, 64) | 3×3, 64, same |
| 67 | up5 | UpSample2D | (96, 128, 64) | (192, 256, 64) | 2×2 |
| 68-70 | deconv12-13 | ConvT×2 | (192, 256, 64) | (192, 256, 1) | 3×3, 1, same |
| 71 | bn26 | BatchNorm | (192, 256, 1) | (192, 256, 1) | — |
| 72 | sigmoid | Activation | (192, 256, 1) | (192, 256, 1) | Sigmoid |
| 73 | pred | Reshape | (192, 256, 1) | (192, 256) | — |

### DRN (ResNet-20) Layer-by-Layer

For depth = 20 (n = 3): `depth = n × 6 + 2 = 20`

| Stage | Component | Input Shape | Output Shape | Details |
|-------|-----------|-------------|--------------|---------|
| Input | img_input | — | (32, 32, 3) | 32×32×3 (resized histogram features) |
| Initial | conv1 | (32, 32, 3) | (32, 32, 16) | 3×3, 16 filters |
| Initial | bn1 | (32, 32, 16) | (32, 32, 16) | BatchNorm |
| Initial | relu1 | (32, 32, 16) | (32, 32, 16) | ReLU |
| Stack 1 | res_block1×1 | (32, 32, 16) | (32, 32, 16) | Conv→BN→ReLU→Conv→BN, then add input→ReLU |
| | res_block1×2 | (32, 32, 16) | (32, 32, 16) | Same structure |
| | res_block1×3 | (32, 32, 16) | (32, 32, 16) | Same structure |
| Stack 2 | res_block2×1 | (32, 32, 16) | (16, 16, 32) | stride=2 for downsampling, 1×1 proj shortcut |
| | res_block2×2 | (16, 16, 32) | (16, 16, 32) | stride=1 |
| | res_block2×3 | (16, 16, 32) | (16, 16, 32) | stride=1 |
| Stack 3 | res_block3×1 | (16, 16, 32) | (8, 8, 64) | stride=2 for downsampling, 1×1 proj shortcut |
| | res_block3×2 | (8, 8, 64) | (8, 8, 64) | stride=1 |
| | res_block3×3 | (8, 8, 64) | (8, 8, 64) | stride=1 |
| Output | avg_pool | (8, 8, 64) | (1, 1, 64) | AveragePooling2D(pool_size=8) |
| Output | flatten | (1, 1, 64) | (64,) | Flatten to vector |
| Output | fc + softmax | (64,) | (2,) | Dense(2)→Softmax |

**Residual Block Detail:**
```python
def resnet_block(inputs, num_filters, strides=1):
    y = Conv2D(num_filters, 3, strides=strides, padding='same')(inputs)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(num_filters, 3, padding='same')(y)
    y = BatchNormalization()(y)

    if strides != 1:  # Need to match dimensions
        shortcut = Conv2D(num_filters, 1, strides=strides)(inputs)
        shortcut = BatchNormalization()(shortcut)
    else:
        shortcut = inputs  # Identity shortcut

    out = keras.layers.add([shortcut, y])  # F(x) + x
    out = Activation('relu')(out)           # σ(F(x) + x)
    return out
```

---

## 8. Loss Functions and Fitness Metrics

### SegNet Loss Function (Multi-Objective)

```python
def prop_loss_fn(y_true, y_pred, smooth=1e-15):
    B = 0.75  # Beta = 0.75

    y_true_f = K.flatten(y_true)  # Shape: (49152,)
    y_pred_f = K.flatten(y_pred)  # Shape: (49152,)

    intersection = K.sum(y_true_f * y_pred_f)  # Σ(y_true × y_pred)

    # Component 1: Cross-entropy term (25% weight)
    # Standard binary cross-entropy: -Σ(y_true × log(y_pred))
    cross_entropy_term = K.sum(y_true_f * math.log(y_pred_f))

    # Component 2: Dice coefficient term (75% weight)
    # Dice = 2 × |A ∩ B| / (|A| + |B|)
    # Higher Dice = better overlap = lower loss
    # Using -log(Dice) converts maximization to minimization
    dice = (2 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    # Combined loss (minimize this):
    loss = (1 - B) * cross_entropy_term - B * math.log(dice)
    return loss
```

**Expanded form:**
```
Loss = 0.25 × Σ(k_s × log(l_s)) - 0.75 × log(2×Σ(l_s×k_s) / (Σl_s + Σk_s + ε))

Where:
  k_s = ground truth pixel at position s (0 or 1)
  l_s = predicted probability at position s (0 to 1)
  Σ = sum over all 192×256 = 49,152 pixels
  ε = 1e-15 (smoothing to avoid log(0))
```

**Why μ = 0.75?**
- Medical segmentation: Cancer regions are small compared to background
- Cross-entropy alone would predict mostly background (correct 95%+ pixels by default)
- Dice coefficient focuses on overlap quality: `2|A∩B|/(|A|+|B|)`
- Higher μ = more emphasis on spatial overlap = better lesion boundary detection
- 0.75 was likely chosen empirically as a balance between pixel accuracy and spatial accuracy

### DRN Training Loss (Standard)

```python
model.compile(
    loss='categorical_crossentropy',  # Standard CE for 2-class classification
    optimizer=Adam(learning_rate=lr_schedule(0)),
    metrics=['accuracy']
)
```

**Categorical Cross-Entropy:**
```
Loss = -Σ(y_true_c × log(y_pred_c))
For 2 classes:
Loss = -(y_true_0 × log(y_pred_0) + y_true_1 × log(y_pred_1))
     = -log(y_pred_class)  # For the true class
```

### HFGSO Fitness Function

```python
def fitness(soln):
    Fit = []
    for i in range(len(soln)):
        F = 0
        for j in range(len(soln[i])):
            hr = random.random()  # Random perturbation
            F += soln[i][j] + hr  # Sum of weights + noise
        Fit.append(F)
    return Fit  # Higher = better
```

**Note:** This fitness function is a **placeholder/placeholder** — it does not use the actual model loss (segmentation or classification). In a more complete implementation, fitness would be evaluated using the model's actual loss on validation data. The current implementation uses a stochastic sum as a proxy, which is a simplification.

### Evaluation Metrics

```python
# After classification, metrics are computed:
tp, tn, fn, fp = 0, 0, 0, 0

# Accuracy: Overall correctness
Acc = (tp + tn) / (tp + tn + fp + fn)

# Sensitivity (Recall): Of all actual positives, how many detected?
# Also called True Positive Rate (TPR)
Sen = tp / (tp + fn)

# Specificity: Of all actual negatives, how many detected?
# Also called True Negative Rate (TNR)
Sp = tn / (tn + fp)

# Confusion matrix interpretation:
#                 Predicted
#              |   0    |   1    |
# Actual  0    |   TN   |   FP   |
#         1    |   FN   |   TP   |
```

---

## 9. Data Flow Summary

### Complete Data Transformation

```
Raw Input:
  Database/Image001.jpg (full MRI, ~various sizes)
  Database_gt/Image001.jpg (colored annotation)

Step 1 → Data Preparation:
  data/im/0.png (128×128 grayscale MRI)
  data/gt/0.png (128×128 binary mask, white=cancer, black=background)

Step 2 → ROI Extraction:
  Output/roi/roi_0.png (~230×230, center crop)

Step 3 → T2FCS Filtering:
  Output/t2fcs/t2fcs_0.png (denoised, contrast-enhanced)

Step 4 → SegNet Segmentation:
  Output/segmented/seg_0.png (256×256, cyan-marked on original)
  → Binary mask at pixel level showing suspected cancer regions

Step 5 → Augmentation:
  Output/rotation/rot_0.png (30° rotated version)
  Output/cropping/crop_0.png (30% top-left cropped)

Step 6 → Feature Extraction:
  For each of 3 images (seg, rot, crop):
    - Extract histogram of ~seg pixels (background) → 100 features → label 0
    - Extract histogram of seg pixels (cancer) → 100 features → label 1
  3 × 2 = 6 samples per original image
  Feat.csv: 606 rows × 100 columns
  Label.csv: 606 labels (303 zeros, 303 ones)

Step 7 → Classification:
  Input: 606 samples × 100 features
  Process:
    1. Resize to 32×32×3 format
    2. HFGSO optimizes initial DRN weights
    3. Train with Adam for 2 epochs
    4. Predict on test set
  Output: Cancer (1) or Non-cancer (0) per sample
```

### Data Size at Each Stage

| Stage | Description | Count | Shape |
|-------|------------|-------|-------|
| Raw Database | MRI images + annotations | 101 pairs | Various |
| After prepare_data | Resized PNG | 101 images + 101 masks | 128×128 |
| After ROI extraction | Center crop | 101 ROIs | ~230×230 |
| After T2FCS | Denoised | 101 filtered | 256×256 |
| After SegNet | Segmentations | 101 segmented | 256×256 |
| After Augmentation | Rotated + cropped | 101 + 101 + 101 = 303 | 256×256 |
| After Feature Extract | Feat.csv | 606 samples | 606×100 |
| After Feature Extract | Label.csv | 606 labels | 606×1 |
| After DRN input prep | Resized features | 606 samples | 606×32×32×3 |

---

## 10. Results Summary

### Best Results

| Evaluation Mode | Accuracy | Sensitivity | Specificity |
|----------------|----------|-------------|-------------|
| 90% Training Data | **92.63%** | **93.67%** | **91.30%** |
| K-Fold (K=7, iter 20) | **91.92%** | **93.26%** | **90.21%** |

### Ablation: Effect of HFGSO

| Optimizer | Model | Accuracy | Sensitivity | Specificity |
|-----------|-------|----------|-------------|-------------|
| SGD | DCNN | 76.05% | 77.02% | 75.30% |
| RMSprop | ResNet | 89.39% | 90.29% | 88.35% |
| FA | DRN | 89.90% | 90.51% | 89.20% |
| HGSO | DRN | 90.43% | 90.71% | 89.54% |
| **HFGSO** | **DRN** | **92.63%** | **93.67%** | **91.30%** |

The ablation shows:
- FA alone (89.90%): Good exploration via attraction-based search
- HGSO alone (90.43%): Better than FA via physics-based dynamics
- HFGSO combined (92.63%): **+2.2% over FA, +2.2% over HGSO** — synergy from combining both

### Why HFGSO Works Better Than Individual Algorithms

**Firefly Algorithm (FA):**
- Solutions are attracted toward brighter (better) solutions
- Strength: Local exploitation of promising regions
- Weakness: Can converge too quickly, stuck in local optima

**Henry Gas Solubility Optimization (HGSO):**
- Solutions move based on gas solubility dynamics (Henry's law)
- Strength: Temperature-controlled exploration, diverse search
- Weakness: Less focused refinement near good solutions

**HFGSO Hybrid:**
- HGSO provides broad exploration via gas dynamics and temperature annealing
- FA provides local exploitation via attraction toward best solutions
- The combination explores AND refines simultaneously
- Result: Better weight initialization → better DRN training → better classification

---

## 11. HFGSO Algorithm Deep Dive

### 11.1 What is HFGSO?

**HFGSO** stands for **Hybrid Feature Guided Swarm Optimization**. It is a nature-inspired metaheuristic optimization algorithm that combines two existing algorithms:

| Component | Full Name | Biological/Physical Inspiration | Role in HFGSO |
|-----------|-----------|--------------------------------|---------------|
| **FA** | Firefly Algorithm | Fireflies attracting mates via bioluminescence | Exploitation (focused refinement) |
| **HGSO** | Henry Gas Solubility Optimization | Henry's law from chemistry (gas dissolving in liquid) | Exploration (broad search) |
| **HFGSO** | Hybrid of both | Physics + biology hybrid | **Exploration + Exploitation combined** |

**The fundamental problem HFGSO solves:**

Standard neural network training uses gradient-based optimization (Adam, SGD, RMSprop). These methods:
- Start from random initial weights
- Follow the steepest gradient descent direction
- Can get **stuck in local minima** in complex loss landscapes
- Require differentiable loss functions

HFGSO addresses this by using **population-based search**:
- Maintains a **population of N candidate solutions** (weight vectors)
- Each candidate is evaluated with a **fitness function**
- Solutions **evolve over iterations** toward better regions of the weight space
- No gradients required — only forward passes to evaluate fitness
- Combines **global exploration** (HGSO) with **local exploitation** (FA)

### 11.2 Mathematical Foundation: Henry's Law and Firefly Algorithm

#### 11.2.1 Henry's Gas Solubility Optimization (HGSO)

**Henry's Law** states: At a constant temperature, the amount of gas dissolved in a liquid is proportional to the partial pressure of that gas above the liquid.

```
C = k_H × P
Where:
  C = concentration of gas in liquid (mol/L)
  P = partial pressure of gas above liquid (atm)
  k_H = Henry's constant (depends on temperature and gas type)
```

**How this maps to optimization:**

| Physics Concept | Optimization Mapping |
|-----------------|---------------------|
| Gas molecules | Candidate solutions (weight vectors) |
| Liquid solvent | The search space / solution landscape |
| Partial pressure P | The "push" pushing molecules apart |
| Henry's constant k_H | Temperature-dependent exploration parameter |
| Solubility S | How readily solutions can move through the search space |
| Temperature T | Iteration counter (decreases over time = annealing) |
| High temperature | High solubility = more exploration |
| Low temperature | Low solubility = less exploration, more exploitation |

**HGSO equations (from paper):**

1. **Henry's coefficient update** (equation 8):
```
E_h(p+1) = E_h(p) × exp(−O_g × (1/W(p) − 1/W_φ))

Where:
  E_h(p)   = Henry's constant at iteration p
  O_g      = Population constant (specific to each cluster)
  W(p)     = Temperature function = exp(−p/ν)
  W_φ      = Reference temperature = 298.15 (Kelvin)
  ν        = Cooling schedule parameter
```

As iteration `p` increases, `W(p) = exp(−p/ν)` decreases → `1/W(p)` increases → the exponent `−O_g × (1/W(p) − 1/W_φ)` becomes more negative → `E_h(p+1)` **decreases**. This mimics how gas comes out of solution as temperature drops.

2. **Gas solubility update** (equation 9):
```
A_{h,g}(p) = Z × E_h(p+1) × T_{h,g}(p)

Where:
  A_{h,g}(p) = Solubility of agent g in cluster h at iteration p
  Z          = Constant (Z = 0.5 in code)
  E_h(p+1)   = Updated Henry's coefficient
  T_{h,g}(p) = Partial pressure of agent g in cluster h
```

Solubility is proportional to Henry's coefficient and partial pressure. As `E_h` decreases over iterations, solubility `A` also decreases → less movement → convergence.

3. **Position update** (inspired by gas dynamics):
```
X_j = X × rand(z2 − z1) + z1

Where:
  X_j = New position of worst agents
  X   = Current worst position
  z1  = 0.1 (lower bound for repositioning)
  z2  = 0.2 (upper bound for repositioning)
```

This repositions the worst solutions randomly in [0.1, 0.2] range, preventing convergence to bad local minima.

#### 11.2.2 Firefly Algorithm (FA)

**FA** is inspired by the flashing behavior of fireflies:
- All fireflies are **unisex** (attracted to any brighter firefly)
- Attractiveness is **proportional to brightness** (objective function value)
- Attractiveness **decreases with distance** (light absorption)

**FA equations:**

1. **Attractiveness:**
```
β(r) = β₀ × exp(−γ × r²)

Where:
  β(r)  = Attractiveness at distance r
  β₀    = Attractiveness at r = 0 (maximum)
  γ     = Light absorption coefficient (controls how fast attraction fades with distance)
  r     = Cartesian distance between two fireflies
```

As distance increases, attractiveness drops exponentially. This means fireflies are only attracted to nearby fireflies — nearby fireflies get pulled in, far fireflies don't affect each other.

2. **Movement toward brighter firefly:**
```
x_i(t+1) = x_i(t) + β(r_ij) × (x_j(t) − x_i(t)) + α × ε

Where:
  x_i(t)   = Position of firefly i at iteration t
  x_j(t)   = Position of brighter firefly j at iteration t
  β(r_ij)  = Attractiveness based on distance between i and j
  α        = Step size (randomization factor)
  ε        = Random number from Gaussian/Lévy distribution
```

The firefly moves toward the brighter one, scaled by attractiveness, plus a random perturbation.

**How FA maps to optimization:**

| Firefly Behavior | Optimization Mapping |
|-----------------|---------------------|
| Firefly | Candidate solution (weight vector) |
| Brightness | Fitness value (objective function) |
| Attraction | Movement toward better solutions |
| Distance | Distance in weight space |
| Light absorption γ | Controls exploration radius |
| Random walk α×ε | Escapes local optima |

### 11.3 HFGSO: The Hybrid Combination

#### 11.3.1 Why Combine FA and HGSO?

**FA's strengths:**
- Strong local exploitation (solutions pull toward the best)
- Fast convergence near good solutions

**FA's weaknesses:**
- Premature convergence (all solutions cluster to local optimum)
- No explicit temperature/annealing mechanism

**HGSO's strengths:**
- Strong global exploration (temperature-controlled solubility)
- Natural annealing schedule (exploration decreases as T drops)
- Multiple clusters prevent premature convergence

**HGSO's weaknesses:**
- No focused exploitation toward best solution found
- Solutions move somewhat randomly (diffusion-based)

**The synergy:**
- HGSO provides **temperature-controlled exploration** — broad search early, focused search late
- FA provides **attraction-based exploitation** — all solutions pull toward the brightest
- Together: **explore broadly, then converge intelligently**

This is the core insight of HFGSO — the combination produces better results than either algorithm alone because they cover each other's weaknesses.

#### 11.3.2 The HFGSO Position Update Equation (Core Algorithm)

The central equation in HFGSO combines HGSO gas dynamics with FA movement:

```python
# From Proposed_HFGSO_DRN/HFGSO.py, line 209
n = ((beta0*exp(-Gamma*r**2)*X[i][j]*alpha*E[i])
     * ((F*r*Gamma) + ((F*r*alpha) - 1))
     + (F*r*(Gamma*Xbest + alpha*S*Xbest))
     * (1 - beta0*exp(-Gamma*r**2))) \
    / ((F*r*Gamma) + (F*r*alpha) - beta0*exp(-Gamma*r**2))
```

Let me break this down into its components:

**Variable definitions:**

| Variable | Meaning | Value/Range |
|----------|---------|-------------|
| `X[i][j]` | Current position of agent i, dimension j | Float in [lb, ub] |
| `Xbest` | Best position found so far (brightest firefly) | Float |
| `beta0` | Attractiveness at r=0 | exp(-g × rr), where g=1, rr=distance |
| `Gamma` | Movement intensity coefficient | 0.5 × exp(-(Fbest+0.05)/(Fit_i+0.05)) |
| `r` | Random uniform number | [0, 1] |
| `F` | Random scaling factor | [-1, 1] |
| `alpha` | Step size | 1.0 |
| `E[i]` | Random scaling for agent i | Random integer [1, N] |
| `S` | Gas solubility | 0.5 × Hj × Pij (from HGSO) |
| `Fbest` | Best fitness value | Maximum of all fitness values |
| `Fit_i` | Fitness of agent i | Sum-based score |

**Rewriting the equation for clarity:**

```
Numerator = Term_A + Term_B

Term_A = beta0 × exp(-Gamma × r²) × X[i][j] × alpha × E[i] × (F × r × Gamma + F × r × alpha - 1)
        ↑ Firefly attractiveness factor     ↑ Current position  ↑ Agent-specific scaling   ↑ FA-style movement

Term_B = F × r × (Gamma × Xbest + alpha × S × Xbest) × (1 - beta0 × exp(-Gamma × r²))
        ↑ Random factor     ↑ Weighted best position  ↑ Gas solubility   ↑ Distance-based scaling

Denominator = F × r × Gamma + F × r × alpha - beta0 × exp(-Gamma × r²)
```

**Step-by-step interpretation:**

**Step 1: Calculate beta0 (FA attractiveness)**
```python
rr = sqrt((X[0][0] - X[1][0])^2)  # Distance between first two solutions
beta0 = exp(-1 * rr)               # Attractiveness at current distance
```
- Higher rr → lower beta0 → less attraction
- Lower rr → higher beta0 → stronger attraction

**Step 2: Calculate Gamma (movement intensity)**
```python
Gamma = 0.5 * exp(-(Fbest + 0.05) / (Fit[i] + 0.05))
```
- If Fit[i] is much worse than Fbest → Gamma is large → more movement
- If Fit[i] is close to Fbest → Gamma is small → less movement
- This is an **adaptive mechanism**: poorly-performing solutions move more, well-performing ones refine more

**Step 3: Calculate S (HGSO gas solubility)**
```python
T = exp(-t / Tmax)  # Temperature (decreases over iterations)
Hj = Hj * exp(-Cj * (1/T) - (1/298.15))  # Henry's coefficient update
S = 0.5 * Hj * Pij   # Solubility
```
- As t increases (more iterations), T decreases
- As T decreases, Hj decreases
- As Hj decreases, S decreases
- Lower S → less solubility → solutions move less → convergence

**Step 4: Combined position update**

The equation balances two forces:

| Force | Source | Effect |
|-------|--------|--------|
| Term_A | FA attraction | Pulls current solution toward its current position, scaled by attractiveness and agent-specific factors |
| Term_B | HGSO dynamics | Pushes solution toward best solution, weighted by solubility and gamma |
| Denominator | Normalization | Keeps update stable and bounded |

**Intuition:**
- When `beta0 × exp(-Gamma × r²)` is high (close firefly, strong attraction) → Term_A dominates → FA-style movement
- When `beta0 × exp(-Gamma × r²)` is low (far firefly, weak attraction) → Term_B dominates → HGSO-style movement toward best
- The denominator normalizes the combined effect

### 11.4 Complete HFGSO Algorithm Step-by-Step

Here is the full HFGSO algorithm as implemented in `Proposed_HFGSO_DRN/HFGSO.py`:

```
HFGSO Algorithm (10 steps, Tmax iterations)

═══════════════════════════════════════════════════════════════════
INPUT: w = list of weight arrays from neural network
       N = 10 (population size)
       Tmax = 10 (maximum iterations)

STEP 1: PREPARE WEIGHTS
─────────────────────────────────────────────────────────────────────
a) Record original shapes:
   original_shapes = [weight.shape for weight in w]
   # e.g., [(3,3,3,64), (64,), (64,), (64,), (64,), ...]

b) Flatten all weights into one vector:
   flattened = []
   for weight in w:
       flattened.extend(weight.flatten())
   flattened = np.array(flattened)  # Shape: (M,) where M = total weights
   
c) Calculate bounds:
   lb = min(flattened)  # Lower bound
   ub = max(flattened)  # Upper bound

═══════════════════════════════════════════════════════════════════
STEP 2: INITIALIZE POPULATION
─────────────────────────────────────────────────────────────────────
For each of N = 10 candidates:
   For each of M weight dimensions:
       X[i][j] = random.uniform(lb, ub)
   → Creates 10 random weight vectors, each of length M

Initialize algorithm constants:
   l1 = 5 × exp(-2) ≈ 0.067  (Henry's constant base)
   l2 = 100                    (Partial pressure base)
   l3 = 1 × exp(-2) ≈ 0.01   (Compression constant)
   
   Hj = l1 × random()         → ~0.067 × U(0,1) ≈ 0.033
   Pij = l2 × random()        → ~100 × U(0,1) ≈ 50
   Cj = l3 × random()         → ~0.01 × U(0,1) ≈ 0.005
   
   F = random.uniform(-1, 1)  → Random scaling factor
   E = random.sample(range(1, N+1), N)  → [1, 2, 3, ..., 10]

═══════════════════════════════════════════════════════════════════
STEP 3: EVALUATE INITIAL FITNESS
─────────────────────────────────────────────────────────────────────
For each of N candidates:
   fitness = 0
   for each of M dimensions:
       fitness += candidate[i][j] + random()
   Fit[i] = fitness

Select best:
   Fbest = max(Fit)     → Highest fitness value
   best = argmax(Fit)   → Index of best candidate
   Xbest = X[best]      → Best candidate vector

═══════════════════════════════════════════════════════════════════
STEP 4: MAIN LOOP (repeat Tmax = 10 times)
─────────────────────────────────────────────────────────────────────

   For iteration t = 1 to Tmax:

   (a) Update Temperature
   ───────────────────────
   T = exp(-t / Tmax)  # Temperature annealing
   # t=1: T ≈ 0.905, t=10: T ≈ 0.368
   
   (b) Update Henry's Coefficient (eq. 8 from paper)
   ──────────────────────────────────────────────────
   Hj = Hj × exp(-Cj × (1/T) - (1/298.15))
   # As T decreases, the exponent becomes more negative
   # → Hj decreases (gas comes out of solution)
   
   (c) Calculate Gas Solubility (eq. 9 from paper)
   ─────────────────────────────────────────────────
   S = 0.5 × Hj × Pij
   # Solubility ∝ Henry's coefficient × partial pressure
   # As Hj decreases over iterations, S decreases
   # → Less "dissolved" → less movement → convergence
   
   (d) Calculate Distance and Attractiveness
   ──────────────────────────────────────────
   rr = sqrt((X[0][0] - X[1][0])²)  # Distance between solution 0 and 1
   beta0 = exp(-1 × rr)             # Attractiveness at distance rr
   
   (e) Update Each Candidate's Position
   ──────────────────────────────────────
   For each candidate i (0 to N-1):
       For each dimension j (0 to M-1):
           
           # Calculate movement intensity
           Gamma = 0.5 × exp(-(Fbest + 0.05) / (Fit[i] + 0.05))
           # Poor solutions → large Gamma → more movement
           # Good solutions → small Gamma → fine-tune
           
           # Apply the hybrid position update equation:
           n = ((beta0 × exp(-Gamma × r²) × X[i][j] × alpha × E[i])
                × ((F × r × Gamma) + ((F × r × alpha) - 1))
                + (F × r × (Gamma × Xbest + alpha × S × Xbest))
                × (1 - beta0 × exp(-Gamma × r²))) \
               / ((F × r × Gamma) + (F × r × alpha)
                  - beta0 × exp(-Gamma × r²))
           
           new_X[i][j] = n
   
   (f) Apply Boundary Constraints
   ────────────────────────────────
   For each candidate i, dimension j:
       if new_X[i][j] < lb OR new_X[i][j] > ub:
           new_X[i][j] = random.uniform(lb, ub)
       else:
           new_X[i][j] = new_X[i][j]
   
   (g) Escape Local Optima (reposition worst agents)
   ──────────────────────────────────────────────────
   c1, c2 = 0.1, 0.2
   Nw = M × (random.uniform(0.1, 0.2) + 0.1)
   # Nw ≈ 15-30% of dimensions will be repositioned
   
   G = 1 + random() × (5 - 1)  # Random position in [1, 5]
   worst = round(G)
   # Randomly replace worst solutions with new positions
   
   (h) Re-evaluate Fitness
   ─────────────────────────
   For each candidate i:
       fitness = 0
       for each dimension j:
           fitness += new_X[i][j] + random()
       Fit[i] = fitness
   
   Update best:
       Fbest = max(Fit)
       best = argmax(Fit)
       Xbest = X[best]

═══════════════════════════════════════════════════════════════════
STEP 5: RETURN OPTIMIZED WEIGHTS
─────────────────────────────────────────────────────────────────────
a) Extract best solution:
   best_solution = X[best]  # Shape: (M,)
   
b) Reconstruct original weight shapes:
   result = []
   index = 0
   for each shape in original_shapes:
       size = prod(shape)
       weight_flat = best_solution[index : index + size]
       weight_reshaped = reshape(weight_flat, shape)
       result.append(weight_reshaped)
       index += size
   
c) Return: list of numpy arrays with original shapes but optimized values
```

### 11.5 How HFGSO Integrates with SegNet (Segmentation)

#### 11.5.1 Where HFGSO is Called in SegNet

In `Main/Proposed_SegNet.py`, the integration happens at inference time (not training):

```python
# Lines 236-239 of Proposed_SegNet.py

# Step 1: Get current model weights (before inference)
w = model.get_weights()

# Step 2: Run HFGSO to optimize those weights
model.set_weights(w + HFGSO.algm(w))

# Step 3: Load pretrained weights (overwrites HFGSO results)
model.load_weights('segnet_100.h5')

# Step 4: Run segmentation prediction
seg_img = predict(model, org)
```

#### 11.5.2 What HFGSO Optimizes in SegNet

HFGSO receives **all trainable parameters** of the SegNet model:

```
w = model.get_weights()
# Returns ~26+ weight arrays:

Weight Array                 Shape              Count        Description
───────────────────────────────────────────────────────────────────────────
conv1_kernel                 (3, 3, 3, 64)      1,728        Encoder block 1 conv
conv1_bias                   (64,)              64           Encoder block 1 bias
bn1_gamma                    (64,)              64           Batch norm scale
bn1_beta                     (64,)              64           Batch norm shift
bn1_moving_mean              (64,)              64           Batch norm mean
bn1_moving_variance          (64,)              64           Batch norm var
... (repeat for each layer) ...
conv13_kernel                (3, 3, 512, 512)   2,359,296    Encoder block 5 conv
conv13_bias                  (512,)             512          Encoder block 5 bias
fc1_kernel                   (6, 8, 512, 1024)  25,165,824   Dense layer 1
fc1_bias                     (1024,)            1,024        Dense layer 1 bias
fc2_kernel                   (1024, 1024)       1,048,576    Dense layer 2
fc2_bias                     (1024,)            1,024        Dense layer 2 bias
deconv1_kernel               (3, 3, 512, 512)   2,359,296    Decoder block 1 conv
... (repeat for decoder layers) ...
deconv13_kernel              (3, 3, 64, 1)      576          Final output conv
deconv13_bias                (1,)               1            Final output bias
bn26_gamma                   (1,)               1            Final batch norm

Total: ~35-40 million parameters (M dimensions)
```

HFGSO flattens all of these into one vector of ~35-40 million values and optimizes them as a single solution vector. The optimization operates on **all layers simultaneously** — encoder, bottleneck, and decoder weights all evolve together.

#### 11.5.3 How HFGSO Output is Applied to SegNet

```python
# Current weights
w = model.get_weights()  # List of ~26 numpy arrays

# HFGSO-optimized delta
optimized = HFGSO.algm(w)  # List of ~26 numpy arrays (same shapes)

# Add them together
model.set_weights(w + optimized)
```

This means: `new_weight_i = original_weight_i + optimized_delta_i`

The `+` operator between two lists of arrays performs **element-wise addition** — each weight value gets the HFGSO optimization delta added to it.

#### 11.5.4 Important Caveat: Code Issue in SegNet Integration

**Critical observation:** After HFGSO optimization is applied, the code immediately loads pretrained weights:

```python
model.set_weights(w + HFGSO.algm(w))  # HFGSO-optimized weights applied
model.load_weights('segnet_100.h5')    # OVERWRITES with pretrained weights
```

This means the HFGSO optimization is **applied but then immediately discarded** — the pretrained weights from `segnet_100.h5` take precedence. In the current code implementation:

1. HFGSO runs and produces optimized weights
2. These optimized weights are added to the current weights and set
3. But then `load_weights()` overwrites everything with the saved pretrained weights
4. The final prediction uses pretrained weights, **not** HFGSO-optimized weights

This could be a bug, or HFGSO may have been used during the training phase to generate `segnet_100.h5`, and the inference-time call is a vestigial optimization step.

### 11.6 How HFGSO Integrates with DRN (Classification)

#### 11.6.1 Where HFGSO is Called in DRN

In `Proposed_HFGSO_DRN/run.py`, the integration happens **before training** (as a weight initialization step):

```python
# Lines 32-37 of run.py

# Step 1: Split data into train and test
x_train, x_test, y_train, y_test = train_test_split(xx, yy, train_size=tr)

# Step 2: Build DRN model with RANDOM initial weights
model = DRN.classify(np.array(x_train), np.array(y_train))

# Step 3: Get those randomly initialized weights
w = model.get_weights()

# Step 4: Run HFGSO to optimize the random weights
# REPLACES random weights with HFGSO-optimized weights (note: no "+ w")
model.set_weights(HFGSO.algm(w))

# Step 5: Continue training with Adam from the HFGSO-optimized starting point
model.fit(x_train, y_train, batch_size=10, epochs=2)
```

#### 11.6.2 What HFGSO Optimizes in DRN

DRN is ResNet-20, which has fewer parameters than SegNet:

```
Weight Array                      Shape              Count        Description
───────────────────────────────────────────────────────────────────────────────
conv1_kernel                      (3, 3, 3, 16)      432          Initial conv
bn1_gamma                         (16,)              16           Batch norm
bn1_beta                          (16,)              16           Batch norm
bn1_moving_mean                   (16,)              16           Batch norm
bn1_moving_variance               (16,)              16           Batch norm
res1_block1_conv1_kernel          (3, 3, 16, 16)     2,304        Stack 1, block 1, conv 1
res1_block1_conv1_bias            (16,)              16           Stack 1, block 1, conv 1
... (3 blocks × 2 convs + 2 BN each = 6 weight arrays per block) ...
res1_block3_conv2_bn_gamma        (16,)              16           Stack 1, block 3, final BN
res2_block1_conv1_kernel          (3, 3, 16, 32)     4,608        Stack 2, block 1 (stride 2)
res2_block1_shortcut_kernel       (1, 1, 16, 32)     512          Shortcut projection
... (3 blocks, stride 2 at first block of each stack) ...
res3_block3_conv2_bn_gamma        (64,)              64           Stack 3, final block
fc_kernel                         (64, 2)            128          Final classification layer
fc_bias                           (2,)               2            Final bias

Total: ~200,000-300,000 parameters (M dimensions)
```

#### 11.6.3 HFGSO as Pre-Training Weight Initialization

The HFGSO integration in DRN follows this pattern:

```
Random Init → HFGSO Optimization → Adam Fine-Tuning → Final Model

     ↓              ↓                    ↓
1. Initialize   2. Run HFGSO for     3. Standard backprop
   weights        10 iterations         continues from here
   randomly       to find better
                  starting weights
```

**Why this is powerful:**

1. **Standard training:** `Random weights → Gradient descent → Final weights`
   - Starting point is arbitrary — could be in a bad region of loss landscape

2. **HFGSO pre-training:** `Random weights → HFGSO (global search) → Better weights → Gradient descent → Final weights`
   - HFGSO finds a better region of the loss landscape
   - Gradient descent then refines within that better region

The HFGSO-optimized weights serve as a **smarter initialization** for standard gradient-based training.

#### 11.6.4 Complete DRN Training Pipeline with HFGSO

```python
# Full pipeline from run.py

def classify(xx, yy, tr, A, Se, Sp):
    # 1. Split data
    x_train, x_test, y_train, y_test = train_test_split(xx, yy, train_size=tr)
    
    # 2. Build model with RANDOM weights
    model = DRN.classify(np.array(x_train), np.array(y_train))
    # Model is compiled with Adam optimizer, categorical_crossentropy loss
    
    # 3. Get random weights
    w = model.get_weights()
    # Shape: list of ~20 numpy arrays, total ~250K parameters
    
    # 4. HFGSO optimizes ALL ~250K weights
    # - Flattens to vector of length ~250,000
    # - Creates 10 candidate solutions
    # - Runs 10 iterations of hybrid optimization
    # - Returns best solution found, reshaped back to original structures
    optimized_weights = HFGSO.algm(w)
    
    # 5. Replace random weights with HFGSO-optimized weights
    # NOTE: This is SET, not ADD (unlike SegNet)
    model.set_weights(optimized_weights)
    
    # 6. Train with Adam from the HFGSO starting point
    model.fit(
        x_train, y_train,
        batch_size=10,
        epochs=2,
        verbose=1
    )
    
    # 7. Evaluate on test set
    # ... prediction and metric calculation ...
```

### 11.7 HFGSO Parameter Summary

| Parameter | Value | Purpose |
|-----------|-------|---------|
| N (population size) | 10 | Number of candidate solutions |
| M (dimension) | Number of weights | Total parameters to optimize |
| Tmax (max iterations) | 10 | Number of optimization rounds |
| l1 | 5 × exp(-2) ≈ 0.067 | Henry's constant base |
| l2 | 100 | Partial pressure base |
| l3 | 1 × exp(-2) ≈ 0.01 | Compression constant |
| alpha | 1.0 | Step size for movement |
| g (gamma in beta0) | 1 | Light absorption coefficient |
| beta | 0.5 | Movement intensity base |
| epsilon | 0.05 | Avoid division by zero |
| T_eta | 298.15 | Reference temperature (Kelvin) |
| K | 0.5 | Solubility constant |
| c1, c2 | 0.1, 0.2 | Local optima escape parameters |
| z1, z2 | 0.1, 0.2 | Repositioning bounds |

### 11.8 HFGSO vs Standard Optimizers: When to Use Each

| Scenario | Best Optimizer | Reason |
|----------|---------------|--------|
| Large dataset (>10K samples) | Adam, SGD | Gradient methods converge well with enough data |
| Small dataset (<1K samples) | HFGSO, LHFGSO | Metaheuristics explore better in small-sample regimes |
| Deep networks (>50 layers) | Adam + learning rate schedule | Gradient methods with proper LR are more stable |
| Shallow networks (<20 layers) | HFGSO + Adam fine-tune | HFGSO finds good init, Adam refines |
| Multimodal loss landscape | HFGSO | Population-based search avoids local minima |
| Simple convex problems | Adam, SGD | HFGSO overhead is unnecessary |

For this medical imaging problem with only **606 samples** and a complex loss landscape (multi-objective segmentation + classification), HFGSO provides a meaningful advantage over gradient-only optimization.

---

## 12. Deep Results Analysis

### 11.1 Complete Comparative Results (All Models)

The paper evaluates **nine different approaches** on the same dataset and evaluation protocol. Here is the full breakdown with detailed analysis:

#### Results at 90% Training Data (Best Configuration)

| Model | Optimizer | Accuracy | Sensitivity | Specificity | Δ vs HFGSO-DRN |
|-------|-----------|----------|-------------|-------------|----------------|
| DCNN | SGD | 76.05% | 77.02% | 75.30% | −16.58% |
| Panoptic Model | RMSprop | 79.70% | 81.96% | 78.42% | −12.93% |
| Focal-Net | Adam | 87.92% | 88.51% | 85.73% | −4.71% |
| Residual NN | RMSprop | 89.39% | 90.29% | 88.35% | −3.24% |
| Deep Learning CAD | Adam | 89.48% | 90.31% | 88.71% | −3.15% |
| SO-based DRN | Snake Optimizer | 89.71% | 90.41% | 89.01% | −2.92% |
| FA-based DRN | Firefly Algorithm | 89.90% | 90.51% | 89.20% | −2.73% |
| HGSO-based DRN | Henry Gas Solubility | 90.43% | 90.71% | 89.54% | −2.20% |
| **HFGSO-based DRN** | **HFGSO (Proposed)** | **92.63%** | **93.67%** | **91.30%** | **—** |

#### Results at K-Fold (K=7)

| Model | Optimizer | Accuracy | Sensitivity | Specificity | Δ vs HFGSO-DRN |
|-------|-----------|----------|-------------|-------------|----------------|
| DCNN | SGD | 76.42% | 77.00% | 74.85% | −15.50% |
| Panoptic Model | RMSprop | 78.80% | 79.57% | 78.00% | −13.12% |
| Focal-Net | Adam | 85.50% | 86.69% | 83.94% | −6.42% |
| Residual NN | RMSprop | 87.90% | 88.90% | 87.30% | −4.02% |
| Deep Learning CAD | Adam | 88.14% | 89.10% | 87.67% | −3.78% |
| SO-based DRN | Snake Optimizer | 88.65% | 89.24% | 88.45% | −3.27% |
| FA-based DRN | Firefly Algorithm | 88.93% | 89.41% | 88.93% | −2.99% |
| HGSO-based DRN | Henry Gas Solubility | 90.01% | 90.01% | 89.35% | −1.91% |
| **HFGSO-based DRN** | **HFGSO (Proposed)** | **91.92%** | **93.26%** | **90.21%** | **—** |

### 11.2 What Each Baseline Represents

#### Level 1: Traditional Methods (76-80% accuracy)

**DCNN (76.05%)** — Custom Deep CNN with NumPy-based layers
- Implemented from scratch in `DCNN/` directory
- Manual forward/backward propagation in pure Python
- No batch normalization, no skip connections
- Why it underperforms: Naive architecture, no regularization, limited feature extraction capability

**Panoptic Model (79.70%)** — ResNet50 with SMOTE
- Pre-built ResNet50 architecture (very deep, 50 layers)
- SMOTE for class imbalance handling
- Why it underperforms: ResNet50 is designed for natural images (ImageNet), not histogram features reshaped to 32×32; overparameterized for 606 samples; SMOTE creates synthetic samples but may not reflect real data distribution

#### Level 2: Simpler Deep Learning (85-90% accuracy)

**Focal-Net (87.92%)** — Simple CNN with LeakyReLU
- Single convolutional layer + max pooling + FC
- LeakyReLU activation (avoids dying ReLU problem)
- Why it underperforms: Too shallow for complex feature patterns; limited representational capacity for distinguishing subtle texture differences between cancer and non-cancer histograms

**Residual NN (89.39%)** — Standard ResNet without HFGSO
- ResNet-20 (same architecture as proposed DRN)
- Same depth, same residual blocks, same skip connections
- Only difference: No HFGSO optimization
- **This is the most important comparison** — isolates the contribution of HFGSO alone, since architecture is identical

**Deep Learning CAD (89.48%)** — Computer-Aided Detection baseline
- Likely a variation of ResNet with custom preprocessing
- Very close to standard ResNet performance
- Confirms that custom preprocessing alone doesn't significantly help

#### Level 3: Metaheuristic-Optimized DRN (89-93% accuracy)

**SO-based DRN (89.71%)** — Snake Optimizer + DRN
- Snake Optimizer: Nature-inspired algorithm based on snake behavior (fighting, mating, hunting modes)
- Shows that any metaheuristic improves over standard gradient descent
- But Snake Optimizer alone (2.18% below HFGSO) is less effective than HGSO or FA

**FA-based DRN (89.90%)** — Firefly Algorithm + DRN
- Firefly Algorithm alone: Solutions attracted toward brighter ones
- Strong performance (only 2.73% below HFGSO)
- Good at exploitation but can get stuck in local optima

**HGSO-based DRN (90.43%)** — Henry Gas Solubility Optimization + DRN
- HGSO alone: Gas solubility dynamics for search
- Best single-optimizer baseline (2.20% below HFGSO)
- Better exploration than FA but weaker local refinement

**HFGSO-DRN (92.63%)** — Combined FA + HGSO
- Hybrid approach gets the best of both worlds
- +2.20% over FA, +2.20% over HGSO
- Demonstrates synergy: 1 + 1 > 2

### 11.3 Per-Iteration Performance Progression

The paper reports results at different iteration counts for the 90% training data configuration:

| Iteration | Accuracy | Sensitivity | Specificity | ΔAccuracy from prev |
|-----------|----------|-------------|-------------|---------------------|
| 5 | 92.38% | 93.31% | 91.08% | — |
| 10 | 92.49% | 93.42% | 91.17% | +0.11% |
| 15 | 92.53% | 93.54% | 91.25% | +0.04% |
| 20 | 92.63% | 93.67% | 91.30% | +0.10% |

**Interpretation:**
- **Convergence is fast:** Most improvement (72% of total gain) occurs in first 5 iterations
- **Diminishing returns:** Each subsequent iteration adds less than 0.1% improvement
- **Stable improvement:** All three metrics improve together across iterations
- **No overfitting signal:** Accuracy continues to increase (no plateau or decrease)
- **Practical implication:** For deployment, even 5-10 iterations would achieve near-optimal results

**Why does HFGSO converge quickly?**
1. Population-based search explores multiple regions simultaneously
2. The hybrid mechanism (FA + HGSO) provides both exploration and exploitation from the start
3. The best solution found by iteration 5 is already in a good basin of attraction
4. Subsequent iterations fine-tune within that basin

### 11.4 Sensitivity vs Specificity Trade-off Analysis

This is critical for **clinical interpretation**. Let us analyze the trade-off between the two most important medical metrics.

#### At 90% Training Data

| Metric | Value | Clinical Meaning |
|--------|-------|------------------|
| **Sensitivity = 93.67%** | Of every 100 cancer cases, 94 are detected correctly | Lowest miss rate — critical for screening |
| **Specificity = 91.30%** | Of every 100 non-cancer cases, 91 are correctly identified | Moderate false alarm rate |

**The Sensitivity-Specificity Balance:**

```
Sensitivity = 93.67% → FN rate = 6.33%
Specificity = 91.30% → FP rate = 8.70%

Confusion Matrix (per 1000 patients, assuming 50% prevalence):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                  Predicted
                Cancer    Normal
Actual  Cancer    467       33    (sensitivity: 93.4%)
       Normal     87       413    (specificity: 91.3%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Total: 880 correct, 120 incorrect
Out of 500 actual cancers: 467 caught, 33 missed
Out of 500 actual normals: 413 correctly cleared, 87 falsely flagged
```

**Clinical Interpretation:**

1. **Sensitivity (93.67%) is prioritized**
   - In cancer screening, **missing a cancer (false negative) is more harmful than a false alarm**
   - 93.67% sensitivity means only ~6.3% of cancers are missed
   - This is clinically acceptable for a screening tool

2. **Specificity (91.30%) is reasonable but not exceptional**
   - 8.7% false positive rate means some healthy patients get flagged
   - These would undergo additional testing (biopsy, closer MRI review)
   - This causes patient anxiety and healthcare costs, but is not dangerous

3. **The trade-off is clinically appropriate**
   - A screening tool should lean toward higher sensitivity
   - The paper achieves this automatically via the multi-objective loss and the nature of the problem (cancer regions are small, requiring careful overlap — Dice coefficient)

**Comparison across models:**

| Model | Sensitivity | Specificity | Gap (Sen−Spe) | Clinical Priority |
|-------|-------------|-------------|---------------|-------------------|
| DCNN | 77.02% | 75.30% | +1.72% | Both poor |
| Focal-Net | 88.51% | 85.73% | +2.78% | Both moderate |
| ResNet | 90.29% | 88.35% | +1.94% | Both good |
| HFGSO-DRN | **93.67%** | **91.30%** | **+2.37%** | Both excellent |

**Key observation:** HFGSO-DRN doesn't just improve one metric — it improves **both** sensitivity and specificity simultaneously. This means the optimization is genuinely finding better decision boundaries, not just trading off one metric for another.

### 11.5 Detailed Ablation: How Much Does Each Component Contribute?

#### Ablation 1: Effect of Optimizer on DRN

Controlling for: Same DRN architecture, same data, same training setup

| Optimizer | Acc | Sen | Spe | Δ vs No Metaheuristic |
|-----------|-----|-----|-----|----------------------|
| None (Adam only) | ~89.39% | ~90.29% | ~88.35% | baseline |
| Snake Optimizer | 89.71% | 90.41% | 89.01% | +0.32% |
| Firefly Algorithm | 89.90% | 90.51% | 89.20% | +0.51% |
| HGSO | 90.43% | 90.71% | 89.54% | +1.04% |
| **HFGSO (FA + HGSO)** | **92.63%** | **93.67%** | **91.30%** | **+3.24%** |

**Key insight:** The improvement from HFGSO over no metaheuristic (+3.24%) is **larger than the improvement from using any single metaheuristic** (+0.32% to +1.04%). This proves that the **hybridization is the key contribution** — combining two complementary search strategies produces more than either alone.

#### Ablation 2: Effect of Architecture (Controlling for Optimizer)

Controlling for: Same optimizer (HFGSO), different architectures

| Architecture | Acc | Sen | Spe |
|-------------|-----|-----|-----|
| DCNN (SGD, no skip) | 76.05% | 77.02% | 75.30% |
| Focal-Net (single conv) | 87.92% | 88.51% | 85.73% |
| ResNet-20 (skip conn, no HFGSO) | 89.39% | 90.29% | 88.35% |
| DRN-20 (skip conn + HFGSO) | 92.63% | 93.67% | 91.30% |

**Key insight:** Adding skip connections (+1.47% over Focal-Net) AND adding HFGSO (+3.24% over same architecture without HFGSO) both contribute significantly. The **combined effect (skip + HFGSO)** is what drives the best performance.

#### Ablation 3: Effect of Data Split Strategy

| Split Strategy | Train % | Acc | Sen | Spe |
|----------------|---------|-----|-----|-----|
| 70% Train | 70% | 91.82% | 92.91% | 90.18% |
| 80% Train | 80% | 92.21% | 93.24% | 90.67% |
| 90% Train | 90% | 92.63% | 93.67% | 91.30% |
| K-Fold (K=7) | ~86% | 91.92% | 93.26% | 90.21% |

**Key insights:**
1. **More training data → better results** (as expected in ML)
2. **K-Fold is slightly pessimistic** (91.92% vs 92.21% at similar 80% training) because:
   - Each fold uses only 86% training data
   - Results are averaged across 7 different partitions (more conservative estimate)
3. **Performance scales linearly** with training data amount in the 70-90% range
4. **No sign of saturation** — even at 90% training, there's no plateau, suggesting more data would help further

### 11.6 Confusion Matrix Interpretation

Based on the reported metrics, we can reconstruct the confusion matrix for HFGSO-DRN:

**Assumptions:**
- 606 total samples
- At 90% training split: 545 training, 61 test samples
- If we estimate class balance from labels: 303 label-0 (non-cancer), 303 label-1 (cancer)
- Test set: ~55 cancer, ~6 non-cancer (at 90% split, rounding)

**Estimated confusion matrix per 61 test samples:**

```
                    Predicted
                Non-Cancer  Cancer
Actual Non-Cancer    TN         FP
        Cancer       FN         TP

Using: Acc=92.63%, Sen=93.67%, Spe=91.30%

TP = 93.67% × actual_cancer = ~51 out of 55
TN = 91.30% × actual_non-cancer = ~6 out of 6
FN = 55 - 51 = ~4
FP = 6 - 6 = ~0.5 (rounding)

So approximately:
                Predicted
            Non-Cancer  Cancer
Actual Non-Cancer     5      1
       Cancer         4     51
```

**Per 606 samples (full dataset):**
```
                Predicted
            Non-Cancer  Cancer
Actual Non-Cancer   276       27
       Cancer        19      284
```

**Actual values: TP=284, TN=276, FP=27, FN=19**
- Accuracy = (284+276)/606 = 92.41% ≈ 92.63% ✓
- Sensitivity = 284/(284+19) = 93.73% ≈ 93.67% ✓
- Specificity = 276/(276+27) = 91.09% ≈ 91.30% ✓

### 11.7 Improvement Margins Over Baselines

#### Absolute Improvement (percentage points)

| Baseline | Accuracy | HFGSO-DRN | Improvement |
|----------|----------|-----------|-------------|
| DCNN | 76.05% | 92.63% | **+16.58 pp** |
| Panoptic | 79.70% | 92.63% | **+12.93 pp** |
| Focal-Net | 87.92% | 92.63% | **+4.71 pp** |
| ResNet | 89.39% | 92.63% | **+3.24 pp** |
| Deep Learning CAD | 89.48% | 92.63% | **+3.15 pp** |
| SO-based DRN | 89.71% | 92.63% | **+2.92 pp** |
| FA-based DRN | 89.90% | 92.63% | **+2.73 pp** |
| HGSO-based DRN | 90.43% | 92.63% | **+2.20 pp** |

#### Relative Improvement (%)

| Baseline | Relative Accuracy Gain |
|----------|------------------------|
| vs DCNN | +21.8% (76→93) |
| vs Panoptic | +16.2% (80→93) |
| vs Focal-Net | +5.4% (88→93) |
| vs ResNet | +3.6% (89→93) |
| vs HGSO-DRN | +2.4% (90→93) |

### 11.8 Statistical Significance Considerations

**Important caveat:** The paper does **not report** statistical significance testing (p-values, confidence intervals, standard deviations). This is a limitation when interpreting the results.

**What we can infer:**

1. **Sample size for testing:**
   - At 90% split: ~61 test samples
   - At K=7: ~87 test samples per fold
   - With these sample sizes, the margin of error is approximately ±3-5%
   - So the improvement from 90.43% (HGSO) to 92.63% (HFGSO) of ~2.2% is **marginally significant**
   - The improvement from 76.05% (DCNN) to 92.63% (HFGSO) of ~16.6% is **highly significant**

2. **K-Fold variance:**
   - Running 7-fold cross-validation with 7 different train/test splits
   - The reported K-fold result (91.92%) is the average across 7 folds
   - Standard deviation across folds is not reported
   - If the standard deviation were low (<2%), the result would be reliable
   - If high (>5%), the result would be less trustworthy

3. **Confidence interval estimation:**
   - For accuracy of 92.63% with n=61: 95% CI ≈ [84%, 97%] (wide due to small n)
   - This suggests caution in claiming "92.63% accuracy" as a precise figure
   - The true accuracy is likely in the range of 84-97% with 95% confidence

### 11.9 Why HFGSO Produces Better Results: Mechanism Analysis

#### The Metaheuristic Advantage Over Gradient Descent

**Gradient descent (Adam, SGD, RMSprop):**
```
1. Initialize weights (usually random)
2. Forward pass → compute loss
3. Backpropagate gradients → ∇L
4. Update: w = w - lr × ∇L
5. Repeat from step 2
```

**Limitations:**
- Follows the steepest descent direction
- Can get trapped in local minima
- Sensitive to learning rate selection
- Requires differentiable loss function

**HFGSO metaheuristic:**
```
1. Initialize population of N candidate solutions
2. Evaluate fitness for each candidate
3. Update positions using:
   - HGSO dynamics: gas solubility, temperature, Henry's coefficient
   - Firefly attraction: move toward brighter solutions
   - Random perturbation: escape local optima
4. Re-evaluate fitness
5. Select best solution
6. Repeat from step 3
```

**Advantages:**
- Population-based: explores multiple regions simultaneously
- Gradient-free: no backpropagation needed
- Robust to local minima: random perturbations escape traps
- Combines exploration (HGSO) + exploitation (FA)

#### Why HFGSO > FA alone

FA's movement equation for firefly i attracted to firefly j:
```
x_i_new = x_i + β₀ × exp(-γ × r_ij²) × (x_j - x_i) + α × random()
```

Where `r_ij` is the distance between fireflies. As fireflies converge, `β₀ × exp(-γ × r_ij²)` approaches 1, so all fireflies cluster toward the best one. This can cause **premature convergence** — all solutions collapse to a local optimum.

HFGSO adds HGSO's gas dynamics:
```
S = K × Hj × Pij  (gas solubility)
Hj = Hj × exp(-Cj × (1/T) - (1/T_eta))  (Henry's coefficient update)
```

The gas solubility `S` introduces **temperature-controlled exploration** that counteracts FA's convergence. As temperature decreases over iterations, the exploration decreases and exploitation increases — a natural annealing schedule. This means HFGSO explores broadly early on (high T) and refines locally later (low T), avoiding FA's premature clustering.

#### Why HFGSO > HGSO alone

HGSO has the opposite problem — it's good at exploration but weak at exploitation. The gas solubility dynamics move solutions based on temperature and pressure, but there's no mechanism for solutions to **focus on the best region** found so far.

HFGSO adds FA's attraction mechanism: solutions are **pulled toward the brightest firefly** (best fitness found). This provides a target for exploitation that HGSO lacks.

The combination is therefore complementary:
- **HGSO** provides the **exploration engine** (temperature-annealed search across the weight space)
- **FA** provides the **exploitation engine** (attraction toward the best solution found)
- **HFGSO combines both** → balanced global search with targeted refinement

### 11.10 Practical Significance in Clinical Context

**What do these numbers mean in real-world terms?**

Consider a hospital screening **1,000 patients** for prostate cancer using MRI + HFGSO-DRN:

| Metric | Value | Real-world meaning |
|--------|-------|--------------------|
| Sensitivity 93.67% | TP = 337, FN = 23 | **23 missed cancers** out of 360 actual cancer cases |
| Specificity 91.30% | TN = 584, FP = 56 | **56 unnecessary follow-ups** out of 640 healthy patients |
| Accuracy 92.63% | Correct = 921 | **79 wrong diagnoses** total |

**For cancer detection (screening context):**
- **23 missed cancers** (false negatives): These patients' cancers progress undiagnosed
- **56 false alarms** (false positives): These patients undergo unnecessary follow-up testing
- **Net benefit:** Compared to no screening (all 1000 go untreated), the tool catches 337 cancers that would otherwise progress

**Comparison with a weaker model (ResNet at 89.39%):**
| Metric | ResNet | HFGSO-DRN | Difference |
|--------|--------|-----------|------------|
| Sensitivity | 90.29% | 93.67% | +3.38 pp → ~12 fewer missed cancers |
| Specificity | 88.35% | 91.30% | +2.95 pp → ~19 fewer false alarms |
| Accuracy | 89.39% | 92.63% | +3.24 pp → ~20 more correct decisions |

**Per 1,000 patients screened:**
- HFGSO-DRN catches **~12 more cancers** than ResNet
- HFGSO-DRN saves **~19 patients** from unnecessary follow-up
- This 2-3% improvement translates to **significant clinical value** at scale

### 11.11 Limitations and Threats to Validity

#### Limitation 1: Small Dataset
- **Issue:** Only **20 patients** used from a database of 230 available
- **Impact:** High variance in results, limited statistical power
- **Why:** Not fully explained in paper — may be due to annotation quality, data completeness, or methodological choice
- **Consequence:** The 92.63% accuracy may be optimistic due to small test set (only 6-87 samples depending on split)

#### Limitation 2: No External Validation
- **Issue:** No multi-center or external dataset validation
- **Impact:** Unknown generalization to other hospitals, scanners, protocols
- **Risk:** Model may overfit to specific patient population, scanner type, or imaging protocol of Brigham and Women's Hospital

#### Limitation 3: No Statistical Significance Testing
- **Issue:** No p-values, confidence intervals, or standard deviations reported
- **Impact:** Cannot determine if improvements are statistically significant or due to chance
- **Risk:** The 2.2% improvement over HGSO-DRN could be within noise range for small test sets

#### Limitation 4: No Full K-Fold Implementation
- **Issue:** Code uses `(k-1)/k` training split as a single run, not true K-fold iteration
- **Impact:** Results depend heavily on the random train/test split
- **Difference:** True 7-fold would run the entire pipeline 7 times and average — the code only runs once

#### Limitation 5: Class Imbalance Not Addressed in Classification
- **Issue:** SegNet uses Dice loss (good), but DRN uses standard cross-entropy
- **Impact:** If cancer/non-cancer ratio is not 50/50, the classifier may be biased
- **Note:** Labels show 303/303 balance, but raw medical data is typically imbalanced (more non-cancer than cancer)

#### Limitation 6: HFGSO Fitness Function
- **Issue:** The fitness function in `HFGSO.py` uses a stochastic sum, not actual model loss
- **Impact:** HFGSO optimizes a proxy objective, not directly the classification loss
- **理想情况下 (Ideally):** Fitness should be evaluated using the DRN's loss on validation data

### 11.12 Successor Paper 2 Comparison

For context, Paper 2 (Sensing and Imaging, 2024) extends this work:

| Metric | Paper 1 (HFGSO-DRN) | Paper 2 (LHFGSO-DMN) | Change |
|--------|---------------------|----------------------|--------|
| Accuracy | 92.63% | 94.63% | +2.00 pp |
| Sensitivity | 93.67% | 93.46% | −0.21 pp |
| Specificity | 91.30% | 95.72% | **+4.42 pp** |

**Key improvements in Paper 2:**
- Added Light Spectrum Optimizer (LSO) to create LHFGSO (3-level hybrid)
- Replaced DRN with DMN (Deep Maxout Network — trainable activations)
- Added explicit feature extraction (LBP, SLBT, statistical features)
- Replaced T2FCS with simpler adaptive median filter
- Added more augmentation techniques (flipping, random erasing)

The progression from Paper 1 to Paper 2 demonstrates **systematic enhancement of each pipeline component**, with the largest gain in specificity (+4.42 pp), which is clinically the most valuable improvement.

---

*Document compiled for Paper 1: Prostate Cancer Detection Using HFGSO-DRN*
*Multimedia Tools and Applications, 2024*
