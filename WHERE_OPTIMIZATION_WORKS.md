# Paper 1: Exactly Where HFGSO Works in the Pipeline

## Paper: Prostate cancer detection using HFGSO-based DRN
## Journal: Multimedia Tools and Applications (2024)

---

## Overview

HFGSO is used at **two distinct points** in the pipeline:

1. **Training the multi-objective SegNet** (segmentation stage)
2. **Training the DRN** (cancer detection stage)

These are two separate applications of the same optimizer on two different networks with two different fitness functions.

---

## 1. HFGSO in SegNet Training (Segmentation)

### What is SegNet?

SegNet is an encoder-decoder network for pixel-level segmentation:

```
SegNet Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  ENCODER (13 Conv layers)              DECODER (13 layers)      │
│  ┌────────┐  ┌────────┐  ┌────┐  ┌────────┐  ┌────────┐       │
│  │ Conv+BN│→ │ Conv+BN│→ │... │→ │Upsample│→ │Upsample│→ ...  │
│  │ +Pool  │  │ +Pool  │  │    │  │ +Conv  │  │ +Conv  │       │
│  └────────┘  └────────┘  └────┘  └────────┘  └────────┘       │
│       │                              ↑                          │
│       └──── pooling indices ─────────┘                          │
│                                                                 │
│  Input: Pre-processed MRI (F_t)    Output: Segmentation map     │
│                                    (pixel-wise cancer mask)     │
│                                                                 │
│  Final layer: Pixel-wise classification (softmax)               │
│               ↑                                                 │
│         THIS is where HFGSO targets                             │
└─────────────────────────────────────────────────────────────────┘
```

### What does HFGSO optimize in SegNet?

**From paper Section 3.4.3:** *"Through the optimization method, named HFGSO, the training of formulated multi-objective SegNet is done."*

The encoder's 13 convolutional layers learn features through standard forward-backward passes (convolution, batch normalization, max-pooling). The decoder upsamples using stored pooling indices.

HFGSO targets the **trainable parameters at the decoder's final classification layer** — the layer that assigns each pixel a class label (cancer / non-cancer). This is the layer where the segmentation decision is made.

### What is the fitness function?

**The fused Dice + Cross-Entropy loss (Equation 4 in paper):**

```
M(k, l) = (1 − μ) × [Σ_s k_s log(l_s)] − μ × log[(2Σ(l_s × k_s) + I) / (Σl_s + Σk_s + I)]
                      \_________________/         \________________________________________/
                       Cross-entropy part                    Dice coefficient part

Where:  μ = 0.75  (75% weight to Dice)
        I = 1e-15 (smoothing to avoid numerical instability)
        k_s = predicted label for category s
        l_s = true label for category s
```

### How does the optimization work step by step?

```
Step 1: Initialize a POPULATION of candidate solutions
        Each candidate = a set of SegNet classification layer weights

        Candidate 1: [w1=0.23, w2=-0.14, ..., b1=0.01, b2=-0.03, ...]
        Candidate 2: [w1=0.55, w2=0.32, ..., b1=0.05, b2=0.11, ...]
        Candidate 3: [w1=-0.08, w2=0.71, ..., b1=-0.02, b2=0.07, ...]
        ...
        Candidate N: [w1=0.41, w2=-0.29, ..., b1=0.03, b2=-0.01, ...]

Step 2: For EACH candidate:
        → Plug its weights into SegNet's classification layer
        → Run full SegNet forward pass on training images
        → Compute fused loss M(k,l) from Eq. 4
        → This loss value = fitness of that candidate

        Candidate 1 → fitness = 0.342
        Candidate 2 → fitness = 0.287  ← better (lower)
        Candidate 3 → fitness = 0.401

Step 3: Apply HFGSO update rules:
        → Firefly rule: Candidates with worse fitness move toward
          candidates with better fitness (attraction)
        → HGSO rule: Henry's coefficient controls exploration/exploitation
          balance; solubility dynamics guide search direction
        → Escape local optimum: Worst candidates are repositioned randomly

Step 4: Repeat Steps 2-3 for max iterations

Step 5: Best candidate's weights → final SegNet classification layer

RESULT: Segmented output C_t (pixel-wise cancer mask)
```

### Why use HFGSO here instead of Adam/SGD?

The segmentation classification layer maps abstract features to pixel labels. With combined Dice+CE loss on small, imbalanced cancer regions, the loss surface has many local minima. HFGSO's population-based search explores more of this surface than gradient descent, potentially finding a better segmentation decision boundary.

---

## 2. HFGSO in DRN Training (Cancer Detection)

### What is DRN?

DRN (Deep Residual Network) is a classification network with these layers:

```
DRN Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  ┌──────────────┐     Trained by standard backpropagation       │
│  │ Conv layer    │     (gradients + learning rate)               │
│  │ J2c(r) = Σ   │                                              │
│  │ B_{m,w}·r    │                                              │
│  └──────┬───────┘                                               │
│         ↓                                                       │
│  ┌──────────────┐     No trainable params                       │
│  │ Pooling layer │     (spatial downsampling only)               │
│  │ m_out=(m_in   │                                              │
│  │  -y_m)/z + 1 │                                              │
│  └──────┬───────┘                                               │
│         ↓                                                       │
│  ┌──────────────┐     No trainable params                       │
│  │ ReLU          │     (fixed function: max(0, r))              │
│  └──────┬───────┘                                               │
│         ↓                                                       │
│  ┌──────────────┐     Trained by standard backpropagation       │
│  │ Batch Norm    │                                              │
│  └──────┬───────┘                                               │
│         ↓                                                       │
│  ┌──────────────┐     Trained by standard backpropagation       │
│  │ Residual      │     e = K(r) + r  (skip connections          │
│  │ Blocks        │      help gradient flow)                     │
│  └──────┬───────┘                                               │
│         ↓                                                       │
│  ╔══════════════╗                                               │
│  ║ FC Layer     ║  ← HFGSO OPTIMIZES THESE                     │
│  ║              ║     R_h ∈ {P_{Y×Z}, U}                       │
│  ║ Weights: P   ║     P = weight matrix (Y × Z)                │
│  ║ Biases:  U   ║     U = bias vector  (Y × O)                 │
│  ╠══════════════╣                                               │
│  ║ Softmax      ║     No trainable params (normalization)       │
│  ╚══════╤═══════╝                                               │
│         ↓                                                       │
│     Output G_t: Cancer / No Cancer                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### What does HFGSO optimize in DRN?

**From paper Section 3.6.2:** *"Here, R_h ∈ {P_{Y×Z}, U}."*

This is explicit: the candidate solution R_h (each "gas molecule" in HFGSO) encodes:
- **P_{Y×Z}** — the weight matrix of the Fully Connected (FC) layer
- **U** — the bias vector of the FC layer

**HFGSO does NOT optimize the convolutional layers, pooling, batch norm, or residual blocks.** Those are trained by standard backpropagation with the learning rate (0.001).

### What is the fitness function?

**MSE — Mean Squared Error (Equation 31 in paper):**

```
ω_f = (1/ε) × Σ_{t=1}^{ε} (G*_t − G_t)²

Where:  G*_t = target output (ground truth label: cancer or not)
        G_t  = DRN's predicted output
        ε    = number of samples
```

The candidate with the **minimum MSE** is considered the best solution.

### How does the optimization work step by step?

```
Step 1: Initialize a POPULATION of candidate FC layer configurations
        Each candidate = one complete set of {P weights, U biases}

        Candidate 1: {P=[0.23, -0.14, 0.87, ...], U=[0.01, -0.03, ...]}
        Candidate 2: {P=[0.55, 0.32, -0.12, ...], U=[0.05, 0.11, ...]}
        ...

Step 2: For EACH candidate:
        → Load its P and U into the FC layer of DRN
        → Run full DRN forward pass (Conv→Pool→ReLU→BN→Residual→FC→Softmax)
          on augmented training images V_t
        → Compute MSE between DRN output G_t and ground truth G*_t
        → This MSE = fitness of that candidate

        Candidate 1 → MSE = 0.0341
        Candidate 2 → MSE = 0.0287  ← better (lower MSE)

Step 3: Apply HFGSO update rules on FC weight/bias values:
        → Firefly attraction: Candidate 1's weights shift toward Candidate 2's
        → HGSO dynamics: Henry's coefficient decays over iterations,
          shifting from exploration (big weight changes) to exploitation
          (small refinements)
        → Escape mechanism: If candidates stagnate, worst are repositioned

Step 4: Repeat Steps 2-3 for max iterations (paper uses up to 20)

Step 5: Best candidate's {P, U} → final FC layer weights and biases

RESULT: Final output G_t (cancer / no-cancer classification)
```

### Why optimize only the FC layer?

| Reason | Explanation |
|--------|-------------|
| **FC layer is the decision boundary** | It directly maps extracted features to the cancer/no-cancer decision. A suboptimal FC layer directly causes misclassification. |
| **Conv layers work well with gradients** | Convolutional features (edges, textures, shapes) are efficiently learned by backpropagation. Gradient signals flow well through conv layers. |
| **Computational feasibility** | Conv layers have far more parameters. Optimizing them all with population search (evaluating N candidates × full forward pass) would be too slow on an Intel i3 with 8GB RAM. |
| **Local minima matter most at the classifier** | The feature-to-decision mapping has a complex, multi-modal loss surface. Population-based search is most valuable here. |

---

## Summary: Where HFGSO Works in Paper 1

```
Full Pipeline with HFGSO locations marked:

MRI Input
    ↓
ROI Extraction          ← No optimization (geometric operation)
    ↓
T2FCS Filtering         ← No HFGSO (uses Cuckoo Search internally)
    ↓
┌───────────────────────────────────────┐
│ SegNet Segmentation                   │
│                                       │
│  Encoder (13 Conv layers) ← Backprop  │
│  Decoder (13 layers)      ← Backprop  │
│  Classification layer     ← ★ HFGSO ★ │
│                                       │
│  Fitness = Dice+CE loss (Eq. 4)       │
└───────────────────────────────────────┘
    ↓
Data Augmentation       ← No optimization (rotation + cropping)
    ↓
┌───────────────────────────────────────┐
│ DRN Cancer Detection                  │
│                                       │
│  Conv layers          ← Backprop      │
│  Pooling              ← No params     │
│  ReLU                 ← No params     │
│  Batch Norm           ← Backprop      │
│  Residual Blocks      ← Backprop      │
│  FC layer {P, U}      ← ★ HFGSO ★    │
│  Softmax              ← No params     │
│                                       │
│  Fitness = MSE (Eq. 31)               │
└───────────────────────────────────────┘
    ↓
Output: Cancer / No Cancer
```

### Key takeaway

HFGSO acts as a **targeted optimizer for the decision-making layers** — the layers where the network makes its final classification choice (pixel labels in SegNet, cancer/no-cancer in DRN). The feature-extraction layers (convolutions, residual blocks) continue to use standard gradient-based training. This is a practical design: metaheuristic search is most valuable at the decision boundary, and computationally feasible only on smaller parameter sets.
