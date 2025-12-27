# Entangled Dual Watermarks (EDW)

A neural network watermarking framework that embeds two watermarks in orthogonal subspaces with asymmetric entanglement coupling.

---

## Overview

EDW is a deep learning model watermarking technique designed for ownership verification. The framework embeds two independent watermarks (A and B) into a neural network such that attempting to remove one watermark preserves the other through an entanglement mechanism.

---

## How It Works

### 1. Orthogonal Key Generation

Two secret keys are generated in high-dimensional space:
- Key A: `k_A ∈ ℝ^d`
- Key B: `k_B ∈ ℝ^d`

These keys are made orthogonal using Gram-Schmidt orthogonalization: `<k_A, k_B> = 0`

### 2. Dual Projection Heads

Two separate neural network branches project feature representations into watermark subspaces:
- Projection Head A: Maps features to subspace aligned with key A
- Projection Head B: Maps features to subspace aligned with key B

Each head is a small multi-layer perceptron (MLP) attached to the model's feature extractor.

### 3. Asymmetric Entanglement Mechanism

The core innovation: Projection B receives signals from Projection A through an entanglement layer, but gradients do not flow backward from B to A.

**Forward pass:**
```
z_A = ProjectionA(features)
z_B_raw = ProjectionB(features)
entangled_signal = EntanglementLayer(detach(z_A))
z_B = z_B_raw + α * entangled_signal
```

The `detach()` operation stops gradient flow, creating asymmetric coupling.

### 4. Training Process

**Phase 1 - Task Training:**
- Train the backbone network and classifier on the primary task
- Standard supervised learning with cross-entropy loss

**Phase 2 - Watermark Embedding:**
- Freeze backbone and classifier weights
- Train only projection heads and entanglement layer
- Optimize two objectives:
  - **Alignment**: Watermarked samples should score high (close to +1) on both watermarks
  - **Separation**: Clean samples should score low (close to 0) on both watermarks

### 5. Verification

To verify ownership of a suspect model:
1. Compute watermark scores on watermarked samples: `s_A = <normalize(z_A), k_A>`
2. Compute watermark scores on clean samples
3. Perform statistical test: if separation is significant (p < 0.05), watermark is present
4. Ownership confirmed if **either** watermark A **or** watermark B is detected

---

## Architecture

```
Input Image
    ↓
[Backbone CNN]
    ↓
Features (f)
    ├─→ [Task Classifier] → Predictions
    ├─→ [Projection Head A] → z_A → Score A
    └─→ [Projection Head B] → z_B → Score B
              ↑
              └─── [Entanglement Layer] ← detach(z_A)
```

---

## Key Components

### Watermark Scores
Computed as cosine similarity between projected features and secret keys:
```
score_A = dot(normalize(z_A), k_A)
score_B = dot(normalize(z_B), k_B)
```

### Entanglement Effect
When an adversary attacks watermark A:
- Projection A is modified to reduce score_A
- Due to entanglement, projection B receives compensating signals
- Score_B remains high despite the attack

Similarly, attacking B preserves A.

### Statistical Verification
Uses two-sample t-test comparing:
- Scores on watermarked data (should be high)
- Scores on clean data (should be low)

Watermark presence confirmed if: `p-value < 0.05`

---

## Method Properties

**Dual Defense:**
At least one watermark survives removal attempts in most scenarios.

**Orthogonal Design:**
Keys in perpendicular subspaces prevent simultaneous minimization through simple gradient descent.

**Gradient Trap:**
The detachment operation creates asymmetric information flow that preserves watermark signals during attacks.

**Architecture Agnostic:**
Works with standard layers (Linear, ReLU, Conv2d). No custom operations required.

**Minimal Overhead:**
Adds ~6% parameters (projection heads). No inference latency impact.

---

## Experimental Dataset

FashionMNIST: 70,000 grayscale images (28×28 pixels), 10 clothing categories.

---

## Model Configurations Tested

| Model  | Conv Channels | Feature Dim | Parameters |
|--------|---------------|-------------|------------|
| Small  | [16, 32]      | 64          | ~85K       |
| Medium | [32, 64]      | 128         | ~340K      |
| Large  | [64, 128]     | 256         | ~1.36M     |

---

## Hyperparameters

**Key Dimension:** 64  
**Entanglement Strength (α):** 0.7  
**Watermark Ratio:** 10% of training data  
**Task Training:** 15 epochs, AdamW, lr=1e-3  
**Watermark Training:** 10 epochs, Adam, lr=1e-3  
**Margin:** 0.1  

---

## Attack Scenarios Evaluated

1. **Targeted Attack A**: Optimize to remove watermark A specifically
2. **Targeted Attack B**: Optimize to remove watermark B specifically  
3. **Dual Attack**: Simultaneously target both watermarks
4. **Fine-Tuning**: Continue training on clean data (3, 5, 10 epochs)
5. **Pruning**: Remove weights by magnitude (30%, 50%, 70%)



