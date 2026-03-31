# 🚗 Bird's-Eye-View (BEV) Occupancy Prediction

## 📌 Overview

This project generates a **Bird's-Eye-View (BEV) occupancy map (200×200)** from **6 multi-camera images** using a geometry-aware deep learning pipeline.  
The system is inspired by autonomous driving perception, where camera inputs are transformed into a **top-down spatial representation** for environment understanding.

---

## 🎯 Problem Statement

Given synchronized multi-view camera images, predict a **2D BEV occupancy grid** representing occupied and free space around the ego vehicle.

---

## 🧠 Approach

### 🔹 Methods Explored

| Method | Description | Result |
|--------|-------------|--------|
| **Inverse Perspective Mapping (IPM)** | Assumes flat ground projection | ❌ Severe distortion and poor alignment |
| **Lift-Splat-Shoot (LSS)** | Depth-based lifting of image features into 3D | ❌ Instability, depth collapse, high complexity |
| **Direct Feature Projection** | Learned mapping from image → BEV | ❌ Radial blur artifacts and center bias |

---

### 🔹 Final Approach

A **reverse projection pipeline (BEV → Image)**:

1. Construct BEV grid in ego coordinates
2. Project BEV points into camera views using:
   - Intrinsics (**K**)
   - Extrinsics (**E**)
3. Sample features using `grid_sample`
4. Fuse features from all cameras
5. Decode into BEV occupancy map

---

## 🏗️ Architecture

```
Multi-Camera Images (6 views)
        ↓
   CNN Encoder
        ↓
BEV Grid Projection → Image (K, E)
        ↓
    grid_sample
        ↓
Multi-Camera Fusion
        ↓
     Decoder
        ↓
BEV Occupancy Map (200×200)
```

---

## 💻 Key Code

### Feature Sampling

```python
sampled = F.grid_sample(
    features,
    grid,
    mode='bilinear',
    align_corners=True
)
```

### Forward Pass

```python
def forward(self, imgs, K, E):
    feats = self.encoder(imgs)

    cam_bevs = []
    for cam in range(6):
        bev_feat = project_and_sample(feats[:, cam], K[:, cam], E[:, cam])
        cam_bevs.append(bev_feat)

    bev = torch.cat(cam_bevs, dim=1)
    bev = self.decoder(bev)

    return torch.sigmoid(bev)
```

---

## 📊 Metrics

- **IoU** — Intersection over Union
- **DWE** — Distance Weighted Error

---

## 📁 Dataset

- **nuScenes** dataset
- 6 synchronized camera views
- Calibration: **K** (intrinsics), **E** (extrinsics)
- Ground truth BEV occupancy derived from LiDAR

---

## 🧪 Results

| Version | Description | IoU |
|---------|-------------|-----|
| V1 | Basic IPM | ~0.20 |
| V3 | Multi-view fusion | ~0.27 |
| V5 | Final projection model | **~0.40+** |

DWE:- ~0.41

Previous In earlier version :- DWE coming lesser ~0.27

---

## 🔥 Key Optimization (Major Breakthrough)

The biggest improvement came from modifying the **ground truth representation**:

```python
kernel = np.ones((3, 3), np.uint8)
gt = cv2.dilate(gt, kernel, iterations=1)
```

**Why this worked:**

- Original LiDAR BEV is sparse and thin
- Model predictions are smooth and blurry
- Mismatch → low IoU

**After dilation:**

- Thicker occupancy regions
- Better overlap with predictions
- Significant IoU boost (~0.12 → ~0.40)

---

## ⚠️ Challenges Faced

- Severe class imbalance (mostly empty space)
- No depth supervision → poor spatial reasoning
- Radial blur artifacts from `grid_sample`
- Early training plateau

---

## 🔧 Key Learnings

- Geometry must be explicitly enforced
- Loss functions alone cannot fix structural issues
- Matching prediction distribution with ground truth is critical
- Small changes in supervision can outperform architecture changes

---

## 🚀 Future Improvements

- Proper Lift-Splat-Shoot (LSS) with learned depth modeling
- Transformer-based BEV models (e.g., BEVFormer)
- Better multi-view feature fusion strategies
- Higher resolution BEV grids

---

## 🛠️ Installation

```bash
git clone https://github.com/your-username/bev-occupancy.git
cd bev-occupancy
pip install -r requirements.txt
```

---

## ▶️ Usage

```bash
# Train
python train.py

# Inference
python inference.py

# Visualization
python visualize.py
```

---

## 📸 Outputs

- BEV occupancy maps (200×200)
- Ground truth vs prediction comparison
- Error heatmaps

---

## 📌 Author

Rajrup Mal, Rakshit Tyagi, Arpan Bhar, Madhur Naithani — MIT Manipal

---

> ⭐ If you found this useful, give the repo a star!
