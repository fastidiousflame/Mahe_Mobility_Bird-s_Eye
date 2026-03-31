# 🚗 Bird’s-Eye-View (BEV) Occupancy Prediction

## 📌 Overview
This project generates a **Bird’s-Eye-View (BEV) occupancy map (200×200)** from **6 multi-camera images** using a geometry-aware deep learning pipeline.

It is inspired by autonomous driving systems where camera inputs are transformed into a top-down spatial representation for understanding surroundings.

---

## 🎯 Problem Statement
Given synchronized multi-view images, predict a **2D BEV occupancy grid** representing occupied and free space around the ego vehicle.

---

## 🧠 Approach

### Methods Explored

- **Inverse Perspective Mapping (IPM)**  
  Simple flat-ground projection  
  Result: distortion and poor alignment  

- **Lift-Splat-Shoot (LSS)**  
  Depth-based lifting to 3D and projection to BEV  
  Result: instability, depth collapse, high complexity  

- **Direct Feature Projection**  
  Learned mapping from images to BEV  
  Result: radial blur artifacts and center bias  

---

### Final Approach (Used)

We use a **reverse projection pipeline (BEV → Image)**:

- Construct BEV grid in ego coordinates  
- Project BEV points into each camera using **K (intrinsics)** and **E (extrinsics)**  
- Sample image features using `grid_sample`  
- Fuse features from all cameras  
- Decode into BEV occupancy map  

---

## 🏗️ Architecture

Multi-Camera Images (6 views)  
↓  
CNN Encoder  
↓  
BEV Grid Projection → Image (K, E)  
↓  
grid_sample  
↓  
Multi-camera Fusion  
↓  
Decoder  
↓  
BEV Occupancy Map (200×200)

---

## 💻 Key Code

```python
# Feature sampling
sampled = F.grid_sample(
    features,
    grid,
    mode='bilinear',
    align_corners=True
)

## 💻 Key Code

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

## 📊 Metrics

- IoU (Intersection over Union)
- Distance Weighted Error (DWE)

---

## 📁 Dataset

- nuScenes  
- 6 camera views  
- Calibration: K (intrinsics), E (extrinsics)  
- Ground truth BEV from LiDAR  

---

## 🧪 Results

| Version | Approach           | IoU   |
|--------|-------------------|-------|
| V1     | IPM               | ~0.20 |
| V2     | Final model       | ~0.27+ |

---

## ⚠️ Challenges

- Class imbalance (mostly empty space)  
- Depth ambiguity  
- Radial blur from interpolation  
- Training plateau  

---

## 🚀 Run

```bash
git clone https://github.com/fastidiousflame/Mahe_Mobility_Bird-s_Eye.git
cd Mahe_Mobility_Bird-s_Eye
pip install -r requirements.txt

python train.py
python inference.py
```

---

## 📌 Author

Rajrup Mal , Rakshit Tyagi, Arpan Bhar, Madhur Naithani
(MIT Manipal) 

---

## ⭐ Star this repo if you found it useful
