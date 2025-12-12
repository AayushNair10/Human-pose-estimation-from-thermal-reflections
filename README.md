# Human Pose Estimation from Curved Thermal Reflections
---
Recovering human pose from distorted thermal reflections using generative depth segmentation, geometric rectification, and full-body silhouette fusion.

---

## Project Overview
Conventional thermal imaging works only in direct line-of-sight.  
However, thermal emissions also reflect off shiny curved surfaces (pressure cooker, kettle, metal objects), creating extremely distorted reflections that still contain usable information.

This project reconstructs a **proportional human silhouette** from a **single curved thermal reflection** using:

- Generative depth estimation (Marigold, MoGe)
- Geometric distortion correction (radial, homography)
- Anthropometric ratio stretching  
- Optional full-body fusion using planar mirror silhouettes

---

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Main Libraries Used

Python 3.8+

OpenCV

PyTorch

Numpy

Marigold depth estimator

MoGe depth estimator

YOLO pose model

---

## Method Pipeline
1. Data Acquisition
- Thermal camera: Topdon TC002C (256×192, 25 Hz)

- RGB camera for depth estimation

- Curved reflector: stainless-steel pressure cooker

- Burst-frame averaging removes microbolometer noise

- Background subtraction removes static room heat

2. Segmentation via Generative Depth
- Marigold depth model identifies cooker as solid object

- MoGe alternative depth estimator

- Depth map → clean mask of reflective region

3. Multi-Stage Geometric Rectification
(Core of the method)

- Radial Distortion Correction

- Brown–Conrady model

- Removes spherical “barrel” distortion

- Keystone Correction (Homography)

- Expands upper-body region

- Fixes “pinhead effect”

- Biological Ratio Stretching

- Target aspect ratio: height:width ≈ 2.1 : 1

- Auto-computes horizontal expansion

4. Full-Body Silhouette Fusion
- Upper body → cooker reflection

- Lower body → planar mirror

- Width alignment + Gaussian seam blending

---

## Results
The system successfully recovers:

Upper-body shapes

Human proportions

Pose silhouettes suitable for keypoint extraction

The results outperform naive unwrapping, eliminating extreme compression artifacts.

---

## Limitations
Single viewpoint → Only 2D silhouettes (not full 3D)

Self-occlusions merge in low-resolution thermal data

Manual tuning needed for reflector curvature

Requires separate RGB camera (thermal camera lacks RGB)

---

## Future Work
Automatic curvature estimation using a CNN

Thermal super-resolution GAN for higher fidelity

Stereo curved-reflector NLOS

Full 3D mesh recovery from curved reflections

