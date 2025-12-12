# Load imports
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from ultralytics import YOLO
import json
from google.colab import files
import shutil

# Install packages
!pip install opencv-python-headless numpy matplotlib -q
!pip install ultralytics -q

# Load images
file_names = ['lower1.png', 'power2.png', 'lower3.png', 'lower4.png']
alt_names = ['lower1.png', 'lower2.png', 'lower3.png', 'lower4.png']
images = {}

for fname in file_names:
    if os.path.exists(fname):
        img = cv2.imread(fname)
        if img is not None:
            images[fname] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

if len(images) < 4:
    for fname in alt_names:
        if fname not in images and os.path.exists(fname):
            img = cv2.imread(fname)
            if img is not None:
                images[fname] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

if len(images) < 4:
    for fname in file_names + alt_names:
        path = f'/content/{fname}'
        if fname not in images and os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                images[fname] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Prepare images for YOLO

def prepare_for_yolo(img, target_height=640):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    pad = 30
    h, w = gray.shape
    padded = np.zeros((h + 2*pad, w + 2*pad), dtype=np.uint8)
    padded[pad:pad+h, pad:pad+w] = gray
    scale = target_height / padded.shape[0]
    new_w = max(1, int(padded.shape[1] * scale))
    resized = cv2.resize(padded, (new_w, target_height))
    gradient = np.linspace(1.0, 0.85, target_height).reshape(-1, 1)
    shaded = (resized * np.tile(gradient, (1, new_w))).astype(np.uint8)
    return cv2.cvtColor(shaded, cv2.COLOR_GRAY2RGB)

yolo_ready = {name: prepare_for_yolo(img) for name, img in images.items()}

# Run YOLO
models = {
    'nano': YOLO('yolov8n-pose.pt'),
    'small': YOLO('yolov8s-pose.pt'),
    'medium': YOLO('yolov8m-pose.pt'),
}

best_results = {}

for model_name, model in models.items():
    for name, img in yolo_ready.items():
        results = model(img, conf=0.05, verbose=False)
        has_detection = (
            len(results) > 0 and
            results[0].keypoints is not None and
            len(results[0].keypoints.xy) > 0 and
            results[0].keypoints.xy.shape[1] > 0
        )
        if has_detection:
            kp = results[0].keypoints.xy[0].cpu().numpy()
            valid = int(np.sum(np.any(kp > 0, axis=1)))
            if name not in best_results or valid > best_results[name]['valid']:
                best_results[name] = {
                    'keypoints': kp,
                    'valid': valid,
                    'model': model_name,
                    'result': results[0]
                }

# Visualize
KEYPOINT_NAMES = [
    'nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder',
    'left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip',
    'left_knee','right_knee','left_ankle','right_ankle'
]

n_images = len(images)
fig, axes = plt.subplots(n_images, 3, figsize=(18, 6 * n_images))
axes = axes.reshape(1, -1) if n_images == 1 else axes

for i, (name, img) in enumerate(images.items()):
    axes[i, 0].imshow(img)
    axes[i, 0].axis('off')
    axes[i, 1].imshow(yolo_ready[name])
    axes[i, 1].axis('off')
    if name in best_results:
        res = best_results[name]
        plot_img = res['result'].plot()
        axes[i, 2].imshow(cv2.cvtColor(plot_img, cv2.COLOR_BGR2RGB))
    else:
        axes[i, 2].imshow(yolo_ready[name])
    axes[i, 2].axis('off')

plt.tight_layout()
plt.savefig('pose_results.png', dpi=150)
plt.show()

# Keypoints
for name, res in best_results.items():
    kp = res['keypoints']
    for j in range(17):
        x, y = kp[j]

# Save
os.makedirs('./results', exist_ok=True)

for name, res in best_results.items():
    plot_img = res['result'].plot()
    safe = name.replace('.png', '')
    cv2.imwrite(f'./results/{safe}_pose.png', plot_img)

def to_json(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.integer, np.int64)): return int(obj)
    if isinstance(obj, (np.floating, np.float64)): return float(obj)
    return obj

export = {name:{'model':r['model'],'valid':r['valid'],
    'keypoints':{KEYPOINT_NAMES[j]:to_json(r['keypoints'][j]) for j in range(17)}}
    for name, r in best_results.items()}

with open('./results/keypoints.json', 'w') as f:
    json.dump(export, f, indent=2)

shutil.copy('pose_results.png', './results/')
!zip -r pose_results.zip results/
files.download('pose_results.zip')