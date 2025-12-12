!pip install opencv-python-headless numpy matplotlib scipy -q
!pip install ultralytics -q

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from google.colab import files
import os
import zipfile
import re
import json

!pip install opencv-python-headless numpy matplotlib scipy -q

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from google.colab import files
import os
import zipfile
import re

print("="*60)
print("STEP 1: Processing Lower Body")
print("="*60)

lower_file_path = None
search_paths = ['/content/lower.png', '/content/Lower.png', './lower.png', 'lower.png']

for path in search_paths:
    if os.path.exists(path):
        lower_file_path = path
        break

if lower_file_path is None and os.path.exists('/content'):
    for f in os.listdir('/content'):
        if 'lower' in f.lower() and f.endswith('.png'):
            lower_file_path = f'/content/{f}'
            break

if lower_file_path is None:
    print("Please upload lower.png:")
    uploaded = files.upload()
    for filename in uploaded.keys():
        if 'lower' in filename.lower():
            lower_file_path = filename
            break

# Load and process lower body
lower_img = cv2.imread(lower_file_path)
lower_img = cv2.cvtColor(lower_img, cv2.COLOR_BGR2RGB)

# Create lower body silhouette (threshold 180, inverted)
gray = cv2.cvtColor(lower_img, cv2.COLOR_RGB2GRAY)
inverted = 255 - gray
blurred = cv2.GaussianBlur(inverted, (5, 5), 0)
_, binary = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)

kernel_small = np.ones((3, 3), np.uint8)
kernel_medium = np.ones((5, 5), np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=2)
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_medium, iterations=3)
binary = ndimage.binary_fill_holes(binary).astype(np.uint8) * 255

contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
lower_silhouette = np.zeros_like(binary)
for cnt in contours:
    if cv2.contourArea(cnt) > 500:
        cv2.drawContours(lower_silhouette, [cnt], -1, 255, -1)

# INVERT for WHITE on BLACK
lower_silhouette = 255 - lower_silhouette



print("\n" + "="*60)
print("STEP 2: Upload thermal_images.zip")
print("="*60)

uploaded = files.upload()

IMAGE_DIR = './thermal_images'
for filename in uploaded.keys():
    if filename.endswith('.zip'):
        print(f"Extracting {filename}...")
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(IMAGE_DIR)
        break


print("\n" + "="*60)
print("STEP 3: Processing upper body images")
print("="*60)

IMAGE_RANGES = {
    'idle_1': (1, 10), 'pose_1': (11, 20),
    'idle_2': (21, 30), 'pose_2': (31, 40),
    'idle_3': (41, 50), 'pose_3': (51, 60),
    'idle_4': (61, 70), 'pose_4': (71, 80)
}

POSE_TO_IDLE = {
    'pose_1': 'idle_1', 'pose_2': 'idle_2',
    'pose_3': 'idle_3', 'pose_4': 'idle_4'
}

def get_image_number(filename):
    base = os.path.splitext(filename)[0]
    match = re.match(r'^(\d+)$', base)
    return int(match.group(1)) if match else None

def is_valid_image(filename):
    if filename.startswith('._') or filename.startswith('.'):
        return False
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        return False
    return get_image_number(filename) is not None

def find_images_dir(base_dir):
    for root, dirs, files_list in os.walk(base_dir):
        valid = [f for f in files_list if is_valid_image(f)]
        if len(valid) >= 10:
            return root
    return base_dir

IMAGE_DIR = find_images_dir('./thermal_images')
IMAGE_MAP = {}
for f in os.listdir(IMAGE_DIR):
    if is_valid_image(f):
        num = get_image_number(f)
        if num:
            IMAGE_MAP[num] = f

print(f"Found {len(IMAGE_MAP)} images in {IMAGE_DIR}")

def load_image_by_number(num):
    if num not in IMAGE_MAP:
        return None
    path = os.path.join(IMAGE_DIR, IMAGE_MAP[num])
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None

def load_burst(start, end):
    return [img for i in range(start, end+1) if (img := load_image_by_number(i)) is not None]

def average_burst(images):
    return np.mean(np.stack(images).astype(np.float32), axis=0).astype(np.uint8) if images else None

def subtract_background(pose, idle):
    return np.clip(pose.astype(np.float32) - idle.astype(np.float32), 0, 255).astype(np.uint8)

def extract_silhouette(img):
    thermal = img[:, :, 0]
    blurred = cv2.GaussianBlur(thermal, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(binary)
        cv2.drawContours(mask, [largest], -1, 255, -1)
        return mask
    return binary

def crop_to_content(sil, padding=10):
    rows = np.where(sil.sum(axis=1) > 0)[0]
    cols = np.where(sil.sum(axis=0) > 0)[0]
    if len(rows) == 0 or len(cols) == 0:
        return sil
    y1, y2 = max(0, rows[0] - padding), min(sil.shape[0], rows[-1] + padding + 1)
    x1, x2 = max(0, cols[0] - padding), min(sil.shape[1], cols[-1] + padding + 1)
    return sil[y1:y2, x1:x2]

# Process all poses
averaged_images = {}
for name, (start, end) in IMAGE_RANGES.items():
    burst = load_burst(start, end)
    if burst:
        averaged_images[name] = average_burst(burst)

upper_silhouettes = {}
for pose, idle in POSE_TO_IDLE.items():
    if pose in averaged_images and idle in averaged_images:
        subtracted = subtract_background(averaged_images[pose], averaged_images[idle])
        sil = extract_silhouette(subtracted)
        upper_silhouettes[pose] = crop_to_content(sil, padding=10)
        print(f"  {pose}: {upper_silhouettes[pose].shape}")

# Crop lower body too
lower_cropped = crop_to_content(lower_silhouette, padding=10)
print(f"  lower: {lower_cropped.shape}")

print("\n" + "="*60)
print("STEP 4: Displaying all silhouettes")
print("="*60)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Row 1: Poses 1-3
for i, name in enumerate(['pose_1', 'pose_2', 'pose_3']):
    if name in upper_silhouettes:
        axes[0, i].imshow(upper_silhouettes[name], cmap='gray')
        h, w = upper_silhouettes[name].shape
        axes[0, i].set_title(f'{name}\n{w}x{h}')
    axes[0, i].axis('off')

# Row 2: Pose 4, Lower Body
if 'pose_4' in upper_silhouettes:
    axes[1, 0].imshow(upper_silhouettes['pose_4'], cmap='gray')
    h, w = upper_silhouettes['pose_4'].shape
    axes[1, 0].set_title(f'pose_4\n{w}x{h}')
axes[1, 0].axis('off')

axes[1, 1].imshow(lower_cropped, cmap='gray')
h, w = lower_cropped.shape
axes[1, 1].set_title(f'LOWER BODY\n{w}x{h}')
axes[1, 1].axis('off')

axes[1, 2].axis('off')

plt.suptitle('Upper Body Poses + Lower Body (WHITE on BLACK)', fontsize=16)
plt.tight_layout()
plt.savefig('all_silhouettes.png', dpi=150)
plt.show()


print("\n" + "="*60)
print("STEP 5: Saving individual images")
print("="*60)

os.makedirs('./silhouettes', exist_ok=True)

for name, sil in upper_silhouettes.items():
    cv2.imwrite(f'./silhouettes/{name}.png', sil)
    print(f"  Saved {name}.png")

cv2.imwrite('./silhouettes/lower.png', lower_cropped)
print(f"  Saved lower.png")

# Zip and download
!zip -r silhouettes.zip silhouettes/
files.download('silhouettes.zip')

print("\n✓ Done! All silhouettes saved and downloaded.")
print("\nDimensions:")
for name, sil in upper_silhouettes.items():
    h, w = sil.shape
    print(f"  {name}: {w}x{h}")
h, w = lower_cropped.shape
print(f"  lower: {w}x{h}")

print("Looking for lower.png...")

lower_file_path = None
search_paths = ['/content/lower.png', '/content/Lower.png', './lower.png', 'lower.png']

for path in search_paths:
    if os.path.exists(path):
        lower_file_path = path
        break

if lower_file_path is None and os.path.exists('/content'):
    for f in os.listdir('/content'):
        if 'lower' in f.lower() and f.endswith('.png'):
            lower_file_path = f'/content/{f}'
            break

if lower_file_path is None:
    print("Please upload lower.png:")
    uploaded = files.upload()
    for filename in uploaded.keys():
        lower_file_path = filename
        break

img = cv2.imread(lower_file_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(f"✓ Loaded: {img.shape}")

gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
inverted = 255 - gray  # Now legs are BRIGHT (white), background is DARK

# Apply blur to reduce noise
blurred = cv2.GaussianBlur(inverted, (5, 5), 0)

print(f"Inverted grayscale - min: {blurred.min()}, max: {blurred.max()}")

def extract_silhouette_from_gray(gray_img, threshold_value, min_area=500):
    """
    Extract silhouette from grayscale image.
    Input should already be: legs=BRIGHT, background=DARK
    Output: WHITE legs on BLACK background
    """
    # Apply threshold
    if threshold_value == 'otsu':
        _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif threshold_value == 'adaptive':
        binary = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 51, -5)
    else:
        _, binary = cv2.threshold(gray_img, threshold_value, 255, cv2.THRESH_BINARY)

    # Morphological cleanup
    kernel_small = np.ones((3, 3), np.uint8)
    kernel_medium = np.ones((5, 5), np.uint8)

    # Remove small noise
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=2)

    # Close gaps
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_medium, iterations=2)

    # Fill holes
    binary = ndimage.binary_fill_holes(binary).astype(np.uint8) * 255

    # Remove small components (noise)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(binary)
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            cv2.drawContours(mask, [cnt], -1, 255, -1)

    return mask

# Test thresholds focused on the working range
thresholds = [140, 150, 160, 165, 170, 175, 180, 185, 190, 195, 200, 'otsu', 'adaptive']

# Calculate grid size
n_cols = 4
n_rows = (len(thresholds) + 3 + n_cols - 1) // n_cols  # +3 for original, gray, inverted

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
axes = axes.flatten()

# Show original
axes[0].imshow(img)
axes[0].set_title('Original')
axes[0].axis('off')

# Show grayscale
axes[1].imshow(gray, cmap='gray')
axes[1].set_title('Grayscale')
axes[1].axis('off')

# Show inverted + blurred (this is what we threshold)
axes[2].imshow(blurred, cmap='gray')
axes[2].set_title('Inverted + Blurred\n(INPUT for thresholding)')
axes[2].axis('off')

# Test each threshold
for i, thresh in enumerate(thresholds):
    sil = extract_silhouette_from_gray(blurred, thresh, min_area=500)
    axes[i + 3].imshow(sil, cmap='gray')

    # Count white pixels to show coverage
    white_pct = (np.sum(sil > 0) / sil.size) * 100
    axes[i + 3].set_title(f'Threshold: {thresh}\n({white_pct:.1f}% white)')
    axes[i + 3].axis('off')

# Hide unused axes
for j in range(len(thresholds) + 3, len(axes)):
    axes[j].axis('off')

plt.suptitle('Silhouette from INVERTED Grayscale - Fine-Tuned Thresholds', fontsize=16)
plt.tight_layout()
plt.savefig('threshold_comparison_v2.png', dpi=150)
plt.show()


blur_sizes = [3, 5, 7, 9, 11, 15]
threshold_val = 175

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

axes[0].imshow(img)
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(inverted, cmap='gray')
axes[1].set_title('Inverted (no blur)')
axes[1].axis('off')

for i, blur_size in enumerate(blur_sizes):
    blurred_test = cv2.GaussianBlur(inverted, (blur_size, blur_size), 0)
    sil = extract_silhouette_from_gray(blurred_test, threshold_val, min_area=500)
    axes[i + 2].imshow(sil, cmap='gray')
    axes[i + 2].set_title(f'Blur: {blur_size}x{blur_size}, Thresh: {threshold_val}')
    axes[i + 2].axis('off')

plt.suptitle('Effect of Blur Size on Silhouette Quality', fontsize=16)
plt.tight_layout()
plt.savefig('blur_comparison.png', dpi=150)
plt.show()
\


