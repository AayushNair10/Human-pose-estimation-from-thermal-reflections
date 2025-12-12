# -*- coding: utf-8 -*-
pip install torch diffusers transformers accelerate

import cv2
import numpy as np
import os
import torch
from diffusers import MarigoldDepthPipeline
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

INPUT_FOLDER = 'thermal_image'
OUTPUT_FOLDER = 'processed_output'
RGB_REFERENCE_PATH = '01.jpg'

PAIRS = [
    (1, 10, 11, 20),
    (21, 30, 31, 40),
    (41, 50, 51, 60),
    (61, 70, 71, 80),
    (81, 90, 91, 100)
]

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print(f"Created output folder: {OUTPUT_FOLDER}")
else:
    print(f"Output folder ready: {OUTPUT_FOLDER}")

def get_cooker_roi_and_mask_from_array(depth_img, target_shape):
    h, w = target_shape
    dsize = (int(w), int(h))
    depth = cv2.resize(depth_img, dsize, interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(depth, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None, None, None
    c = max(contours, key=cv2.contourArea)
    x, y, w_rect, h_rect = cv2.boundingRect(c)
    return x, y, w_rect, h_rect, mask

def load_and_average(start_idx, end_idx, folder):
    accum = None
    count = 0
    for i in range(start_idx, end_idx + 1):
        filename = f"{i}.JPG"
        path = os.path.join(folder, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        if accum is None:
            accum = img.astype(np.float32)
        else:
            accum += img.astype(np.float32)
        count += 1
    if count == 0: return None
    return (accum / count).astype(np.uint8)

def uncurl_cylindrical(img, roi, strength=1.0):
    x, y, w, h = roi
    y2 = min(y+h, img.shape[0])
    x2 = min(x+w, img.shape[1])
    crop = img[y:y2, x:x2]
    height, width = crop.shape
    if width == 0 or height == 0: return img
    map_x = np.zeros((height, width), np.float32)
    map_y = np.zeros((height, width), np.float32)
    center_x = width / 2
    radius = width / (2 * np.sin(0.5 * strength))
    for i in range(width):
        for j in range(height):
            norm_x = (i - center_x) / (width / 2)
            theta = norm_x * (np.pi / 2) * 0.5
            src_x = center_x + radius * np.sin(theta)
            map_x[j, i] = src_x
            map_y[j, i] = j
    return cv2.remap(crop, map_x, map_y, cv2.INTER_LINEAR)

MANUAL_ROTATE_FIX = False
ROTATION_ANGLE = -90

def fix_image_orientation(img):
    img = ImageOps.exif_transpose(img)
    if MANUAL_ROTATE_FIX:
        img = img.rotate(ROTATION_ANGLE, expand=True)
    return img

if not os.path.exists(RGB_REFERENCE_PATH):
    print(f"ERROR: File not found: {RGB_REFERENCE_PATH}")
else:
    print(f"Loading Marigold model...")
    try:
        pipe = MarigoldDepthPipeline.from_pretrained(
            "prs-eth/marigold-lcm-v1-0",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)
        print(f"Model loaded on {device}.")
        raw_image = Image.open(RGB_REFERENCE_PATH)
        input_image = fix_image_orientation(raw_image)
        print(f"Input image size: {input_image.size}")
        print("Running depth estimation...")
        with torch.no_grad():
            pipeline_output = pipe(
                input_image,
                num_inference_steps=4,
                ensemble_size=1,
                processing_resolution=768,
                match_input_resolution=True
            )
        if hasattr(pipeline_output, 'prediction'):
            depth_tensor = pipeline_output.prediction
        elif hasattr(pipeline_output, 'depth_np'):
            depth_tensor = torch.from_numpy(pipeline_output.depth_np)
        else:
            depth_tensor = pipeline_output[0]
        depth_pred = depth_tensor.cpu().numpy() if hasattr(depth_tensor, 'cpu') else depth_tensor
        depth_pred = np.squeeze(depth_pred)
        depth_min, depth_max = depth_pred.min(), depth_pred.max()
        depth_norm = (depth_pred - depth_min) / (depth_max - depth_min)
        generated_depth = (depth_norm * 255).astype(np.uint8)
        print(f"Depth map ready: {generated_depth.shape}")
        plt.figure(figsize=(6,6))
        plt.imshow(generated_depth, cmap='inferno')
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"Error: {e}")

if 'generated_depth' not in globals():
    print("Missing depth map.")
else:
    sample_path = os.path.join(INPUT_FOLDER, '1.JPG')
    sample_img = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
    if sample_img is None:
        print(f"ERROR: Missing thermal sample {sample_path}")
        cooker_mask = None
    else:
        thermal_h, thermal_w = sample_img.shape
        print(f"Thermal size: {thermal_w}x{thermal_h}")
        try:
            roi_x, roi_y, roi_w, roi_h, cooker_mask = get_cooker_roi_and_mask_from_array(generated_depth, (thermal_h, thermal_w))
            if roi_x is None:
                print("ERROR: Cooker not found")
            else:
                print(f"Cooker ROI: x={roi_x}, y={roi_y}, w={roi_w}, h={roi_h}")
                plt.imshow(cooker_mask, cmap='gray')
                plt.axis('off')
                plt.show()
        except Exception as e:
            print(f"Mask error: {e}")
            cooker_mask = None

CYLINDER_STRENGTH = 1.2
STRETCH_X = 2.8
HEAD_EXPANSION = 1.4
LEG_EXPANSION = 1.0

def correct_perspective(img, top_factor=1.0, bottom_factor=1.0):
    h, w = img.shape[:2]
    src_pts = np.float32([[0,0],[w,0],[0,h],[w,h]])
    max_factor = max(top_factor, bottom_factor)
    new_w = int(w * max_factor)
    top_w = int(w * top_factor)
    top_pad = (new_w - top_w)//2
    btm_w = int(w * bottom_factor)
    btm_pad = (new_w - btm_w)//2
    dst_pts = np.float32([
        [top_pad,0],
        [top_pad+top_w,0],
        [btm_pad,h],
        [btm_pad+btm_w,h]
    ])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv0.warpPerspective(img, M, (new_w, h))

def smart_center_and_crop(img):
    _, thresh = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return img
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    pad_x, pad_y = 20, 10
    return img[max(0,y-pad_y):min(img.shape[0],y+h+pad_y), max(0,x-pad_x):min(img.shape[1],x+w+pad_x)]

if 'cooker_mask' not in globals() or cooker_mask is None:
    print("Mask missing.")
else:
    print(f"Processing...")
    for idx, (idle_start, idle_end, pose_start, pose_end) in enumerate(PAIRS):
        avg_idle = load_and_average(idle_start, idle_end, INPUT_FOLDER)
        avg_pose = load_and_average(pose_start, pose_end, INPUT_FOLDER)
        if avg_idle is None or avg_pose is None: continue
        diff = avg_pose.astype(np.float32) - avg_idle.astype(np.float32)
        diff[diff < 0] = 0
        diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        masked_diff = cv2.bitwise_and(diff_norm, diff_norm, mask=cooker_mask)
        uncurled_raw = uncurl_cylindrical(masked_diff, (roi_x, roi_y, roi_w, roi_h), strength=CYLINDER_STRENGTH)
        centered = smart_center_and_crop(uncurled_raw)
        corrected = correct_perspective(centered, top_factor=HEAD_EXPANSION, bottom_factor=LEG_EXPANSION)
        h, w = corrected.shape
        final_w = int(w * STRETCH_X)
        final_result = cv2.resize(corrected, (final_w, h), interpolation=cv2.INTER_LINEAR)
        out_path = os.path.join(OUTPUT_FOLDER, f'set_{idx+1}_final.png')
        cv2.imwrite(out_path, final_result)
        plt.figure(figsize=(4,6))
        plt.imshow(final_result, cmap='inferno')
        plt.axis('off')
        plt.show()
