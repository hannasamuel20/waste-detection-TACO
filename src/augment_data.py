import albumentations as A
import cv2
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

# --- Augmentation pipeline ---
transform = A.Compose([
    A.MotionBlur(p=0.3),
    A.RandomBrightnessContrast(p=0.4),
    A.RandomRotate90(p=0.5),
    A.Affine(translate_percent=0.1, scale=(0.9, 1.1), rotate=45, p=0.5),
    A.RandomCrop(width=640, height=640, p=0.3),
    A.Resize(height=640, width=640, p=1.0),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.2))

# --- Paths ---
project_dir = Path.home() / "course_project"
input_img_dir = project_dir / "data/TACO/yolo_dataset/train/images"
input_label_dir = project_dir / "data/TACO/yolo_dataset/train/labels"
output_img_dir = project_dir / "data/TACO/yolo_dataset/train_augmented/images"
output_label_dir = project_dir / "data/TACO/yolo_dataset/train_augmented/labels"

output_img_dir.mkdir(parents=True, exist_ok=True)
output_label_dir.mkdir(parents=True, exist_ok=True)

# --- Process and augment ---
image_files = [f for f in os.listdir(input_img_dir) if f.endswith('.jpg')]

for img_name in tqdm(image_files, desc="Augmenting"):
    img_path = input_img_dir / img_name
    label_path = input_label_dir / img_name.replace('.jpg', '.txt')

    # Load image and labels
    image = cv2.imread(str(img_path))
    if image is None or not label_path.exists():
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with open(label_path, 'r') as f:
        lines = f.readlines()

    bboxes, class_labels = [], []
    for line in lines:
        try:
            class_id, x, y, w, h = map(float, line.strip().split())
            x, y, w, h = np.clip([x, y, w, h], 0, 1)
            x_min, x_max = x - w / 2, x + w / 2
            y_min, y_max = y - h / 2, y + h / 2
            if 0 <= x_min <= 1 and 0 <= x_max <= 1 and 0 <= y_min <= 1 and 0 <= y_max <= 1:
                bboxes.append([x, y, w, h])
                class_labels.append(int(class_id))
        except ValueError:
            continue

    if not bboxes:
        continue

    # Apply augmentation
    try:
        augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    except Exception as e:
        print(f"Augmentation failed for {img_name}: {e}")
        continue

    aug_img = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
    aug_bboxes = augmented['bboxes']
    aug_classes = augmented['class_labels']

    if not aug_bboxes:
        continue

    # Save augmented image
    aug_img_name = f"aug_{img_name}"
    aug_img_path = output_img_dir / aug_img_name
    cv2.imwrite(str(aug_img_path), aug_img)

    # Save corresponding YOLO labels
    aug_label_path = output_label_dir / aug_img_name.replace('.jpg', '.txt')
    with open(aug_label_path, 'w') as f:
        for bbox, class_id in zip(aug_bboxes, aug_classes):
            f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

print(f"\n✅ Augmented {len(image_files)} images. Data saved to: {output_img_dir.parent}")
