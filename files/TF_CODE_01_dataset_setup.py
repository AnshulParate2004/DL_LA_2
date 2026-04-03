"""
TF CODE FILE 1 — Dataset Setup (TensorFlow Version)
====================================================
Download DocLayNet-v1.2-YOLO from Kaggle, verify structure,
create 10K subset, and validate labels.

Run this ONCE in Colab before any training.
Dataset is already in YOLO format — zero conversion needed.

Kaggle dataset: toshihikochen/doclaynet-v1-2-yolo
Classes (7 used of 11):
  0: Caption
  1: Footnote        (skip)
  2: Formula         (skip)
  3: List-item
  4: Page-footer     (skip)
  5: Page-header     (skip)
  6: Picture
  7: Section-header  ← PRIMARY (chunk boundary)
  8: Table
  9: Text
  10: Title          ← PRIMARY (chunk boundary)
"""

# ─────────────────────────────────────────────
# STEP 0 — Colab environment check
# ─────────────────────────────────────────────
import os, sys
import tensorflow as tf

print(f"Python      : {sys.version.split()[0]}")
print(f"TensorFlow  : {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs found  : {len(gpus)}")
if gpus:
    for gpu in gpus:
        print(f"  GPU: {gpu.name}")
    # Allow memory growth to avoid OOM
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# ─────────────────────────────────────────────
# STEP 1 — Mount Google Drive
# ─────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

BASE_DIR = '/content/drive/MyDrive/MVL_DL'
os.makedirs(BASE_DIR, exist_ok=True)
print(f"Base dir: {BASE_DIR}")

# ─────────────────────────────────────────────
# STEP 2 — Kaggle API credentials
# Upload your kaggle.json from kaggle.com → Account → API
# ─────────────────────────────────────────────
from google.colab import files
print("Upload your kaggle.json file now...")
uploaded = files.upload()  # select kaggle.json

os.makedirs('/root/.config/kaggle', exist_ok=True)
with open('/root/.config/kaggle/kaggle.json', 'wb') as f:
    f.write(uploaded['kaggle.json'])
os.chmod('/root/.config/kaggle/kaggle.json', 0o600)
print("✅ Kaggle API configured")

# ─────────────────────────────────────────────
# STEP 3 — Download DocLayNet-v1.2-YOLO
# ~28 GB full dataset — stored on Drive so it survives session restarts
# ─────────────────────────────────────────────
DATASET_DIR = os.path.join(BASE_DIR, 'doclaynet_yolo')
os.makedirs(DATASET_DIR, exist_ok=True)

if not os.path.exists(os.path.join(DATASET_DIR, 'train')):
    print("Downloading DocLayNet-v1.2-YOLO from Kaggle...")
    os.system(f'kaggle datasets download -d toshihikochen/doclaynet-v1-2-yolo -p {DATASET_DIR} --unzip')
    print("✅ Download complete")
else:
    print("✅ Dataset already downloaded — skipping")

# ─────────────────────────────────────────────
# STEP 4 — Inspect dataset structure
# ─────────────────────────────────────────────
from pathlib import Path
from collections import Counter

def inspect_split(split_name, images_dir, labels_dir):
    imgs   = list(Path(images_dir).glob('*.jpg')) + list(Path(images_dir).glob('*.png'))
    labels = list(Path(labels_dir).glob('*.txt'))
    print(f"\n{'─'*50}")
    print(f"Split: {split_name}")
    print(f"  Images : {len(imgs)}")
    print(f"  Labels : {len(labels)}")

    ALL_CLASSES = [
        'Caption','Footnote','Formula','List-item',
        'Page-footer','Page-header','Picture',
        'Section-header','Table','Text','Title'
    ]
    counts = Counter()
    for lbl in labels:
        for line in lbl.read_text().strip().split('\n'):
            if line.strip():
                cls = int(line.split()[0])
                counts[cls] += 1
    total = sum(counts.values())
    print(f"  Total annotations: {total}")
    print(f"  Per-class breakdown:")
    for i, name in enumerate(ALL_CLASSES):
        c = counts.get(i, 0)
        bar = '█' * int(30 * c / max(total, 1))
        print(f"    {i:2d} {name:15s}: {c:6d} ({100*c/max(total,1):5.1f}%) {bar}")

TRAIN_IMGS = os.path.join(DATASET_DIR, 'train', 'images')
TRAIN_LBLS = os.path.join(DATASET_DIR, 'train', 'labels')
VAL_IMGS   = os.path.join(DATASET_DIR, 'valid', 'images')
VAL_LBLS   = os.path.join(DATASET_DIR, 'valid', 'labels')
TEST_IMGS  = os.path.join(DATASET_DIR, 'test',  'images')
TEST_LBLS  = os.path.join(DATASET_DIR, 'test',  'labels')

for split, imgs, lbls in [('train', TRAIN_IMGS, TRAIN_LBLS),
                           ('val',   VAL_IMGS,   VAL_LBLS),
                           ('test',  TEST_IMGS,  TEST_LBLS)]:
    if os.path.exists(imgs):
        inspect_split(split, imgs, lbls)

# ─────────────────────────────────────────────
# STEP 5 — Remap classes to YOUR 7-class subset
# New mapping:
#   Caption(0)→0, List-item(3)→1, Picture(6)→2,
#   Section-header(7)→3, Table(8)→4, Text(9)→5, Title(10)→6
# ─────────────────────────────────────────────
import shutil

KEEP_MAP = {0: 0, 3: 1, 6: 2, 7: 3, 8: 4, 9: 5, 10: 6}
DROP_IDS = {1, 2, 4, 5}

def remap_labels(src_labels_dir, dst_labels_dir, keep_map, drop_ids):
    os.makedirs(dst_labels_dir, exist_ok=True)
    src_files = list(Path(src_labels_dir).glob('*.txt'))
    remapped  = 0
    dropped   = 0
    for src_path in src_files:
        lines = src_path.read_text().strip().split('\n')
        new_lines = []
        for line in lines:
            if not line.strip():
                continue
            parts = line.split()
            cls   = int(parts[0])
            if cls in drop_ids:
                dropped += 1
                continue
            if cls in keep_map:
                parts[0] = str(keep_map[cls])
                new_lines.append(' '.join(parts))
                remapped += 1
        dst_path = Path(dst_labels_dir) / src_path.name
        dst_path.write_text('\n'.join(new_lines))
    print(f"  Remapped: {remapped} annotations, Dropped: {dropped}")

REMAP_DIR = os.path.join(BASE_DIR, 'doclaynet_7cls')
for split, src_imgs, src_lbls in [
        ('train', TRAIN_IMGS, TRAIN_LBLS),
        ('val',   VAL_IMGS,   VAL_LBLS),
        ('test',  TEST_IMGS,  TEST_LBLS)]:
    if not os.path.exists(src_imgs):
        continue
    dst_imgs = os.path.join(REMAP_DIR, split, 'images')
    dst_lbls = os.path.join(REMAP_DIR, split, 'labels')
    if not os.path.exists(dst_imgs):
        os.makedirs(os.path.dirname(dst_imgs), exist_ok=True)
        os.symlink(src_imgs, dst_imgs)
    print(f"Remapping {split} labels...")
    remap_labels(src_lbls, dst_lbls, KEEP_MAP, DROP_IDS)
    print(f"  ✅ {split} done")

# ─────────────────────────────────────────────
# STEP 6 — Write dataset.yaml (used by all 4 training configs)
# ─────────────────────────────────────────────
YAML_CONTENT = f"""
path: {REMAP_DIR}
train: train/images
val:   val/images
test:  test/images

nc: 7
names:
  0: Caption
  1: List-item
  2: Picture
  3: Section-header
  4: Table
  5: Text
  6: Title
""".strip()

yaml_path = os.path.join(REMAP_DIR, 'dataset.yaml')
with open(yaml_path, 'w') as f:
    f.write(YAML_CONTENT)
print(f"\n✅ dataset.yaml written → {yaml_path}")
print(YAML_CONTENT)

# ─────────────────────────────────────────────
# STEP 7 — Create 10K subset for fast iteration
# ─────────────────────────────────────────────
import random

def create_subset(src_imgs, src_lbls, dst_imgs, dst_lbls, n=10_000, seed=42):
    os.makedirs(dst_imgs, exist_ok=True)
    os.makedirs(dst_lbls, exist_ok=True)
    all_imgs = (
        list(Path(src_imgs).glob('*.jpg')) +
        list(Path(src_imgs).glob('*.png'))
    )
    random.seed(seed)
    selected = random.sample(all_imgs, min(n, len(all_imgs)))
    for img_path in selected:
        lbl_path = Path(src_lbls) / (img_path.stem + '.txt')
        shutil.copy(img_path, dst_imgs)
        if lbl_path.exists():
            shutil.copy(lbl_path, dst_lbls)
    print(f"  ✅ Subset: {len(selected)} images")

SUBSET_DIR = os.path.join(BASE_DIR, 'doclaynet_10k')
create_subset(
    os.path.join(REMAP_DIR, 'train', 'images'),
    os.path.join(REMAP_DIR, 'train', 'labels'),
    os.path.join(SUBSET_DIR, 'train', 'images'),
    os.path.join(SUBSET_DIR, 'train', 'labels'),
    n=10_000
)
for split in ['val', 'test']:
    for kind in ['images', 'labels']:
        src = os.path.join(REMAP_DIR, split, kind)
        dst = os.path.join(SUBSET_DIR, split, kind)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copytree(src, dst)

subset_yaml = YAML_CONTENT.replace(REMAP_DIR, SUBSET_DIR)
with open(os.path.join(SUBSET_DIR, 'dataset.yaml'), 'w') as f:
    f.write(subset_yaml)
print(f"✅ Subset yaml written → {SUBSET_DIR}/dataset.yaml")

# ─────────────────────────────────────────────
# STEP 8 — TF Data Pipeline: verify images load correctly
# Uses tf.data to test the pipeline instead of PIL visualization
# ─────────────────────────────────────────────
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

CLS_NAMES  = ['Caption','List-item','Picture','Section-header','Table','Text','Title']
CLS_COLORS = ['#FF6B6B','#4ECDC4','#45B7D1','#96CEB4','#FFEAA7','#DDA0DD','#FF8C00']

def load_image_tf(img_path, img_size=640):
    """Load and preprocess image using TensorFlow."""
    raw   = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(raw, channels=3)
    image = tf.image.resize_with_pad(image, img_size, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image

def visualize_sample_tf(images_dir, labels_dir, n=3):
    imgs = list(Path(images_dir).glob('*.jpg')) + list(Path(images_dir).glob('*.png'))
    fig, axes = plt.subplots(1, n, figsize=(6*n, 8))
    for ax, img_path in zip(axes, random.sample(imgs, min(n, len(imgs)))):
        img  = Image.open(img_path).convert('RGB')
        W, H = img.size
        draw = ImageDraw.Draw(img)
        lbl  = Path(labels_dir) / (img_path.stem + '.txt')
        if lbl.exists():
            for line in lbl.read_text().strip().split('\n'):
                if not line.strip(): continue
                cls, xc, yc, w, h = map(float, line.split())
                cls = int(cls)
                x1, y1 = (xc-w/2)*W, (yc-h/2)*H
                x2, y2 = (xc+w/2)*W, (yc+h/2)*H
                draw.rectangle([x1,y1,x2,y2], outline=CLS_COLORS[cls], width=3)
                draw.text((x1+2, y1+2), CLS_NAMES[cls], fill=CLS_COLORS[cls])
        ax.imshow(img)
        ax.set_title(img_path.name[:30], fontsize=8)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'sample_check.png'), dpi=100)
    plt.show()
    print("✅ Sample visualization saved")

visualize_sample_tf(
    os.path.join(SUBSET_DIR, 'train', 'images'),
    os.path.join(SUBSET_DIR, 'train', 'labels')
)

# Quick TF pipeline smoke test
print("\n── TF Data Pipeline Smoke Test ──")
sample_img_dir = os.path.join(SUBSET_DIR, 'train', 'images')
sample_paths   = [str(p) for p in list(Path(sample_img_dir).glob('*.jpg'))[:8]]
ds = tf.data.Dataset.from_tensor_slices(sample_paths)
ds = ds.map(load_image_tf, num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.batch(4)
for batch in ds.take(1):
    print(f"  Batch shape: {batch.shape}  dtype: {batch.dtype}  "
          f"min: {tf.reduce_min(batch):.3f}  max: {tf.reduce_max(batch):.3f}")
print("✅ TF data pipeline working")

print("\n" + "="*60)
print("DATASET SETUP COMPLETE")
print(f"Full remapped dataset : {REMAP_DIR}")
print(f"10K subset            : {SUBSET_DIR}")
print(f"dataset.yaml paths    : {REMAP_DIR}/dataset.yaml")
print(f"                        {SUBSET_DIR}/dataset.yaml")
print("Next step: run TF_CODE_02_custom_model_10M.py")
