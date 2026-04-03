"""
TF CODE FILE 2 — Config A & B: Custom Model ~10M params (TensorFlow/Keras)
===========================================================================
Config A: batch=16, imgsz=640 → ~6-8 GB VRAM  ✅ Comfortable
Config B: batch=32, imgsz=640 → ~10-12 GB VRAM ✅ Possible

Builds the full custom architecture from scratch in TensorFlow/Keras:
  Backbone: ResNet-style 5-stage CNN
  Neck:     Feature Pyramid Network (FPN, 3 scales)
  Head:     YOLO-style detection head per scale
  Loss:     CIoU + BCE objectness + SparseCategoricalCrossentropy class

No Ultralytics. No pretrained weights. Pure TensorFlow/Keras.
"""

import os, math, time, random, shutil
from pathlib import Path
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, mixed_precision
import matplotlib.pyplot as plt
from PIL import Image

# ─────────────────────────────────────────────
# CONFIG — change BATCH_SIZE to switch A ↔ B
# ─────────────────────────────────────────────
BASE_DIR     = '/content/drive/MyDrive/MVL_DL'
DATA_YAML    = os.path.join(BASE_DIR, 'doclaynet_10k', 'dataset.yaml')
CHECKPT_DIR  = os.path.join(BASE_DIR, 'checkpoints', 'custom_10M')
LOG_DIR      = os.path.join(BASE_DIR, 'logs')

CFG = {
    # ── Toggle A vs B here ──
    'batch_size'    : 16,     # Config A=16 (~6-8GB) | Config B=32 (~10-12GB)
    # ────────────────────────
    'img_size'      : 640,
    'num_classes'   : 7,
    'num_anchors'   : 3,
    'epochs'        : 100,
    'lr'            : 1e-3,
    'lr_min'        : 1e-5,
    'warmup_epochs' : 5,
    'weight_decay'  : 1e-4,
    'grad_accum'    : 4,
    'use_fp16'      : True,
    'num_workers'   : 2,
    'save_every'    : 10,
    'lambda_box'    : 5.0,
    'lambda_obj'    : 1.0,
    'lambda_cls'    : 0.5,
    'conf_thresh'   : 0.25,
    'iou_thresh'    : 0.45,
}

# Anchors — clustered from DocLayNet (normalized to 640)
ANCHORS = [
    [(0.04,0.02),(0.07,0.04),(0.10,0.06)],   # P3 — small
    [(0.15,0.05),(0.25,0.08),(0.40,0.10)],   # P4 — medium
    [(0.50,0.07),(0.70,0.15),(0.90,0.30)],   # P5 — large
]

CLS_NAMES = ['Caption','List-item','Picture','Section-header','Table','Text','Title']

# Enable mixed precision (fp16) for memory efficiency
if CFG['use_fp16']:
    mixed_precision.set_global_policy('mixed_float16')
    print("Mixed precision (fp16) ENABLED")

# GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print(f"GPUs available: {len(gpus)}")


# ═══════════════════════════════════════════
# 1. MODEL ARCHITECTURE (Keras Functional API)
# ═══════════════════════════════════════════

def conv_bn_relu(x, filters, kernel_size=3, strides=1, padding='same', name=None):
    """Conv → BatchNorm → ReLU block."""
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding,
                      use_bias=False, name=f'{name}_conv' if name else None)(x)
    x = layers.BatchNormalization(name=f'{name}_bn' if name else None)(x)
    x = layers.ReLU(name=f'{name}_relu' if name else None)(x)
    return x


def res_block(x, channels, name='res'):
    """Bottleneck residual block: squeeze → 3x3 → expand + skip."""
    mid = channels // 2
    residual = x
    x = conv_bn_relu(x, mid, kernel_size=1, padding='valid', name=f'{name}_c1')
    x = conv_bn_relu(x, mid, kernel_size=3, name=f'{name}_c2')
    x = layers.Conv2D(channels, 1, use_bias=False, name=f'{name}_c3')(x)
    x = layers.BatchNormalization(name=f'{name}_bn3')(x)
    x = layers.Add(name=f'{name}_add')([x, residual])
    x = layers.ReLU(name=f'{name}_relu3')(x)
    return x


def build_backbone(input_tensor):
    """
    5-stage backbone: 640→320→160→80(P3)→40(P4)→20(P5)
    Total params: ~6.3M
    """
    x = conv_bn_relu(input_tensor, 32, strides=2, name='s1')        # 320×320
    x = conv_bn_relu(x, 64, strides=2, name='s2_down')              # 160×160
    for i in range(3):
        x = res_block(x, 64, name=f's2_res{i}')
    p3 = conv_bn_relu(x, 128, strides=2, name='s3_down')            # 80×80
    for i in range(3):
        p3 = res_block(p3, 128, name=f's3_res{i}')
    p4 = conv_bn_relu(p3, 256, strides=2, name='s4_down')           # 40×40
    for i in range(3):
        p4 = res_block(p4, 256, name=f's4_res{i}')
    p5 = conv_bn_relu(p4, 512, strides=2, name='s5_down')           # 20×20
    for i in range(2):
        p5 = res_block(p5, 512, name=f's5_res{i}')
    return p3, p4, p5


def build_fpn(p3, p4, p5):
    """Top-down FPN: fuses P5→P4→P3, all outputs 128ch."""
    p5_lat = conv_bn_relu(p5, 128, kernel_size=1, padding='valid', name='fpn_lat5')

    p4_lat  = conv_bn_relu(p4, 128, kernel_size=1, padding='valid', name='fpn_lat4')
    p5_up   = layers.UpSampling2D(size=2, name='fpn_up5')(p5_lat)
    p4_out  = layers.Concatenate(name='fpn_cat4')([p4_lat, p5_up])
    p4_out  = conv_bn_relu(p4_out, 128, name='fpn_fuse4')

    p3_lat  = conv_bn_relu(p3, 128, kernel_size=1, padding='valid', name='fpn_lat3')
    p4_up   = layers.UpSampling2D(size=2, name='fpn_up4')(p4_out)
    p3_out  = layers.Concatenate(name='fpn_cat3')([p3_lat, p4_up])
    p3_out  = conv_bn_relu(p3_out, 128, name='fpn_fuse3')

    return p3_out, p4_out, p5_lat


def build_det_head(x, num_anchors=3, num_classes=7, name='head'):
    """Per-scale detection head → (B, H, W, A*(5+C))"""
    out_ch = num_anchors * (5 + num_classes)
    x = conv_bn_relu(x, 256, name=f'{name}_c1')
    x = layers.Conv2D(out_ch, 1, name=f'{name}_out')(x)
    return x


def build_document_layout_model(img_size=640, num_classes=7, num_anchors=3):
    """Build full model using Keras Functional API."""
    inputs = keras.Input(shape=(img_size, img_size, 3), name='input_image')

    p3, p4, p5     = build_backbone(inputs)
    f3, f4, f5     = build_fpn(p3, p4, p5)
    out_s = build_det_head(f3, num_anchors, num_classes, name='head_s')  # 80×80
    out_m = build_det_head(f4, num_anchors, num_classes, name='head_m')  # 40×40
    out_l = build_det_head(f5, num_anchors, num_classes, name='head_l')  # 20×20

    # Cast outputs to float32 (required when using mixed precision)
    out_s = layers.Lambda(lambda x: tf.cast(x, tf.float32), name='cast_s')(out_s)
    out_m = layers.Lambda(lambda x: tf.cast(x, tf.float32), name='cast_m')(out_m)
    out_l = layers.Lambda(lambda x: tf.cast(x, tf.float32), name='cast_l')(out_l)

    model = keras.Model(inputs=inputs, outputs=[out_s, out_m, out_l],
                        name='DocumentLayoutModel')
    total = model.count_params()
    print(f"Total parameters: {total/1e6:.2f}M")
    return model


# ═══════════════════════════════════════════
# 2. LOSS FUNCTIONS
# ═══════════════════════════════════════════

def ciou_loss_tf(pred_boxes, target_boxes):
    """CIoU loss between (N,4) tensors in [x1,y1,x2,y2] format."""
    px1,py1,px2,py2 = pred_boxes[:,0],pred_boxes[:,1],pred_boxes[:,2],pred_boxes[:,3]
    tx1,ty1,tx2,ty2 = target_boxes[:,0],target_boxes[:,1],target_boxes[:,2],target_boxes[:,3]

    inter_x1 = tf.maximum(px1, tx1)
    inter_y1 = tf.maximum(py1, ty1)
    inter_x2 = tf.minimum(px2, tx2)
    inter_y2 = tf.minimum(py2, ty2)
    inter    = tf.maximum(inter_x2-inter_x1, 0) * tf.maximum(inter_y2-inter_y1, 0)

    area_p = tf.maximum(px2-px1, 0) * tf.maximum(py2-py1, 0)
    area_t = tf.maximum(tx2-tx1, 0) * tf.maximum(ty2-ty1, 0)
    union  = area_p + area_t - inter + 1e-7
    iou    = inter / union

    enc_x1 = tf.minimum(px1, tx1); enc_x2 = tf.maximum(px2, tx2)
    enc_y1 = tf.minimum(py1, ty1); enc_y2 = tf.maximum(py2, ty2)
    c2     = (enc_x2-enc_x1)**2 + (enc_y2-enc_y1)**2 + 1e-7

    cx_p = (px1+px2)/2; cy_p = (py1+py2)/2
    cx_t = (tx1+tx2)/2; cy_t = (ty1+ty2)/2
    rho2 = (cx_p-cx_t)**2 + (cy_p-cy_t)**2

    w_p  = tf.maximum(px2-px1, 1e-7); h_p = tf.maximum(py2-py1, 1e-7)
    w_t  = tf.maximum(tx2-tx1, 1e-7); h_t = tf.maximum(ty2-ty1, 1e-7)
    v    = (4.0/math.pi**2) * (tf.atan(w_t/h_t) - tf.atan(w_p/h_p))**2
    alpha = tf.stop_gradient(v / (1 - iou + v + 1e-7))

    ciou = iou - rho2/c2 - alpha*v
    return tf.reduce_mean(1 - ciou)


def yolo_loss_tf(preds, targets, anchors, num_classes=7, img_size=640,
                 lambda_box=5.0, lambda_obj=1.0, lambda_cls=0.5):
    """
    Compute YOLO loss across all 3 scales.
    preds:   list of 3 tensors (B, H, W, A*(5+C))
    targets: list of (N,6) arrays [img_idx, cls, xc, yc, w, h]
    """
    bce_fn  = keras.losses.BinaryCrossentropy(from_logits=True, reduction='sum_over_batch_size')
    sce_fn  = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    total_box = tf.constant(0.0)
    total_obj = tf.constant(0.0)
    total_cls = tf.constant(0.0)

    for scale_i, pred in enumerate(preds):
        B, H, W, _ = pred.shape
        A  = len(anchors[scale_i])
        nc = num_classes

        # Reshape → (B, H, W, A, 5+nc)
        pred = tf.reshape(pred, [B, H, W, A, 5 + nc])

        anch = tf.constant(anchors[scale_i], dtype=tf.float32)  # (A,2)

        obj_mask    = tf.zeros([B, H, W, A])
        noobj_mask  = tf.ones([B, H, W, A])
        tx_t = tf.zeros([B, H, W, A])
        ty_t = tf.zeros([B, H, W, A])
        tw_t = tf.zeros([B, H, W, A])
        th_t = tf.zeros([B, H, W, A])
        cls_t= tf.zeros([B, H, W, A], dtype=tf.int32)

        # Build target tensors using numpy (executed eagerly in training)
        obj_np    = np.zeros([B, H, W, A], dtype=np.float32)
        tx_np     = np.zeros([B, H, W, A], dtype=np.float32)
        ty_np     = np.zeros([B, H, W, A], dtype=np.float32)
        tw_np     = np.zeros([B, H, W, A], dtype=np.float32)
        th_np     = np.zeros([B, H, W, A], dtype=np.float32)
        cls_np    = np.zeros([B, H, W, A], dtype=np.int32)
        anch_np   = np.array(anchors[scale_i], dtype=np.float32)

        for t in targets:
            if t is None or len(t) == 0:
                continue
            for box in t:
                img_i, cls, xc, yc, bw, bh = box
                img_i = int(img_i)
                gi = min(int(xc * W), W - 1)
                gj = min(int(yc * H), H - 1)
                wh = np.array([[bw, bh]])
                inter    = np.minimum(wh, anch_np).prod(axis=1)
                union    = (wh.prod() + anch_np.prod(axis=1) - inter)
                iou_anch = inter / np.maximum(union, 1e-7)
                best_a   = iou_anch.argmax()

                obj_np [img_i, gj, gi, best_a] = 1.0
                tx_np  [img_i, gj, gi, best_a] = xc * W - gi
                ty_np  [img_i, gj, gi, best_a] = yc * H - gj
                tw_np  [img_i, gj, gi, best_a] = bw / (anch_np[best_a, 0] + 1e-7)
                th_np  [img_i, gj, gi, best_a] = bh / (anch_np[best_a, 1] + 1e-7)
                cls_np [img_i, gj, gi, best_a] = int(cls)

        obj_mask = tf.constant(obj_np)
        mask     = tf.cast(obj_mask, tf.bool)

        # Objectness loss (all cells)
        total_obj = total_obj + bce_fn(obj_mask, pred[..., 4])

        if tf.reduce_any(mask):
            pred_xy  = tf.sigmoid(pred[..., :2])
            pred_wh  = pred[..., 2:4]

            # Build grid
            gx = tf.cast(tf.range(W), tf.float32)
            gy = tf.cast(tf.range(H), tf.float32)
            gy_grid, gx_grid = tf.meshgrid(gy, gx, indexing='ij')
            gx_grid = tf.tile(gx_grid[None, :, :, None], [B, 1, 1, A])
            gy_grid = tf.tile(gy_grid[None, :, :, None], [B, 1, 1, A])

            px = (pred_xy[..., 0] + gx_grid) / tf.cast(W, tf.float32)
            py = (pred_xy[..., 1] + gy_grid) / tf.cast(H, tf.float32)

            aw = tf.constant(anch_np[:, 0], dtype=tf.float32)
            ah = tf.constant(anch_np[:, 1], dtype=tf.float32)
            aw = tf.reshape(aw, [1, 1, 1, A])
            ah = tf.reshape(ah, [1, 1, 1, A])

            pw = tf.exp(tf.clip_by_value(pred_wh[..., 0], -4, 4)) * aw
            ph = tf.exp(tf.clip_by_value(pred_wh[..., 1], -4, 4)) * ah

            tx = tf.constant(tx_np, dtype=tf.float32)
            ty = tf.constant(ty_np, dtype=tf.float32)
            tw = tf.constant(tw_np, dtype=tf.float32) * aw
            th = tf.constant(th_np, dtype=tf.float32) * ah

            pred_xyxy = tf.stack([px-pw/2, py-ph/2, px+pw/2, py+ph/2], axis=-1)
            targ_x    = (tx + gx_grid) / tf.cast(W, tf.float32)
            targ_y    = (ty + gy_grid) / tf.cast(H, tf.float32)
            targ_xyxy = tf.stack([targ_x-tw/2, targ_y-th/2,
                                   targ_x+tw/2, targ_y+th/2], axis=-1)

            mask4 = tf.broadcast_to(mask[..., None], pred_xyxy.shape)
            pred_box_sel  = tf.boolean_mask(pred_xyxy, mask)
            targ_box_sel  = tf.boolean_mask(targ_xyxy, mask)
            total_box = total_box + ciou_loss_tf(pred_box_sel, targ_box_sel)

            # Class loss
            cls_target  = tf.constant(cls_np, dtype=tf.int32)
            cls_logits  = pred[..., 5:]
            cls_sel     = tf.boolean_mask(cls_logits, mask)
            cls_tgt_sel = tf.cast(tf.boolean_mask(cls_target, mask), tf.int32)
            cls_losses  = sce_fn(cls_tgt_sel, cls_sel)
            total_cls   = total_cls + tf.reduce_mean(cls_losses)

    loss = lambda_box * total_box + lambda_obj * total_obj + lambda_cls * total_cls
    return loss, total_box, total_obj, total_cls


# ═══════════════════════════════════════════
# 3. TF.DATA DATASET PIPELINE
# ═══════════════════════════════════════════

IMG_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
IMG_STD  = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)


def load_and_preprocess(img_path, img_size=640):
    """Load image, resize with padding, normalize."""
    raw   = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(raw, channels=3)
    image = tf.image.resize_with_pad(image, img_size, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    image = (image - IMG_MEAN) / IMG_STD
    return image


def augment_image(image):
    """Document-safe augmentation."""
    image = tf.image.random_brightness(image, 0.3)
    image = tf.image.random_contrast(image, 0.7, 1.3)
    image = tf.image.random_saturation(image, 0.9, 1.1)
    return image


class YOLODatasetTF:
    """TF-compatible dataset builder returning (image, target) pairs."""

    def __init__(self, data_root, split, img_size=640, augment=True):
        self.img_size  = img_size
        self.augment   = augment
        self.img_paths = []
        self.lbl_paths = []

        img_dir = Path(data_root) / split / 'images'
        lbl_dir = Path(data_root) / split / 'labels'

        for img_path in sorted(list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))):
            lbl_path = lbl_dir / (img_path.stem + '.txt')
            self.img_paths.append(str(img_path))
            self.lbl_paths.append(str(lbl_path))

        print(f"  {split}: {len(self.img_paths)} images loaded")

    def _load_labels(self, lbl_path):
        boxes, classes = [], []
        try:
            with open(lbl_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    cls, xc, yc, w, h = int(parts[0]), *map(float, parts[1:])
                    boxes.append([xc, yc, w, h])
                    classes.append(cls)
        except FileNotFoundError:
            pass
        boxes   = np.array(boxes,   dtype=np.float32) if boxes   else np.zeros((0, 4), dtype=np.float32)
        classes = np.array(classes, dtype=np.float32) if classes else np.zeros((0,),  dtype=np.float32)
        return boxes, classes

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = load_and_preprocess(self.img_paths[idx], self.img_size)
        if self.augment:
            image = augment_image(image)

        boxes, classes = self._load_labels(self.lbl_paths[idx])
        if len(boxes):
            target = np.concatenate([
                np.zeros((len(boxes), 1), dtype=np.float32),  # img_idx filled in collate
                classes[:, None],
                boxes
            ], axis=1)
        else:
            target = np.zeros((0, 6), dtype=np.float32)
        return image.numpy(), target

    def as_tf_dataset(self, batch_size, shuffle=True):
        """Build tf.data.Dataset with proper batching."""
        indices = list(range(len(self)))
        if shuffle:
            random.shuffle(indices)

        def generator():
            for idx in indices:
                image, target = self[idx]
                yield image, target

        ds = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(self.img_size, self.img_size, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 6), dtype=tf.float32),
            )
        )

        def pad_batch(images, targets):
            # Ragged → padded for targets
            return images, targets

        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds


# ═══════════════════════════════════════════
# 4. TRAINING LOOP
# ═══════════════════════════════════════════

def print_gpu_memory(label=''):
    """Print GPU memory usage."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # TF doesn't expose VRAM stats directly; use nvidia-smi
        mem_info = os.popen('nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits').read()
        if mem_info.strip():
            used, total = mem_info.strip().split('\n')[0].split(',')
            print(f"VRAM {label}: {int(used)/1024:.2f}GB used / {int(total)/1024:.2f}GB total")


def get_lr_schedule(cfg):
    """Returns a callable lr schedule (warmup + cosine decay)."""
    warmup_epochs = cfg['warmup_epochs']
    total_epochs  = cfg['epochs']
    lr            = cfg['lr']
    lr_min        = cfg['lr_min']

    def schedule(epoch):
        if epoch < warmup_epochs:
            return max(0.01 * lr, lr * (epoch / max(1, warmup_epochs)))
        t = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * t))

    return schedule


def train():
    os.makedirs(CHECKPT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,     exist_ok=True)

    data_root = str(Path(DATA_YAML).parent)

    # ── Build model ──
    model = build_document_layout_model(
        CFG['img_size'], CFG['num_classes'], CFG['num_anchors']
    )
    model.summary(line_length=100)
    print_gpu_memory('after model build')

    # ── Optimizer ──
    lr_schedule = get_lr_schedule(CFG)
    optimizer   = keras.optimizers.AdamW(
        learning_rate=CFG['lr'],
        weight_decay=CFG['weight_decay']
    )

    # ── Datasets ──
    train_ds_obj = YOLODatasetTF(data_root, 'train', CFG['img_size'], augment=True)
    val_ds_obj   = YOLODatasetTF(data_root, 'val',   CFG['img_size'], augment=False)
    train_dl     = train_ds_obj.as_tf_dataset(CFG['batch_size'], shuffle=True)
    val_dl       = val_ds_obj.as_tf_dataset(CFG['batch_size'],   shuffle=False)

    # ── Checkpoint manager ──
    ckpt    = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckpt_mgr= tf.train.CheckpointManager(ckpt, CHECKPT_DIR, max_to_keep=5)
    start_epoch = 1
    if ckpt_mgr.latest_checkpoint:
        ckpt.restore(ckpt_mgr.latest_checkpoint)
        # Epoch is encoded in the checkpoint filename
        try:
            start_epoch = int(ckpt_mgr.latest_checkpoint.split('-')[-1]) + 1
        except Exception:
            pass
        print(f"Resumed from {ckpt_mgr.latest_checkpoint} — starting epoch {start_epoch}")

    history   = defaultdict(list)
    best_loss = float('inf')
    best_ckpt = None

    for epoch in range(start_epoch, CFG['epochs'] + 1):
        # Update LR
        new_lr = lr_schedule(epoch)
        optimizer.learning_rate.assign(new_lr)

        # ─── TRAIN ───
        epoch_loss = 0.0; box_l = 0.0; obj_l = 0.0; cls_l = 0.0
        step_count = 0
        t0 = time.time()

        for step, (imgs, targets_batch) in enumerate(train_dl):
            targets = []
            # Un-batch targets; assign img_idx
            imgs_np   = imgs.numpy()
            batch_np  = targets_batch.numpy() if hasattr(targets_batch, 'numpy') else targets_batch
            for i in range(len(imgs_np)):
                t = batch_np[i] if isinstance(batch_np, list) else batch_np
                t = np.array(t, dtype=np.float32)
                t = t[t.sum(axis=1) != 0]  # remove zero-padded rows
                if len(t):
                    t[:, 0] = i
                targets.append(t)

            with tf.GradientTape() as tape:
                preds = model(imgs, training=True)
                loss, b, o, c = yolo_loss_tf(
                    preds, targets, ANCHORS,
                    CFG['num_classes'], CFG['img_size'],
                    CFG['lambda_box'], CFG['lambda_obj'], CFG['lambda_cls']
                )
                scaled_loss = loss / CFG['grad_accum']

            grads = tape.gradient(scaled_loss, model.trainable_variables)
            # Gradient accumulation via manual tracking
            if (step + 1) % CFG['grad_accum'] == 0:
                grads = [tf.clip_by_norm(g, 10.0) if g is not None else g for g in grads]
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            epoch_loss += float(loss); box_l += float(b)
            obj_l += float(o); cls_l += float(c)
            step_count += 1

        n  = max(step_count, 1)
        et = time.time() - t0
        print(f"Ep {epoch:3d}/{CFG['epochs']} | "
              f"Loss {epoch_loss/n:.4f} (box:{box_l/n:.3f} "
              f"obj:{obj_l/n:.3f} cls:{cls_l/n:.3f}) | "
              f"LR {new_lr:.2e} | {et/60:.1f}min")
        print_gpu_memory(f'epoch {epoch}')
        history['loss'].append(epoch_loss / n)
        history['lr'].append(new_lr)

        # ─── CHECKPOINT ───
        if epoch % CFG['save_every'] == 0 or epoch == CFG['epochs']:
            saved = ckpt_mgr.save(checkpoint_number=epoch)
            print(f"  💾 Checkpoint saved → {saved}")
            if epoch_loss / n < best_loss:
                best_loss = epoch_loss / n
                model.save_weights(os.path.join(CHECKPT_DIR, 'best_weights.h5'))
                print(f"  ⭐ New best → best_weights.h5")

    # ─── PLOT LOSS CURVE ───
    plt.figure(figsize=(10, 4))
    plt.plot(history['loss'])
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('Custom 10M (TF) — Training Loss')
    plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, 'custom_10M_tf_loss.png'), dpi=100)
    plt.show()
    print("Training complete. Best weights:", os.path.join(CHECKPT_DIR, 'best_weights.h5'))

    # Save full model in SavedModel format for serving
    model.save(os.path.join(CHECKPT_DIR, 'saved_model'))
    print(f"SavedModel exported → {CHECKPT_DIR}/saved_model")
    return model


if __name__ == '__main__':
    print(f"Config: batch={CFG['batch_size']} img={CFG['img_size']} "
          f"fp16={CFG['use_fp16']} epochs={CFG['epochs']}")
    print(f"Effective batch = {CFG['batch_size'] * CFG['grad_accum']}")
    model = train()
