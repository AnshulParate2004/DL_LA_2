import io, sys, json
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from salg.salg import (
    SALG, Detection,
    CLASSES, GROUP_REMAP,
    DETECTION_COLORS, GROUP_COLORS,
    groups_to_json,
)

st.set_page_config(page_title="DocLayout YOLO + SALG", page_icon="📄", layout="wide")

INFER_CONF  = 0.25
INFER_IOU   = 0.45
INFER_IMGSZ = 1280
SALG_VERT_GAP  = 0.06
SALG_NMS_IOU   = 0.45
SALG_MIN_CONF  = 0.35
SALG_FLOAT_GAP = 0.03

@st.cache_resource(show_spinner="Loading model…")
def load_model(path: str):
    from ultralytics import YOLO
    return YOLO(path)

def run_inference(model, img, device):
    result = model(img, imgsz=INFER_IMGSZ, conf=INFER_CONF,
                   iou=INFER_IOU, device=device, verbose=False)[0]
    iw, ih = img.size
    return [
        Detection(CLASSES[int(b.cls)], int(b.cls),
                  float(b.conf), tuple(b.xyxy[0].tolist()))
        for b in result.boxes
    ], iw, ih

def run_salg(detections, iw, ih):
    salg = SALG(img_h=ih, img_w=iw,
                vert_gap_ratio=SALG_VERT_GAP, nms_iou_thresh=SALG_NMS_IOU,
                min_conf=SALG_MIN_CONF, float_merge_gap=SALG_FLOAT_GAP)
    groups = salg.group(detections)
    for g in groups:
        g.group_type = GROUP_REMAP.get(g.group_type, g.group_type)
    return groups, salg

def build_figure(img, detections, groups):
    fig, axes = plt.subplots(1, 2, figsize=(22, 14))

    ax = axes[0]
    ax.imshow(img)
    ax.set_title('YOLO Raw Detections', fontsize=14, fontweight='bold')
    for det in detections:
        x1, y1, x2, y2 = det.box
        c = DETECTION_COLORS.get(det.cls_name, '#AAAAAA')
        ax.add_patch(patches.Rectangle((x1,y1), x2-x1, y2-y1,
                     linewidth=1.5, edgecolor=c, facecolor='none', alpha=0.9))
        ax.text(x1+2, y1-4, f'{det.cls_name} {det.conf:.2f}',
                fontsize=6, color=c,
                bbox=dict(facecolor='white', alpha=0.5, pad=1, edgecolor='none'))
    ax.axis('off')

    ax = axes[1]
    ax.imshow(img)
    ax.set_title('SALG Semantic Groups (reading order)', fontsize=14, fontweight='bold')
    for g in groups:
        x1, y1, x2, y2 = g.bbox
        c = GROUP_COLORS.get(g.group_type, '#AAAAAA')
        ax.add_patch(patches.Rectangle((x1,y1), x2-x1, y2-y1,
                     linewidth=2, edgecolor=c, facecolor=c, alpha=0.12))
        ax.add_patch(patches.Rectangle((x1,y1), x2-x1, y2-y1,
                     linewidth=2, edgecolor=c, facecolor='none'))
        ax.text(x1+4, y1+14, f'#{g.reading_order} {g.group_type}',
                fontsize=7, color='white', fontweight='bold',
                bbox=dict(facecolor=c, alpha=0.85, pad=2, edgecolor='none'))
    ax.axis('off')

    plt.tight_layout()
    return fig

with st.sidebar:
    st.header("⚙️ Settings")
    model_path = st.text_input(
        "Model path (.pt)",
        value=r"D:\Projects_Main\DL_LA_2\yolo26s_doclaynet_best.pt",
    )
    device = st.selectbox("Device", ["cpu", "0", "cuda"], index=0)

st.title("📄 Document Layout Analysis")
st.markdown(
    "**YOLOv12-S** fine-tuned on **DocLayNet** — detects 11 document layout classes "
    "and applies **SALG** (Semantic-Aware Layout Grouping) to build a clean reading order."
)
st.divider()

uploaded = st.file_uploader(
    "Upload a document image",
    type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
)
if uploaded is None:
    st.stop()

img = PILImage.open(uploaded).convert("RGB")
iw, ih = img.size
st.image(img, caption=f"{uploaded.name}  ·  {iw}×{ih} px", use_container_width=True)

if not Path(model_path).exists():
    st.error(f"**Model not found:** `{model_path}`  \nUpdate the path in the sidebar ↖")
    st.stop()

model = load_model(model_path)

with st.spinner("Running YOLO inference…"):
    detections, iw, ih = run_inference(model, img, device)

with st.spinner("Applying SALG…"):
    groups, salg_obj = run_salg(detections, iw, ih)

c1, c2, c3 = st.columns(3)
c1.metric("Raw detections", len(detections))
c2.metric("SALG groups",    len(groups))
n_cols = SALG(ih, iw)._detect_columns(
    [d for d in detections if d.cls_name in SALG.FLOW_CLASSES | SALG.HEADING_CLASSES])
c3.metric("Columns detected", n_cols)

st.subheader("Visualisation")
with st.spinner("Rendering…"):
    fig = build_figure(img, detections, groups)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
st.image(buf, use_container_width=True)

with st.expander("📊 Detections by class "):
    counts = Counter(g.group_type for g in groups)
    if counts:
        names = list(counts.keys())
        vals  = [counts[k] for k in names]
        fig2, ax2 = plt.subplots(figsize=(8, max(3, len(names) * 0.55)))
        bars = ax2.barh(names, vals, color=[GROUP_COLORS.get(n, '#AAA') for n in names])
        ax2.bar_label(bars, padding=3, fontsize=10)
        ax2.set_xlabel("Count"); ax2.set_title("Groups per type (post-SALG + remap)"); ax2.invert_yaxis()
        plt.tight_layout(); st.pyplot(fig2); plt.close(fig2)
    else:
        st.write("No groups found.")

with st.expander("📋 SALG reading order"):
    st.table([
        {"Order": g.reading_order, "Type": g.group_type,
         "Elements": ", ".join(e.cls_name for e in g.elements),
         "Bbox": [round(v) for v in g.bbox]}
        for g in groups
    ])

st.subheader("💾 Export")
dl1, dl2 = st.columns(2)
stem = Path(uploaded.name).stem
with dl1:
    buf.seek(0)
    st.download_button("⬇️ Visualisation (PNG)", data=buf,
                       file_name=f"{stem}_doclayout.png", mime="image/png")
with dl2:
    st.download_button("⬇️ SALG layout (JSON)",
                       data=json.dumps(groups_to_json(groups, uploaded.name), indent=2).encode(),
                       file_name=f"{stem}_salg.json", mime="application/json")
