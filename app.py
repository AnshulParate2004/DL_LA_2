import streamlit as st
import numpy as np
import json
import io
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from salg import SALG, Detection, SemanticGroup, groups_to_json

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH = "yolo26s_doclaynet_best.pt"

CLASSES = [
    'Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer',
    'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'
]

REMAP = {
    'Title'          : 'Text',
    'Section-header' : 'Picture',
}

GROUP_COLORS = {
    'text'            : '#3498DB',
    'title'           : '#27AE60',
    'image'           : '#F39C12',
    'flow'            : '#3498DB',
    'margin'          : '#95A5A6',
    'isolated_caption': '#BDC3C7',
}

GROUP_REMAP = {
    'float'  : 'title',
    'title'  : 'text',
    'section': 'image',
}

# ── Load model once ───────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

# ── Helper: smart remap ───────────────────────────────────────────────────────
def smart_remap(d):
    name = REMAP.get(d.cls_name, d.cls_name)
    if d.cls_name == 'Table':
        aspect = d.w / max(d.h, 1)
        if aspect > 4:
            name = 'Text'
    return Detection(
        cls_name = name,
        cls_id   = CLASSES.index(name),
        conf     = d.conf,
        box      = d.box
    )

# ── Helper: draw groups on image ──────────────────────────────────────────────
def draw_groups(img: Image.Image, groups) -> Image.Image:
    img = img.copy().convert('RGBA')
    overlay = Image.new('RGBA', img.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)

    for g in groups:
        x1,y1,x2,y2 = g.bbox
        label = GROUP_REMAP.get(g.group_type, g.group_type)
        hex_c = GROUP_COLORS.get(label, '#AAAAAA').lstrip('#')
        r,g_c,b = tuple(int(hex_c[i:i+2],16) for i in (0,2,4))

        # Fill
        draw.rectangle([x1,y1,x2,y2], fill=(r,g_c,b,30))
        # Border
        draw.rectangle([x1,y1,x2,y2], outline=(r,g_c,b,220), width=2)
        # Label
        draw.rectangle([x1,y1,x1+120,y1+18], fill=(r,g_c,b,200))
        draw.text((x1+4,y1+2), f'#{g.reading_order} {label}', fill=(255,255,255,255))

    return Image.alpha_composite(img, overlay).convert('RGB')

# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title='Document Layout Analyser', layout='wide')
st.title('📄 Document Layout Analyser')
st.caption('Upload a document image — YOLO + SALG will detect and group layout elements.')

uploaded = st.file_uploader('Upload image', type=['png','jpg','jpeg','webp'])

with st.sidebar:
    st.header('⚙️ Settings')
    conf_thresh = st.slider('YOLO confidence',  0.10, 0.80, 0.25, 0.05)
    iou_thresh  = st.slider('YOLO IoU',         0.10, 0.80, 0.45, 0.05)
    min_conf    = st.slider('SALG min conf',     0.10, 0.80, 0.35, 0.05)
    show_json   = st.checkbox('Show JSON output', value=True)

if uploaded:
    img = Image.open(uploaded).convert('RGB')
    img_w, img_h = img.size

    with st.spinner('Running inference...'):
        model = load_model()
        result = model(
            np.array(img),
            imgsz=1280, conf=conf_thresh, iou=iou_thresh,
            device='cpu', verbose=False
        )[0]

    # Build detections
    detections = [
        Detection(CLASSES[int(b.cls)], int(b.cls), float(b.conf), tuple(b.xyxy[0].tolist()))
        for b in result.boxes
    ]

    # Remap labels
    detections = [smart_remap(d) for d in detections]

    # Run SALG
    salg = SALG(img_h=img_h, img_w=img_w, min_conf=min_conf)
    groups = salg.group(detections)

    # Rename group types
    for g in groups:
        g.group_type = GROUP_REMAP.get(g.group_type, g.group_type)

    # Draw
    annotated = draw_groups(img, groups)

    # Display
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Original')
        st.image(img, use_container_width=True)
    with col2:
        st.subheader(f'Layout Groups ({len(groups)} found)')
        st.image(annotated, use_container_width=True)

    # Download annotated image
    buf = io.BytesIO()
    annotated.save(buf, format='PNG')
    st.download_button('⬇️ Download annotated image', buf.getvalue(),
                       file_name='layout.png', mime='image/png')

    # Reading order summary
    st.subheader('📋 Reading Order')
    for g in groups:
        st.write(f'`[{g.reading_order:02d}]` **{g.group_type}** — '
                 f'{[e.cls_name for e in g.elements]}')

    # JSON output
    if show_json:
        st.subheader('🗂️ JSON Output')
        json_out = groups_to_json(groups, uploaded.name)
        st.json(json_out)
        st.download_button('⬇️ Download JSON', json.dumps(json_out, indent=2),
                           file_name='layout.json', mime='application/json')
