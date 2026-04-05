import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import fitz  # PyMuPDF
import tempfile
import os

# --- Page Config ---
st.set_page_config(
    page_title="YOLO Document Layout Analysis",
    page_icon="📄",
    layout="wide",
)

# --- Styling ---
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    .stProgress .st-bo {
        background-color: #007bff;
    }
</style>
""", unsafe_allow_html=True)

# --- Title & Description ---
st.title("🎯 YOLO Document Layout Analysis")
st.markdown("""
Upload a **PDF**, **PNG**, or **JPG** document to detect layout elements like **Titles, Tables, Pictures, and Text Blocks**.
Powered by YOLOv8/v10/v11 (DocLayNet).
""")

# --- Constants & Sidebar ---
CLASS_NAMES = {
    0: 'Caption',
    1: 'Footnote',
    2: 'Formula',
    3: 'List-item',
    4: 'Page-footer',
    5: 'Page-header',
    6: 'Picture',
    7: 'Section-header',
    8: 'Table',
    9: 'Text',
    10: 'Title'
}

# Distinct colors for each class (RGBA)
CLASS_COLORS = {
    0: (255, 165, 0),    # Caption: Orange
    1: (128, 128, 128),  # Footnote: Gray
    2: (128, 0, 128),    # Formula: Purple
    3: (0, 191, 255),    # List-item: DeepSkyBlue
    4: (50, 205, 50),     # Page-footer: LimeGreen
    5: (34, 139, 34),     # Page-header: ForestGreen
    6: (255, 0, 0),       # Picture: Red
    7: (255, 69, 0),      # Section-header: RedOrange
    8: (0, 255, 255),     # Table: Cyan
    9: (255, 255, 0),     # Text: Yellow
    10: (220, 20, 60)     # Title: Crimson
}

st.sidebar.title("🛠️ Model Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25)
iou_threshold = st.sidebar.slider("IoU Threshold", 0.1, 1.0, 0.45)
model_path = "final_doclayout_model.pt"

# --- Model Loading ---
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        st.error(f"Model file not found at {path}. Please ensure 'final_doclayout_model.pt' is in the project root.")
        return None
    return YOLO(path)

model = load_model(model_path)

# --- Helper Functions ---
def draw_detections(image, detections):
    """Draw bounding boxes and labels on the image."""
    img_array = np.array(image)
    overlay = img_array.copy()
    
    for det in detections:
        x1, y1, x2, y2 = map(int, det.xyxy[0])
        conf = float(det.conf[0])
        cls = int(det.cls[0])
        
        color = CLASS_COLORS.get(cls, (255, 255, 255))
        label = f"{CLASS_NAMES.get(cls, 'Unknown')} {conf:.2f}"
        
        # Draw bounding box
        cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 2)
        
        # Transparent fill
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        
        # Label background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img_array, (x1, y1 - 25), (x1 + w, y1), color, -1)
        cv2.putText(img_array, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Blend overlay for transparency
    alpha = 0.2
    cv2.addWeighted(overlay, alpha, img_array, 1 - alpha, 0, img_array)
    return Image.fromarray(img_array)

# --- File Processing ---
uploaded_file = st.file_uploader("Upload Document", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file is not None and model is not None:
    file_extension = uploaded_file.name.split(".")[-1].lower()
    
    if file_extension == "pdf":
        # Process PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        doc = fitz.open(tmp_path)
        total_pages = len(doc)
        
        st.sidebar.divider()
        page_number = st.sidebar.number_input(f"Page (1-{total_pages})", min_value=1, max_value=total_pages, value=1) - 1
        
        # Convert PDF page to PIL Image
        page = doc[page_number]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Scale up for better detection
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        doc.close()
        os.remove(tmp_path)
    else:
        # Process Image
        img = Image.open(uploaded_file).convert("RGB")

    # --- Inference ---
    with st.spinner("Analyzing layout..."):
        results = model.predict(img, conf=conf_threshold, iou=iou_threshold)
        detections = results[0].boxes
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🖼️ Layout Visualization")
        result_img = draw_detections(img, detections)
        st.image(result_img, use_column_width=True)
        
    with col2:
        st.subheader("📊 Detected Elements")
        if len(detections) > 0:
            counts = {}
            for det in detections:
                cls = int(det.cls[0])
                name = CLASS_NAMES.get(cls, "Unknown")
                counts[name] = counts.get(name, 0) + 1
            
            for name, count in sorted(counts.items()):
                st.write(f"**{name}:** {count}")
                
            # Detailed Table
            det_data = []
            for det in detections:
                cls = int(det.cls[0])
                name = CLASS_NAMES.get(cls, "Unknown")
                conf = float(det.conf[0])
                det_data.append({"Element": name, "Confidence": f"{conf:.2%}"})
            st.table(det_data)
        else:
            st.info("No elements detected with current settings.")

else:
    if model is None:
        st.warning("Please upload a trained model first.")
    else:
        st.info("Upload a file to get started!")
