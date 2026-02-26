import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Fabric Defect Detector",
    page_icon="🧵",
    layout="wide"
)

# --- SIDEBAR SETTINGS ---
st.sidebar.title("Settings")
st.sidebar.subheader("Model Configuration")

# Path to your trained model (Ensure this file exists)
# Based on your training script, likely: runs/detect/compare_yolo11n/weights/best.pt
MODEL_PATH = './model/yolov8_best.pt' 

# Allow user to upload a model if the default path isn't found
model_source = st.sidebar.radio("Model Source", ["Default Path", "Upload Model"])

conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
iou_threshold = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.05)

# --- LOAD MODEL ---
@st.cache_resource
def load_model(path):
    try:
        return YOLO(path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = None

if model_source == "Default Path":
    try:
        model = load_model(MODEL_PATH)
        # Add this temporary debug line
        if model:
            st.write("Debug - Model Classes:", model.names)
        st.sidebar.success(f"Loaded: {MODEL_PATH}")
    except:
        st.sidebar.warning(f"Default model not found at {MODEL_PATH}. Please check path or use Upload.")
else:
    uploaded_model = st.sidebar.file_uploader("Upload .pt file", type=["pt"])
    if uploaded_model:
        # Save temp to load
        with open("temp_model.pt", "wb") as f:
            f.write(uploaded_model.getbuffer())
        model = load_model("temp_model.pt")
        st.sidebar.success("Custom model loaded!")

# --- MAIN INTERFACE ---
st.title("🧵 Fabric Defect Detection System")
st.write("Upload a fabric image to detect defects using YOLO11n.")

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png', 'bmp'])

if uploaded_file is not None and model is not None:
    # 1. Read Image
    image = Image.open(uploaded_file)
    
    # 2. Run Inference
    # Forces the inference to match your training exactly
    results = model.predict(image, conf=conf_threshold, iou=iou_threshold, imgsz=416)
    
    # 3. Process Results
    result = results[0]
    
    # Render the detections on the image
    # plot() returns a BGR numpy array, we convert to RGB for Streamlit
    res_plotted = result.plot()[:, :, ::-1]
    
    # 4. Display Side-by-Side
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Detected Defects")
        st.image(res_plotted, use_container_width=True, caption="Inference Result")

    # 5. Show Statistics
    st.divider()
    st.subheader("Detection Details")
    
    if len(result.boxes) > 0:
        # Count classes
        class_counts = {}
        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = result.names[cls_id]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        st.write("Found the following defects:")
        # specific layout for stats
        cols = st.columns(len(class_counts))
        for idx, (name, count) in enumerate(class_counts.items()):
            with cols[idx]:
                st.metric(label=name, value=count)
                
        # Optional: Show raw data
        with st.expander("See Raw Bounding Box Data"):
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                st.write(f"- **{result.names[cls_id]}**: Confidence {conf:.2f}")
    else:
        st.success("No defects detected! Fabric appears clean.")

elif model is None:
    st.info("Please verify the model path in the code or upload a model file in the sidebar.")