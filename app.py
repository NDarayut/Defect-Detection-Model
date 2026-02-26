import streamlit as st
from ultralytics import YOLO
import cv2
import serial
import time
import numpy as np
import pandas as pd
import os
import re
from PIL import Image

# --- CONFIGURATION ---
TEST_IMAGES_FOLDER = "test_images"  # Folder containing your pre-captured images
MODEL_PATH = './model/yolov8_best.pt' # Path to your model

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Robot Fabric Defect Detector (Offline Mode)",
    page_icon="🧵",
    layout="wide"
)

# --- SESSION STATE INITIALIZATION ---
if 'ser' not in st.session_state:
    st.session_state.ser = None
if 'is_scanning' not in st.session_state:
    st.session_state.is_scanning = False
if 'scan_finished' not in st.session_state:
    st.session_state.scan_finished = False
if 'final_stats' not in st.session_state:
    st.session_state.final_stats = {}
if 'image_list' not in st.session_state:
    st.session_state.image_list = []

# --- HELPER: NATURAL SORTING ---
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def get_test_images():
    if not os.path.exists(TEST_IMAGES_FOLDER):
        return []
    files = [f for f in os.listdir(TEST_IMAGES_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return sorted(files, key=natural_sort_key)

# --- SIDEBAR SETTINGS ---
st.sidebar.title("Settings")

st.sidebar.subheader("Model Config")
model_source = st.sidebar.radio("Model Source", ["Default Path", "Upload Model"])
conf_threshold = st.sidebar.slider("Confidence", 0.0, 1.0, 0.25, 0.05)
iou_threshold = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.05)

st.sidebar.subheader("Robot Config")
serial_port = st.sidebar.text_input("COM Port", "COM5")
baud_rate = st.sidebar.selectbox("Baud Rate", [9600, 115200], index=0)

# --- NEW: REAL-TIME STATS PLACEHOLDER ---
st.sidebar.divider()
st.sidebar.subheader("Live Statistics")
sidebar_stats_placeholder = st.sidebar.empty()

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
        st.sidebar.success(f"Loaded: {MODEL_PATH}")
    except:
        st.sidebar.warning(f"Default model not found at {MODEL_PATH}")
else:
    uploaded_model = st.sidebar.file_uploader("Upload .pt file", type=["pt"])
    if uploaded_model:
        with open("temp_model.pt", "wb") as f:
            f.write(uploaded_model.getbuffer())
        model = load_model("temp_model.pt")

# --- ROBOT CONNECTION ---
def connect_arduino():
    try:
        if st.session_state.ser is not None:
            st.session_state.ser.close()
        st.session_state.ser = serial.Serial(serial_port, baud_rate, timeout=0.1)
        time.sleep(2)
        st.sidebar.success(f"Connected to {serial_port}")
        return True
    except Exception as e:
        st.sidebar.error(f"Connection Failed: {e}")
        return False

# --- MAIN UI ---
st.title("🧵 Fabric Defect Detection (Hybrid Mode)")
st.markdown(f"**Status:** {'✅ Connected' if st.session_state.ser else '❌ Disconnected'}")

ctrl_col1, ctrl_col2 = st.columns([1, 4])

# START BUTTON
with ctrl_col1:
    if st.button("▶️ START INSPECTION", type="primary"):
        # 1. Connect Robot
        if st.session_state.ser is None:
            if not connect_arduino():
                st.stop()
        
        # 2. Load Images from Folder
        images = get_test_images()
        if not images:
            st.error(f"No images found in folder '{TEST_IMAGES_FOLDER}'!")
            st.stop()
        
        st.session_state.image_list = images
        st.session_state.is_scanning = True
        st.session_state.scan_finished = False
        st.session_state.final_stats = {}
        
        st.toast(f"Loaded {len(images)} images for processing.")
        
        # 3. Send Start Signal to Robot
        try:
            st.session_state.ser.write(b's')
        except Exception as e:
            st.error(f"Serial Error: {e}")
            st.session_state.is_scanning = False

with ctrl_col2:
    if st.button("⏹️ STOP"):
        st.session_state.is_scanning = False


# --- PROCESSING LOOP ---
if st.session_state.is_scanning and model is not None:
    
    current_scan_stats = {}
    image_index = 0
    total_images = len(st.session_state.image_list)

    # Open Camera
    cap = cv2.VideoCapture(1) 
    
    st.divider()
    col_video, col_result = st.columns(2)
    with col_video:
        st.subheader("Robot Live View (416x416)")
        live_feed_placeholder = st.empty()
    with col_result:
        st.subheader(f"Inference (Image {image_index}/{total_images})")
        result_placeholder = st.empty()

    st.info("Robot running... Processing pre-saved images.")

    # Initialize sidebar stats at 0
    with sidebar_stats_placeholder.container():
        st.metric("Total Defects", 0)
        st.info("Waiting for data...")

    while st.session_state.is_scanning:
        
        # A. SERIAL COMMUNICATION
        if st.session_state.ser and st.session_state.ser.in_waiting > 0:
            try:
                line = st.session_state.ser.readline().decode('utf-8', errors='ignore').strip()
                if line: print(f"Arduino: {line}")

                if line == "READY_FOR_CAM":
                    # Robot is in position. LOAD THE NEXT FILE.
                    if image_index < total_images:
                        filename = st.session_state.image_list[image_index]
                        file_path = os.path.join(TEST_IMAGES_FOLDER, filename)
                        
                        try:
                            # 1. Load & Predict
                            pil_image = Image.open(file_path)
                            results = model.predict(pil_image, conf=conf_threshold, iou=iou_threshold, imgsz=416)
                            result = results[0]
                            
                            # 2. Update UI
                            res_plotted = result.plot()
                            res_plotted_rgb = res_plotted[:, :, ::-1]
                            result_placeholder.image(res_plotted_rgb, caption=f"Analyzed: {filename}", use_container_width=True)
                            
                            # 3. Stats Update
                            if len(result.boxes) > 0:
                                for box in result.boxes:
                                    cls_id = int(box.cls[0])
                                    class_name = result.names[cls_id]
                                    current_scan_stats[class_name] = current_scan_stats.get(class_name, 0) + 1
                            
                            # --- NEW: UPDATE SIDEBAR REAL-TIME ---
                            total_defects = sum(current_scan_stats.values())
                            with sidebar_stats_placeholder.container():
                                st.metric("Total Defects Found", total_defects)
                                if current_scan_stats:
                                    df_stats = pd.DataFrame(list(current_scan_stats.items()), columns=["Defect", "Count"])
                                    st.dataframe(df_stats, use_container_width=True, hide_index=True)
                                else:
                                    st.success("Fabric Clean So Far")
                            # -------------------------------------

                            image_index += 1
                            
                            # 4. Tell Robot to Move
                            time.sleep(0.1) 
                            st.session_state.ser.write(b'n')
                            
                        except Exception as e:
                            st.error(f"Error processing {filename}: {e}")
                            st.session_state.is_scanning = False
                    else:
                        st.warning("Ran out of images before robot finished!")
                        st.session_state.ser.write(b'n')

                elif "SCAN COMPLETE" in line:
                    st.success("Inspection Complete.")
                    st.session_state.is_scanning = False
                    st.session_state.scan_finished = True
                    st.session_state.final_stats = current_scan_stats
                    break
                    
            except Exception as e:
                print(f"Serial Error: {e}")

        # B. UPDATE LIVE CAMERA
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # --- NEW: RESIZE LIVE FEED TO 416x416 ---
                frame_resized = cv2.resize(frame, (416, 416))
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                live_feed_placeholder.image(frame_rgb, use_container_width=True)
        
        time.sleep(0.05)

    if cap.isOpened():
        cap.release()
    st.rerun()

# --- FINAL REPORT ---
if st.session_state.scan_finished:
    st.divider()
    st.header("Final Inspection Report")
    
    stats = st.session_state.final_stats
    
    if not stats:
        st.success("Fabric is clean! No defects detected.")
    else:
        st.subheader("Summary")
        cols = st.columns(len(stats))
        for i, (defect, count) in enumerate(stats.items()):
            cols[i].metric(label=defect, value=count)
        
        st.subheader("Defect Chart")
        df = pd.DataFrame(list(stats.items()), columns=["Defect Type", "Count"])
        df.set_index("Defect Type", inplace=True)
        st.bar_chart(df, color="#FF4B4B")

    if st.button("Start New Scan"):
        st.session_state.scan_finished = False
        st.rerun()