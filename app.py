import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

# 1. Page Configuration and Setup
st.set_page_config(
    page_title="Demographics AI - Age & Gender Detection",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Base directory for absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Premium Styling using Custom CSS
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Outfit:wght@500;700&display=swap" rel="stylesheet">
<style>
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styling */
    .main-title {
        font-family: 'Outfit', sans-serif;
        background: linear-gradient(135deg, #00FF7F, #00BFFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        font-family: 'Inter', sans-serif;
        color: #9CA3AF;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Custom Sidebar styling */
    .sidebar-header {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 1.3rem;
        color: #E5E7EB;
        margin-bottom: 1rem;
        border-bottom: 2px solid #00FF7F;
        padding-bottom: 0.3rem;
    }
    
    /* Card Container for Analysis */
    .analysis-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    
    .card-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        font-size: 1.25rem;
        color: #00FF7F;
        margin-bottom: 0.5rem;
    }
    
    .metric-text {
        font-size: 1.05rem;
        color: #F3F4F6;
        margin-bottom: 0.25rem;
    }
    
    .metric-value {
        font-weight: 600;
        color: #00BFFF;
    }
</style>
""", unsafe_allow_html=True)

# 3. Model Loading with Cache
@st.cache_resource
def load_models():
    """Loads and caches the face detector, age model, and gender model."""
    face_proto = os.path.join(BASE_DIR, "opencv_face_detector.pbtxt")
    face_weights = os.path.join(BASE_DIR, "opencv_face_detector_uint8.pb")
    age_proto = os.path.join(BASE_DIR, "age_deploy.prototxt")
    age_weights = os.path.join(BASE_DIR, "age_net.caffemodel")
    gender_proto = os.path.join(BASE_DIR, "gender_deploy.prototxt")
    gender_weights = os.path.join(BASE_DIR, "gender_net.caffemodel")

    # Load DNN networks
    face_net = cv2.dnn.readNet(face_weights, face_proto)
    age_net = cv2.dnn.readNet(age_weights, age_proto)
    gender_net = cv2.dnn.readNet(gender_weights, gender_proto)

    return face_net, age_net, gender_net

# Model constant mean values and target sizes
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_GROUPS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_CLASSES = ['Male', 'Female']

# 4. Processing Functions
def detect_and_predict(image_bgr, conf_threshold=0.7, padding=15):
    """Detects faces in BGR image and predicts age and gender for each."""
    face_net, age_net, gender_net = load_models()
    
    fr_cv = image_bgr.copy()
    fr_h, fr_w = fr_cv.shape[:2]
    
    # 1. Prepare blob for face detection
    blob = cv2.dnn.blobFromImage(fr_cv, 1.0, (300, 300), [104, 117, 123], True, False)
    face_net.setInput(blob)
    detections = face_net.forward()
    
    results = []
    
    # 2. Iterate through detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            # Face bounding box coords
            x1 = int(detections[0, 0, i, 3] * fr_w)
            y1 = int(detections[0, 0, i, 4] * fr_h)
            x2 = int(detections[0, 0, i, 5] * fr_w)
            y2 = int(detections[0, 0, i, 6] * fr_h)
            
            # Boundary clamps
            x1_clamp = max(0, x1)
            y1_clamp = max(0, y1)
            x2_clamp = min(fr_w - 1, x2)
            y2_clamp = min(fr_h - 1, y2)
            
            # Crop box with padding
            face_y1 = max(0, y1_clamp - padding)
            face_y2 = min(fr_h - 1, y2_clamp + padding)
            face_x1 = max(0, x1_clamp - padding)
            face_x2 = min(fr_w - 1, x2_clamp + padding)
            
            if face_y2 > face_y1 and face_x2 > face_x1:
                # Crop original face image without box overlays
                face_img = image_bgr[face_y1:face_y2, face_x1:face_x2]
                
                # Blob for classifier (227x227 image size)
                face_blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                
                # Gender estimation
                gender_net.setInput(face_blob)
                gender_preds = gender_net.forward()
                # Apply Softmax for probabilities
                gender_probs_exp = np.exp(gender_preds[0] - np.max(gender_preds[0]))
                gender_probs = gender_probs_exp / np.sum(gender_probs_exp)
                gender_idx = gender_probs.argmax()
                gender_label = GENDER_CLASSES[gender_idx]
                
                # Age estimation
                age_net.setInput(face_blob)
                age_preds = age_net.forward()
                # Apply Softmax
                age_probs_exp = np.exp(age_preds[0] - np.max(age_preds[0]))
                age_probs = age_probs_exp / np.sum(age_probs_exp)
                age_idx = age_probs.argmax()
                age_label = AGE_GROUPS[age_idx]
                
                results.append({
                    "box": (x1_clamp, y1_clamp, x2_clamp, y2_clamp),
                    "face_img": face_img,
                    "gender": gender_label,
                    "gender_conf": gender_probs[gender_idx],
                    "gender_probs": gender_probs.tolist(),
                    "age": age_label,
                    "age_conf": age_probs[age_idx],
                    "age_probs": age_probs.tolist(),
                    "face_conf": confidence
                })
                
    return results

def draw_annotations(image_bgr, results):
    """Draws custom neon green boxes and overlay tags on the image."""
    annotated = image_bgr.copy()
    h, w = annotated.shape[:2]
    
    # Standardize drawing parameters dynamically based on image size
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = max(0.5, min(w, h) / 900.0)
    thickness = max(1, int(min(w, h) / 400))
    box_thickness = max(2, int(min(w, h) / 250))
    
    for i, res in enumerate(results):
        x1, y1, x2, y2 = res["box"]
        # Draw neon green box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (57, 255, 20), box_thickness, cv2.LINE_AA)
        
        # Tag text: e.g. "#1 Female, (25-32)"
        tag = f"#{i+1} {res['gender']}, {res['age']}"
        (t_w, t_h), baseline = cv2.getTextSize(tag, font, font_scale, thickness)
        
        # Determine tag position (above box, or inside if too close to top)
        t_y = max(y1, t_h + 10)
        
        # Draw semi-transparent background box for readability
        cv2.rectangle(
            annotated,
            (x1, t_y - t_h - 10),
            (x1 + t_w + 10, t_y + baseline - 5),
            (57, 255, 20),
            cv2.FILLED
        )
        
        # Draw tag text in black
        cv2.putText(
            annotated,
            tag,
            (x1 + 5, t_y - 5),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA
        )
        
    return annotated

# 5. UI Plotting Functions
def render_plotly_charts(gender_probs, age_probs):
    """Returns Plotly figures for Gender and Age distribution."""
    # Gender Chart
    fig_gen = go.Figure(go.Bar(
        x=[p * 100 for p in gender_probs],
        y=GENDER_CLASSES,
        orientation='h',
        marker=dict(
            color=['#4F46E5', '#EC4899'], # Slate Blue & Pink Rose
            line=dict(color='rgba(255,255,255,0.15)', width=1)
        ),
        text=[f"{p*100:.1f}%" for p in gender_probs],
        textposition='outside',
        cliponaxis=False
    ))
    
    fig_gen.update_layout(
        title=dict(text="Gender Estimate Confidence", font=dict(size=12, color='#E5E7EB')),
        height=140,
        margin=dict(l=65, r=50, t=30, b=15),
        xaxis=dict(range=[0, 115], showgrid=True, gridcolor='rgba(255,255,255,0.08)', ticksuffix="%"),
        yaxis=dict(showgrid=False),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#9CA3AF', size=11),
    )
    
    # Age Chart
    fig_age = go.Figure(go.Bar(
        x=[p * 100 for p in age_probs],
        y=AGE_GROUPS,
        orientation='h',
        marker=dict(
            color='#06B6D4', # Deep Cyan
            line=dict(color='rgba(255,255,255,0.15)', width=1)
        ),
        text=[f"{p*100:.1f}%" if p > 0.03 else "" for p in age_probs],
        textposition='outside',
        cliponaxis=False
    ))
    
    fig_age.update_layout(
        title=dict(text="Age Bracket Distribution", font=dict(size=12, color='#E5E7EB')),
        height=260,
        margin=dict(l=65, r=50, t=30, b=15),
        xaxis=dict(range=[0, 115], showgrid=True, gridcolor='rgba(255,255,255,0.08)', ticksuffix="%"),
        yaxis=dict(showgrid=False),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#9CA3AF', size=11),
    )
    
    return fig_gen, fig_age


# 6. Main Dashboard Layout
def main():
    st.markdown('<div class="main-title">Demographics AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Multi-Face Age & Gender Estimation using OpenCV DNN Caffe Models</div>', unsafe_allow_html=True)
    
    # Sidebar Configuration Section
    st.sidebar.markdown('<div class="sidebar-header">⚙️ Configuration</div>', unsafe_allow_html=True)
    
    conf_threshold = st.sidebar.slider(
        "Face Detection Confidence",
        min_value=0.3, max_value=0.99, value=0.7, step=0.05,
        help="Higher thresholds lower false positives but may miss faces."
    )
    
    crop_padding = st.sidebar.slider(
        "Face Crop Padding (px)",
        min_value=0, max_value=50, value=15, step=5,
        help="Adjusts the surrounding margin included when classifying a cropped face."
    )
    
    st.sidebar.markdown('<div class="sidebar-header">📤 Input Source</div>', unsafe_allow_html=True)
    input_source = st.sidebar.selectbox(
        "Choose how to supply an image",
        ["Use Sample Images", "Upload Image File", "Use Live Camera Feed"]
    )
    
    # Load and resolve input image
    image_to_process = None
    
    if input_source == "Use Sample Images":
        sample_choice = st.sidebar.radio(
            "Select Sample",
            ["Sample 1: Young Woman", "Sample 2: Elderly Man"]
        )
        if sample_choice == "Sample 1: Young Woman":
            filename = os.path.join(BASE_DIR, "sample_woman.png")
        else:
            filename = os.path.join(BASE_DIR, "sample_man.png")
            
        if os.path.exists(filename):
            pil_img = Image.open(filename)
            image_to_process = np.array(pil_img)
            # If color channels are RGBA, convert to RGB
            if image_to_process.shape[-1] == 4:
                image_to_process = cv2.cvtColor(image_to_process, cv2.COLOR_RGBA2RGB)
            # Convert RGB to BGR for OpenCV
            image_to_process = cv2.cvtColor(image_to_process, cv2.COLOR_RGB2BGR)
        else:
            st.error(f"Sample file {filename} could not be loaded.")
            
    elif input_source == "Upload Image File":
        uploaded_file = st.sidebar.file_uploader(
            "Choose a PNG, JPG, or JPEG file...",
            type=["png", "jpg", "jpeg"]
        )
        if uploaded_file is not None:
            pil_img = Image.open(uploaded_file)
            image_to_process = np.array(pil_img)
            if len(image_to_process.shape) == 2: # Gray scale
                image_to_process = cv2.cvtColor(image_to_process, cv2.COLOR_GRAY2BGR)
            elif image_to_process.shape[-1] == 4: # RGBA
                image_to_process = cv2.cvtColor(image_to_process, cv2.COLOR_RGBA2BGR)
            else: # RGB
                image_to_process = cv2.cvtColor(image_to_process, cv2.COLOR_RGB2BGR)
                
    elif input_source == "Use Live Camera Feed":
        camera_img = st.camera_input("Take a photo to analyze")
        if camera_img is not None:
            pil_img = Image.open(camera_img)
            image_to_process = np.array(pil_img)
            image_to_process = cv2.cvtColor(image_to_process, cv2.COLOR_RGB2BGR)

    # 7. Processing & View rendering
    if image_to_process is not None:
        # Resize to reasonable dimensions for display & inference if extremely large
        h, w = image_to_process.shape[:2]
        if w > 1200 or h > 1200:
            scale = 1200 / max(w, h)
            image_to_process = cv2.resize(image_to_process, (int(w * scale), int(h * scale)))
            
        results = detect_and_predict(image_to_process, conf_threshold, crop_padding)
        
        # Dual-Column Layout
        col_main, col_analysis = st.columns([5, 4])
        
        with col_main:
            st.subheader("📷 Detections Overview")
            annotated_bgr = draw_annotations(image_to_process, results)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, use_container_width=True)
            
            # Quick summary stats
            st.markdown(f"**Total Faces Detected:** `{len(results)}`")
            
        with col_analysis:
            st.subheader("🔍 Demographic Analysis")
            
            if len(results) == 0:
                st.info("No faces detected in the current image. Try adjusting the confidence threshold slider in the sidebar.")
            else:
                for idx, res in enumerate(results):
                    st.markdown(f'<div class="analysis-card">', unsafe_allow_html=True)
                    
                    # Row with index + crop and primary stats
                    card_col_1, card_col_2 = st.columns([1, 2])
                    with card_col_1:
                        # Display Cropped Face (RGB)
                        face_rgb = cv2.cvtColor(res["face_img"], cv2.COLOR_BGR2RGB)
                        st.image(face_rgb, width=120)
                    with card_col_2:
                        st.markdown(f'<div class="card-title">Face #{idx+1}</div>', unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="metric-text">Predicted Gender: <span class="metric-value">{res["gender"]}</span> '
                            f'({res["gender_conf"]*100:.1f}%)</div>', 
                            unsafe_allow_html=True
                        )
                        st.markdown(
                            f'<div class="metric-text">Predicted Age: <span class="metric-value">{res["age"]}</span> '
                            f'({res["age_conf"]*100:.1f}%)</div>', 
                            unsafe_allow_html=True
                        )
                        st.markdown(
                            f'<div class="metric-text">Face Confidence: <span class="metric-value">{res["face_conf"]*100:.1f}%)</span></div>', 
                            unsafe_allow_html=True
                        )
                    
                    # Bar charts of probabilities
                    fig_gen, fig_age = render_plotly_charts(res["gender_probs"], res["age_probs"])
                    st.plotly_chart(fig_gen, use_container_width=True, config={'displayModeBar': False})
                    st.plotly_chart(fig_age, use_container_width=True, config={'displayModeBar': False})
                    
                    st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Prompt state
        st.info("👈 Please upload an image, activate your live camera feed, or use the sample images option in the sidebar to begin analysis.")

if __name__ == "__main__":
    main()
