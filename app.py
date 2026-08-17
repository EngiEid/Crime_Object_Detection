import streamlit as st
import pandas as pd
from PIL import Image
from main import ObjectDetector


MODEL_PATH = "best.pt"

st.set_page_config(
    page_title="Object Detection",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 YOLO Object Detection")
st.write("Upload an image and the trained YOLO model will detect the objects automatically.")

@st.cache_resource
def load_detector():
    return ObjectDetector(MODEL_PATH)


try:
    detector = load_detector()
except FileNotFoundError as e:
    st.error(str(e))
    st.info(
        "Make sure your trained model exists at "
        "`runs/detect/train/weights/best.pt`, or change MODEL_PATH in app.py."
    )
    st.stop()

st.sidebar.header("⚙️ Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.25,
    step=0.05,
)

st.sidebar.markdown("---")
st.sidebar.write("**Model:** YOLO11")
st.sidebar.write(f"**Weights:** `{MODEL_PATH}`")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
)

camera_image = st.camera_input("Or take a picture")

image_source = uploaded_file if uploaded_file is not None else camera_image


if image_source is not None:
    image = Image.open(image_source).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with st.spinner("Detecting objects..."):
        annotated_image, detections = detector.predict(
            image,
            conf=confidence,
        )

    with col2:
        st.subheader("Detection Result")
        st.image(annotated_image, use_container_width=True)

    st.subheader("📊 Detection Results")

    if detections:
        df = pd.DataFrame(detections)
        bbox_df = pd.DataFrame(
            df["bbox"].tolist(),
            columns=["x1", "y1", "x2", "y2"],
        )

        result_df = pd.concat(
            [
                df[["class_name", "confidence"]],
                bbox_df,
            ],
            axis=1,
        )

        result_df["confidence"] = (
            result_df["confidence"] * 100
        ).round(2).astype(str) + "%"

        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True,
        )

        st.metric("Objects Detected", len(detections))

        class_counts = (
            df["class_name"]
            .value_counts()
            .rename_axis("Class")
            .reset_index(name="Count")
        )

        st.subheader("📌 Objects by Class")
        st.bar_chart(
            class_counts.set_index("Class")
        )

    else:
        st.warning(
            "No objects were detected. "
            "Try lowering the confidence threshold."
        )

else:
    st.info("👆 Upload an image or use the camera to start detection.")
