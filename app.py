import streamlit as st
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np

# =========================
# Configuration
# =========================

import os

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dr_model_best.pth")

CLASS_NAMES = [
    "No Diabetic Retinopathy",
    "Mild Diabetic Retinopathy",
    "Moderate Diabetic Retinopathy",
    "Severe Diabetic Retinopathy",
    "Proliferative Diabetic Retinopathy"
]

# =========================
# Page configuration
# =========================

st.set_page_config(
    page_title="DR Explainable AI",
    page_icon="👁️",
    layout="wide"
)

# =========================
# Title
# =========================

st.title("👁️ Explainable Diabetic Retinopathy AI")

st.write(
    "Upload a retinal fundus image to obtain an AI-based "
    "DR grade prediction and Grad-CAM visual explanation."
)

st.warning(
    "Prototype for educational/research demonstration only. "
    "This result is not a medical diagnosis."
)

# =========================
# Load model
# =========================

@st.cache_resource
def load_model():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = models.efficientnet_b0(
        weights=None
    )

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        5
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model = model.to(device)

    model.eval()

    return model, device


model, device = load_model()

# =========================
# Image transformation
# =========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# =========================
# Upload image
# =========================

uploaded_file = st.file_uploader(
    "📤 Upload retinal fundus image",
    type=["jpg", "jpeg", "png"]
)

# =========================
# Analysis
# =========================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.subheader("Uploaded Image")

    st.image(
        image,
        caption="Retinal Fundus Image",
        width=600
    )

    if st.button(
        "🔍 Analyze Image",
        type="primary"
    ):

        with st.spinner(
            "Analyzing retinal image..."
        ):

            # Prepare image
            input_tensor = transform(
                image
            ).unsqueeze(0).to(device)

            # Prediction
            with torch.no_grad():

                outputs = model(
                    input_tensor
                )

                probabilities = torch.softmax(
                    outputs,
                    dim=1
                )

                predicted_class = torch.argmax(
                    probabilities,
                    dim=1
                ).item()

                confidence = probabilities[
                    0,
                    predicted_class
                ].item()

            # =========================
            # Display prediction
            # =========================

            st.subheader(
                "🧠 AI Prediction"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Predicted DR Grade",
                    str(predicted_class)
                )

            with col2:

                st.metric(
                    "Confidence",
                    f"{confidence * 100:.2f}%"
                )

            st.info(
                f"Prediction: "
                f"**{CLASS_NAMES[predicted_class]}**"
            )

            # =========================
            # Grad-CAM
            # =========================

            rgb_image = np.array(
                image.resize((224, 224))
            ).astype(
                np.float32
            ) / 255.0

            target_layers = [
                model.features[-1]
            ]

            targets = [
                ClassifierOutputTarget(
                    predicted_class
                )
            ]

            with GradCAM(
                model=model,
                target_layers=target_layers
            ) as cam:

                grayscale_cam = cam(
                    input_tensor=input_tensor,
                    targets=targets
                )

                grayscale_cam = (
                    grayscale_cam[0]
                )

            visualization = show_cam_on_image(
                rgb_image,
                grayscale_cam,
                use_rgb=True
            )

            # =========================
            # Display Grad-CAM
            # =========================

            st.subheader(
                "🔥 Explainable AI — Grad-CAM"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.image(
                    image.resize((224, 224)),
                    caption="Original Image"
                )

            with col2:

                st.image(
                    visualization,
                    caption="Grad-CAM Heatmap"
                )

            st.success(
                "The highlighted regions represent "
                "areas that contributed to the model's prediction."
            )

# =========================
# Footer
# =========================

st.markdown("---")

st.caption(
    "AI-assisted diabetic retinopathy screening prototype | "
    "EfficientNet-B0 + Grad-CAM"
)