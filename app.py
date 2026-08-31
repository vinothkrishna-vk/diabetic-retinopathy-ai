import os
import numpy as np
import streamlit as st
import torch
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries
from torch import nn
from torchvision import models, transforms


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "dr_model_best.pth"
)

CONFIDENCE_THRESHOLD = 0.50

CLASS_NAMES = [
    "No Diabetic Retinopathy",
    "Mild Diabetic Retinopathy",
    "Moderate Diabetic Retinopathy",
    "Severe Diabetic Retinopathy",
    "Proliferative Diabetic Retinopathy"
]


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Explainable Diabetic Retinopathy AI",
    page_icon="👁️",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 40px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666666;
        margin-bottom: 20px;
    }

    .grade-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #dddddd;
        margin-top: 10px;
        margin-bottom: 10px;
    }

    .grade-number {
        font-size: 30px;
        font-weight: 700;
    }

    .grade-name {
        font-size: 20px;
        margin-top: 5px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">👁️ Explainable Diabetic Retinopathy AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-assisted diabetic retinopathy grading with LIME-based explainability'
    '</div>',
    unsafe_allow_html=True
)

st.warning(
    "⚠️ Prototype for educational/research demonstration only. "
    "This system is not a medical diagnosis."
)


# ============================================================
# LOAD MODEL
# ============================================================

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


# Load model safely
try:

    model, device = load_model()

except Exception as e:

    st.error(
        "❌ Failed to load the trained DR model."
    )

    st.exception(e)

    st.stop()


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ============================================================
# MODEL PREDICTION FUNCTION
# ============================================================

def predict_image(image):

    """
    Runs the EfficientNet-B0 model on a single image.
    Returns predicted class and probabilities.
    """

    input_tensor = transform(
        image
    ).unsqueeze(0).to(device)

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

    return (
        predicted_class,
        confidence,
        probabilities[0].cpu().numpy()
    )


# ============================================================
# LIME PREDICTION FUNCTION
# ============================================================

def predict_for_lime(images):

    """
    Prediction function required by LIME.

    LIME sends multiple modified versions of the image.
    This function converts those images into tensors and
    returns model probabilities.
    """

    batch = []

    for img in images:

        img = np.clip(
            img,
            0,
            1
        )

        img_pil = Image.fromarray(
            (
                img * 255
            ).astype(
                np.uint8
            )
        )

        tensor = transform(
            img_pil
        )

        batch.append(
            tensor
        )

    batch = torch.stack(
        batch
    ).to(device)

    with torch.no_grad():

        outputs = model(
            batch
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

    return probabilities.cpu().numpy()


# ============================================================
# LIME EXPLANATION FUNCTION
# ============================================================

def generate_lime_explanation(
    image,
    predicted_class
):

    """
    Generates a LIME explanation for the predicted DR class.
    """

    image_array = np.array(
        image.resize((224, 224))
    ).astype(
        np.float32
    ) / 255.0

    explainer = lime_image.LimeImageExplainer(
        verbose=False
    )

    explanation = explainer.explain_instance(
        image_array,
        predict_for_lime,
        top_labels=5,
        hide_color=0,
        num_samples=100
    )

    explanation_image, mask = (
        explanation.get_image_and_mask(
            predicted_class,
            positive_only=False,
            num_features=10,
            hide_rest=False
        )
    )

    explanation_image = np.clip(
        explanation_image,
        0,
        1
    )

    visualization = mark_boundaries(
        explanation_image,
        mask
    )

    return visualization


# ============================================================
# FILE UPLOADER
# ============================================================

st.subheader("📤 Upload Retinal Fundus Image")

uploaded_file = st.file_uploader(
    "Choose a JPG, JPEG or PNG retinal fundus image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# IMAGE ANALYSIS
# ============================================================

if uploaded_file is not None:

    try:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

    except Exception:

        st.error(
            "❌ Unable to read the uploaded image."
        )

        st.stop()


    # --------------------------------------------------------
    # DISPLAY UPLOADED IMAGE
    # --------------------------------------------------------

    st.subheader("🖼️ Uploaded Image")

    st.image(
        image,
        caption="Uploaded Fundus Image",
        width=600
    )


    # --------------------------------------------------------
    # ANALYZE BUTTON
    # --------------------------------------------------------

    analyze = st.button(
        "🔍 Analyze Image",
        type="primary",
        width="stretch"
    )


    if analyze:

        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        with st.spinner(
            "Analyzing retinal image..."
        ):

            predicted_class, confidence, probabilities = (
                predict_image(image)
            )


        # ====================================================
        # CONFIDENCE / UNCERTAINTY SAFEGUARD
        # ====================================================

        if confidence < CONFIDENCE_THRESHOLD:

            st.subheader(
                "⚠️ Uncertain Prediction"
            )

            st.warning(
                f"The model confidence is "
                f"**{confidence * 100:.2f}%**, which is below "
                f"the configured confidence threshold of "
                f"**{CONFIDENCE_THRESHOLD * 100:.0f}%**."
            )

            st.info(
                "Please upload a clear retinal fundus image "
                "for a more reliable analysis."
            )

            st.caption(
                "The system does not provide a DR grade when "
                "the model is insufficiently confident."
            )

            st.stop()


        # ====================================================
        # PREDICTION RESULTS
        # ====================================================

        st.subheader(
            "🧠 AI Prediction"
        )

        col1, col2 = st.columns(2)


        # ----------------------------------------------------
        # DR GRADE
        # ----------------------------------------------------

        with col1:

            st.metric(
                "DR Grade",
                f"Grade {predicted_class}"
            )

            st.write(
                f"**{CLASS_NAMES[predicted_class]}**"
            )


        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        with col2:

            st.metric(
                "Model Confidence",
                f"{confidence * 100:.2f}%"
            )


        # ----------------------------------------------------
        # RESULT MESSAGE
        # ----------------------------------------------------

        st.success(
            f"DR Grade {predicted_class}: "
            f"{CLASS_NAMES[predicted_class]}"
        )


        # ====================================================
        # CLASS PROBABILITIES
        # ====================================================

        st.subheader(
            "📊 Classification Probabilities"
        )

        for i, probability in enumerate(probabilities):

            st.write(
                f"**Grade {i} — {CLASS_NAMES[i]}**"
            )

            st.progress(
                float(probability)
            )

            st.caption(
                f"{probability * 100:.2f}%"
            )


        # ====================================================
        # LIME EXPLAINABILITY
        # ====================================================

        st.subheader(
            "🔍 Explainable AI — LIME"
        )

        st.write(
            "LIME identifies image regions that influenced "
            "the model's prediction."
        )


        # ----------------------------------------------------
        # Generate LIME
        # ----------------------------------------------------

        with st.spinner(
            "Generating LIME explanation..."
        ):

            try:

                lime_visualization = (
                    generate_lime_explanation(
                        image,
                        predicted_class
                    )
                )

            except Exception as e:

                st.error(
                    "❌ Unable to generate the LIME explanation."
                )

                st.exception(e)

                st.stop()


        # ====================================================
        # DISPLAY ORIGINAL AND LIME
        # ====================================================

        col1, col2 = st.columns(2)


        with col1:

            st.image(
                image.resize((224, 224)),
                caption="Original Fundus Image",
                width="stretch"
            )


        with col2:

            st.image(
                lime_visualization,
                caption="LIME Explanation",
                width="stretch"
            )


        # ====================================================
        # LIME EXPLANATION
        # ====================================================

        st.success(
            "LIME highlights regions that influenced "
            "the model's prediction."
        )


        with st.expander(
            "🧩 How does LIME work?"
        ):

            st.write(
                "LIME stands for Local Interpretable "
                "Model-agnostic Explanations."
            )

            st.write(
                "LIME explains an individual prediction by "
                "creating modified versions of the input image "
                "and observing how the model's prediction changes."
            )

            st.write(
                "The image is divided into smaller meaningful "
                "regions called superpixels. LIME identifies "
                "regions that have greater influence on the "
                "predicted DR class."
            )


        # ====================================================
        # DR GRADING INFORMATION
        # ====================================================

        st.subheader(
            "📚 Diabetic Retinopathy Grading"
        )

        grading_data = [
            (
                "Grade 0",
                "No Diabetic Retinopathy"
            ),
            (
                "Grade 1",
                "Mild Diabetic Retinopathy"
            ),
            (
                "Grade 2",
                "Moderate Diabetic Retinopathy"
            ),
            (
                "Grade 3",
                "Severe Diabetic Retinopathy"
            ),
            (
                "Grade 4",
                "Proliferative Diabetic Retinopathy"
            )
        ]

        for grade, description in grading_data:

            st.write(
                f"**{grade}** — {description}"
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "AI-assisted diabetic retinopathy screening prototype | "
    "EfficientNet-B0 + LIME"
)

st.caption(
    "For educational and research demonstration only. "
    "Not intended for medical diagnosis."
)
