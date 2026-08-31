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
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Explainable Diabetic Retinopathy AI",
    page_icon="👁️",
    layout="wide"
)


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
# DR INFORMATION
# ============================================================

DR_INFORMATION = {

    0: {
        "stage": "No Diabetic Retinopathy",
        "description":
            "The model classified the submitted image as "
            "Grade 0 within the five-class diabetic retinopathy "
            "classification system.",
        "meaning":
            "No diabetic retinopathy category is predicted by "
            "the model for this image."
    },

    1: {
        "stage": "Mild Diabetic Retinopathy",
        "description":
            "The model classified the submitted image as "
            "Grade 1, corresponding to mild diabetic retinopathy "
            "within the classification system.",
        "meaning":
            "This represents an early category of diabetic "
            "retinopathy in the five-level grading system."
    },

    2: {
        "stage": "Moderate Diabetic Retinopathy",
        "description":
            "The model classified the submitted image as "
            "Grade 2, corresponding to moderate diabetic "
            "retinopathy.",
        "meaning":
            "This represents an intermediate category in the "
            "five-level grading system."
    },

    3: {
        "stage": "Severe Diabetic Retinopathy",
        "description":
            "The model classified the submitted image as "
            "Grade 3, corresponding to severe diabetic "
            "retinopathy.",
        "meaning":
            "This represents an advanced category in the "
            "five-level grading system."
    },

    4: {
        "stage": "Proliferative Diabetic Retinopathy",
        "description":
            "The model classified the submitted image as "
            "Grade 4, corresponding to proliferative diabetic "
            "retinopathy.",
        "meaning":
            "This is the highest category in the five-level "
            "classification system."
    }
}


# ============================================================
# HEADER
# ============================================================

st.title(
    "👁️ Explainable Diabetic Retinopathy AI"
)

st.write(
    "AI-assisted diabetic retinopathy grading with "
    "interpretable LIME explanations."
)

st.caption(
    "EfficientNet-B0 • Five-Class Classification • "
    "LIME Explainability"
)

st.warning(
    "Educational and research prototype only. "
    "The output is not a medical diagnosis and should not "
    "replace evaluation by a qualified healthcare professional."
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


# ============================================================
# LOAD MODEL SAFELY
# ============================================================

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

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_image(image):

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
# LIME EXPLANATION
# ============================================================

def generate_lime_explanation(
    image,
    predicted_class
):

    image_array = np.array(
        image.resize(
            (224, 224)
        )
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
# SECTION 1 — UPLOAD
# ============================================================

st.header(
    "1. Upload Retinal Fundus Image"
)

st.write(
    "Upload a JPG, JPEG, or PNG retinal fundus image "
    "for AI-assisted classification."
)

uploaded_file = st.file_uploader(
    "Choose a retinal fundus image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# IMAGE PROCESSING
# ============================================================

if uploaded_file is not None:

    # ========================================================
    # OPEN IMAGE
    # ========================================================

    try:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

    except Exception:

        st.error(
            "❌ Unable to read the uploaded image."
        )

        st.stop()


    # ========================================================
    # SECTION 2 — IMAGE
    # ========================================================

    st.header(
        "2. Uploaded Image"
    )

    col1, col2, col3 = st.columns(
        [1, 2, 1]
    )

    with col2:

        st.image(
            image,
            caption="Retinal Fundus Image",
            width="stretch"
        )


    # ========================================================
    # ANALYZE BUTTON
    # ========================================================

    st.write("")

    analyze = st.button(
        "🔍 Analyze Retinal Image",
        type="primary",
        width="stretch"
    )


    # ========================================================
    # ANALYSIS
    # ========================================================

    if analyze:

        # ====================================================
        # PREDICTION
        # ====================================================

        with st.spinner(
            "Analyzing retinal image..."
        ):

            predicted_class, confidence, probabilities = (
                predict_image(image)
            )


        # ====================================================
        # SECTION 3 — PREDICTION
        # ====================================================

        st.header(
            "3. AI Prediction"
        )


        prediction_col1, prediction_col2 = st.columns(
            2
        )


        with prediction_col1:

            st.metric(
                "Predicted DR Grade",
                f"Grade {predicted_class}"
            )

            st.write(
                f"**{CLASS_NAMES[predicted_class]}**"
            )


        with prediction_col2:

            st.metric(
                "Model Confidence",
                f"{confidence * 100:.2f}%"
            )


        # ====================================================
        # CONFIDENCE STATUS
        # ====================================================

        if confidence >= CONFIDENCE_THRESHOLD:

            st.success(
                f"Model confidence is above the configured "
                f"{CONFIDENCE_THRESHOLD * 100:.0f}% threshold."
            )

        else:

            st.warning(
                f"Low-confidence prediction: "
                f"{confidence * 100:.2f}% "
                f"(threshold: "
                f"{CONFIDENCE_THRESHOLD * 100:.0f}%)."
            )

            st.info(
                "The model's predicted class is shown for "
                "demonstration, but this result should be "
                "treated with additional caution."
            )


        # ====================================================
        # PREDICTED CLASS
        # ====================================================

        st.subheader(
            "Predicted Classification"
        )

        st.write(
            f"### Grade {predicted_class}"
        )

        st.write(
            f"**{CLASS_NAMES[predicted_class]}**"
        )


        # ====================================================
        # SECTION 4 — PROBABILITIES
        # ====================================================

        st.header(
            "4. Classification Probabilities"
        )

        st.write(
            "The following values represent the model's "
            "predicted probability for each classification."
        )

        for i, probability in enumerate(
            probabilities
        ):

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
        # SECTION 5 — LIME
        # ====================================================

        st.header(
            "5. Explainable AI — LIME"
        )

        st.write(
            "LIME provides a local explanation of the "
            "model's prediction by identifying image regions "
            "that influenced the predicted class."
        )


        # ====================================================
        # GENERATE LIME
        # ====================================================

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
        # ORIGINAL + LIME
        # ====================================================

        lime_col1, lime_col2 = st.columns(
            2
        )


        with lime_col1:

            st.subheader(
                "Original Fundus Image"
            )

            st.image(
                image.resize(
                    (224, 224)
                ),
                caption="Original Image",
                width="stretch"
            )


        with lime_col2:

            st.subheader(
                "LIME Explanation"
            )

            st.image(
                lime_visualization,
                caption="LIME Visualization",
                width="stretch"
            )


        st.success(
            "The highlighted regions represent areas "
            "identified by LIME as influential to the "
            "model's prediction."
        )


        # ====================================================
        # HOW LIME WORKS
        # ====================================================

        with st.expander(
            "🧩 How does LIME work?"
        ):

            st.write(
                "LIME stands for Local Interpretable "
                "Model-agnostic Explanations."
            )

            st.write(
                "For an individual image, LIME creates "
                "multiple modified versions of the image "
                "and observes how the model's predictions "
                "change."
            )

            st.write(
                "The image is divided into smaller regions "
                "called superpixels. LIME determines which "
                "regions have greater influence on the "
                "specific prediction."
            )

            st.write(
                "The explanation is local, meaning it "
                "describes the model's reasoning for this "
                "particular image rather than explaining "
                "the entire model."
            )


        # ====================================================
        # SECTION 6 — DR GRADING
        # ====================================================

        st.header(
            "6. Diabetic Retinopathy Grading"
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


        # ====================================================
        # SECTION 7 — RESULT INTERPRETATION
        # ====================================================

        st.header(
            "7. Result Interpretation"
        )

        selected_info = DR_INFORMATION[
            predicted_class
        ]


        st.subheader(
            selected_info["stage"]
        )

        st.write(
            selected_info["description"]
        )

        st.write(
            selected_info["meaning"]
        )


        # ====================================================
        # TECHNICAL SUMMARY
        # ====================================================

        st.header(
            "8. Technical Summary"
        )

        technical_col1, technical_col2 = st.columns(
            2
        )


        with technical_col1:

            st.write(
                "**Model**"
            )

            st.write(
                "EfficientNet-B0"
            )

            st.write(
                "**Input Size**"
            )

            st.write(
                "224 × 224 pixels"
            )

            st.write(
                "**Number of Classes**"
            )

            st.write(
                "5"
            )


        with technical_col2:

            st.write(
                "**Explainability Method**"
            )

            st.write(
                "LIME"
            )

            st.write(
                "**LIME Samples**"
            )

            st.write(
                "100"
            )

            st.write(
                "**LIME Features**"
            )

            st.write(
                "10 superpixels"
            )


        # ====================================================
        # FINAL DISCLAIMER
        # ====================================================

        st.header(
            "9. Important Disclaimer"
        )

        st.warning(
            "This application is an educational and research "
            "prototype. Its predictions and LIME explanations "
            "are generated by an AI model and should not be "
            "used as a medical diagnosis or as a substitute "
            "for evaluation by a qualified healthcare professional."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "👁️ Explainable Diabetic Retinopathy AI"
)

st.caption(
    "EfficientNet-B0 Classification • LIME Explainability"
)

st.caption(
    "Educational and research demonstration only • "
    "Not intended for medical diagnosis"
)