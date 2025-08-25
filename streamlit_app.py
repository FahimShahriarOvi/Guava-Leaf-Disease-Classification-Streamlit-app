import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import io
import zipfile
import os
import json
import timm
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from lime import lime_image
from skimage.segmentation import mark_boundaries
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM, AblationCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import logging
import warnings
import threading
import gc
import hashlib
from scipy import ndimage
from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_session_state():
    if "processing_id" not in st.session_state:
        st.session_state.processing_id = None
    if "current_image_hash" not in st.session_state:
        st.session_state.current_image_hash = None
    if "stop_processing" not in st.session_state:
        st.session_state.stop_processing = threading.Event()
    if "processing_lock" not in st.session_state:
        st.session_state.processing_lock = threading.Lock()


def get_image_hash(image_pil, model_name, methods):
    img_bytes = io.BytesIO()
    image_pil.save(img_bytes, format="PNG")
    img_data = img_bytes.getvalue()

    hash_input = img_data + model_name.encode() + str(sorted(methods)).encode()
    return hashlib.md5(hash_input).hexdigest()


def clear_processing_memory():
    plt.close("all")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def check_should_stop():
    return st.session_state.stop_processing.is_set()


CLASS_NAMES = ["Anthracnose", "Canker", "Dot", "Healthy", "Rust"]
NUM_CLASSES = len(CLASS_NAMES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_CONFIGS = {
    "Custom CNN": {
        "repo_id": "fahimShahriarOvi/CSE366_Project_Weights",
        "filename": "custom_cnn_best.pth",
        "architecture": "Custom Convolutional Neural Network",
        "input_size": (224, 224),
        "classes": NUM_CLASSES,
        "description": "5-layer CNN with BatchNorm and Dropout",
    },
    "ConvNeXt-Tiny": {
        "repo_id": "fahimShahriarOvi/CSE366_Project_Weights",
        "filename": "convnext_tiny_best.pth",
        "architecture": "ConvNeXt-Tiny",
        "input_size": (224, 224),
        "classes": NUM_CLASSES,
        "description": "Modern CNN inspired by Vision Transformers",
    },
    "DenseNet-121": {
        "repo_id": "fahimShahriarOvi/CSE366_Project_Weights",
        "filename": "densenet121_best.pth",
        "architecture": "DenseNet-121",
        "input_size": (224, 224),
        "classes": NUM_CLASSES,
        "description": "Densely connected convolutional network",
    },
    "EfficientNet-B0": {
        "repo_id": "fahimShahriarOvi/CSE366_Project_Weights",
        "filename": "efficientnet_b0_best.pth",
        "architecture": "EfficientNet-B0",
        "input_size": (224, 224),
        "classes": NUM_CLASSES,
        "description": "Compound scaling method for CNN efficiency",
    },
    "Inception-V3": {
        "repo_id": "fahimShahriarOvi/CSE366_Project_Weights",
        "filename": "inception_v3_best.pth",
        "architecture": "Inception-V3",
        "input_size": (299, 299),
        "classes": NUM_CLASSES,
        "description": "Multi-scale convolutional architecture",
    },
    "ViT-Base-16": {
        "repo_id": "fahimShahriarOvi/CSE366_Project_Weights",
        "filename": "vit_base_(b_16)_best.pth",
        "architecture": "Vision Transformer Base Patch 16",
        "input_size": (224, 224),
        "classes": NUM_CLASSES,
        "description": "Vision Transformer with 16x16 patches, attention-based architecture",
    },
}

SAMPLE_IMAGES = {
    "Anthracnose Sample": "../sample_images/Anthracnose.jpg",
    "Canker Sample": "../sample_images/Canker.jpg",
    "Healthy Leaf Sample": "../sample_images/Healthy.jpg",
    "Dot Disease Sample": "../sample_images/Dot.jpg",
    "Rust Sample": "../sample_images/Rust.jpg",
}


class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


@st.cache_data
def download_model_from_hub(repo_id, filename):
    try:
        with st.spinner(f"Downloading {filename} from Hugging Face Hub..."):
            model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        logger.info(f"Successfully downloaded {filename} from {repo_id}")
        return model_path
    except Exception as e:
        logger.error(f"Error downloading {filename} from {repo_id}: {str(e)}")
        raise e


def load_model(model_name, model_path, num_classes, device):
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=device)
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        if model_name == "Custom CNN":
            model = CustomCNN(num_classes).to(device)

        elif model_name == "ConvNeXt-Tiny":
            model = timm.create_model(
                "convnext_tiny", pretrained=False, num_classes=num_classes
            )
            model = model.to(device)

        elif model_name == "DenseNet-121":
            model = models.densenet121(weights=None)
            classifier_key = None
            input_features = None

            for key in state_dict.keys():
                if "classifier" in key and "weight" in key:
                    classifier_key = key
                    weight_shape = state_dict[key].shape
                    input_features = weight_shape[1]
                    break

            if input_features is None:
                input_features = model.classifier.in_features

            if "classifier.weight" in state_dict:
                model.classifier = nn.Linear(input_features, num_classes)
            elif "classifier.1.weight" in state_dict:
                model.classifier = nn.Sequential(
                    nn.Dropout(0.5), nn.Linear(input_features, num_classes)
                )
            else:
                model.classifier = nn.Linear(input_features, num_classes)

            model = model.to(device)

        elif model_name == "EfficientNet-B0":
            model = models.efficientnet_b0(weights=None)
            if "classifier.1.weight" in state_dict:
                model.classifier[1] = nn.Linear(
                    model.classifier[1].in_features, num_classes
                )
            elif "classifier.1.1.weight" in state_dict:
                model.classifier = nn.Sequential(
                    model.classifier[0],
                    nn.Sequential(
                        nn.Dropout(0.2),
                        nn.Linear(1280, num_classes),
                    ),
                )
            model = model.to(device)

        elif model_name == "Inception-V3":
            model = models.inception_v3(weights=None, aux_logits=True)
            model.fc = nn.Sequential(
                nn.Dropout(0.5), nn.Linear(model.fc.in_features, num_classes)
            )

            if hasattr(model, "AuxLogits"):
                model.AuxLogits.fc = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(model.AuxLogits.fc.in_features, num_classes),
                )

            model = model.to(device)

        elif model_name == "ViT-Base-16":
            model = timm.create_model(
                "vit_base_patch16_224", pretrained=False, num_classes=num_classes
            )
            if "head.1.weight" in state_dict:
                model.head = nn.Sequential(
                    nn.Dropout(0.5), nn.Linear(model.head.in_features, num_classes)
                )
            model = model.to(device)

        else:
            raise ValueError(f"Unknown model: {model_name}")

        model.load_state_dict(state_dict)
        model.eval()
        logger.info(f"Successfully loaded {model_name}")
        return model

    except Exception as e:
        logger.error(f"Error loading {model_name}: {str(e)}")
        raise e


def get_target_layers(model, model_name):
    try:
        if model_name == "Custom CNN":
            return [model.features[15]]

        elif model_name == "ConvNeXt-Tiny":
            if hasattr(model, "stages"):
                return [model.stages[-1].blocks[-1].conv_dw]
            else:
                return [model.norm]

        elif model_name == "DenseNet-121":
            return [model.features.norm5]

        elif model_name == "EfficientNet-B0":
            return [model.features[8]]

        elif model_name == "Inception-V3":
            if hasattr(model, "Mixed_7c"):
                return [model.Mixed_7c]
            else:
                return [model.avgpool]

        elif model_name == "ViT-Base-16":
            if hasattr(model, "blocks"):
                return [model.blocks[-1].norm1]
            else:
                return [model.norm]

        else:
            for name, module in reversed(list(model.named_modules())):
                if isinstance(module, torch.nn.Conv2d):
                    return [module]
            raise ValueError(f"Could not find target layer for {model_name}")

    except Exception as e:
        logger.error(f"Error getting target layers for {model_name}: {str(e)}")
        return [model]


def get_transform(model_name):
    img_size = MODEL_CONFIGS[model_name]["input_size"]

    return transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def check_model_availability():
    available_models = {}
    missing_models = {}

    for model_name, config in MODEL_CONFIGS.items():
        try:
            from huggingface_hub import repo_exists

            if repo_exists(config["repo_id"]):
                available_models[model_name] = config
            else:
                missing_models[model_name] = config
        except Exception as e:
            available_models[model_name] = config
            logger.warning(f"Could not verify {model_name} availability: {str(e)}")

    return available_models, missing_models


st.set_page_config(
    page_title="Guava Leaf Disease Classifier",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .model-info {
        background-color: #2E8B57;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #2E8B57;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #2E8B57;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
</style>
""",
    unsafe_allow_html=True,
)


st.markdown(
    '<h1 class="main-header"> Guava Leaf Disease Classifier with XAI</h1>',
    unsafe_allow_html=True,
)


st.sidebar.header("Control Panel")

available_models, missing_models = check_model_availability()

st.sidebar.subheader("Model Selection")

if not available_models:
    st.sidebar.error("No model weights found! Please check the weights directory.")
    st.stop()

if missing_models:
    with st.sidebar.expander("Missing Models", expanded=False):
        for name, config in missing_models.items():
            st.write(f"**{name}**: `{config['path']}`")

selected_model = st.sidebar.selectbox(
    "Choose Model:",
    list(available_models.keys()),
    help="Select a pre-trained model for classification",
)

if selected_model:
    config = available_models[selected_model]
    st.sidebar.markdown(
        f"""
    <div class="model-info">
    <h4>Model Information</h4>
    <p><strong>Architecture:</strong> {config['architecture']}</p>
    <p><strong>Input Size:</strong> {config['input_size']}</p>
    <p><strong>Classes:</strong> {config['classes']}</p>
    <p><strong>Description:</strong> {config['description']}</p>
    <p><strong>Repository:</strong> <code>{config['repo_id']}</code></p>
    <p><strong>File:</strong> <code>{config['filename']}</code></p>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.sidebar.subheader("Image Input")

uploaded_file = st.sidebar.file_uploader(
    "Upload Image:",
    type=["jpg", "jpeg", "png"],
    help="Upload a leaf image for disease classification",
)

st.sidebar.write("**Or choose a sample:**")
sample_choice = st.sidebar.selectbox(
    "Sample Images:",
    ["None"] + list(SAMPLE_IMAGES.keys()),
    help="Select from bundled sample images",
)

st.sidebar.subheader("Explanation Methods")

if selected_model == "ViT-Base-16":
    available_methods = ["Attention Visualization", "LIME"]
    default_methods = ["Attention Visualization", "LIME"]
    method_help = (
        "ViT uses attention mechanisms - CAM methods work differently for transformers"
    )
else:
    available_methods = ["Grad-CAM", "Grad-CAM++", "Eigen-CAM", "Ablation-CAM", "LIME"]
    default_methods = ["Grad-CAM", "LIME"]
    method_help = "Choose explanation methods to visualize model decisions"

selected_methods = st.sidebar.multiselect(
    "Select XAI Methods:",
    available_methods,
    default=default_methods,
    help=method_help,
)

with st.sidebar.expander("Advanced Options"):
    show_heatmap_only = st.checkbox("Show heatmap only (no overlay)", value=False)
    explanation_alpha = st.slider("Overlay transparency", 0.0, 1.0, 0.4, 0.1)
    colormap = st.selectbox(
        "Heatmap colormap", ["jet", "hot", "viridis", "plasma"], index=0
    )

st.sidebar.markdown("---")
st.sidebar.info(f"Device: {DEVICE}")
if torch.cuda.is_available():
    st.sidebar.success("CUDA Available")
else:
    st.sidebar.warning("CPU Only")


@st.cache_resource
def load_selected_model(model_name):
    config = available_models[model_name]
    model_path = download_model_from_hub(config["repo_id"], config["filename"])
    return load_model(model_name, model_path, NUM_CLASSES, DEVICE)


def predict_image(image_pil, model, model_name):
    transform = get_transform(model_name)
    img_size = MODEL_CONFIGS[model_name]["input_size"]

    if image_pil.size != img_size:
        image_pil = image_pil.resize(img_size)

    tensor = transform(image_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    return probs, tensor, img_size


def generate_vit_attention_visualization(model, tensor, orig_img_np, processing_id):
    try:
        if check_should_stop() or st.session_state.processing_id != processing_id:
            return None

        print("Attempting gradient-based ViT visualization...")

        model.eval()
        tensor.requires_grad_(True)

        output = model(tensor)
        if isinstance(output, tuple):
            output = output[0]
        predicted_class = output.argmax().item()

        model.zero_grad()
        output[0, predicted_class].backward()

        gradients = tensor.grad.cpu().numpy()[0]

        grad_magnitude = np.abs(gradients).sum(axis=0)

        grad_magnitude = (grad_magnitude - grad_magnitude.min()) / (
            grad_magnitude.max() - grad_magnitude.min()
        )

        if check_should_stop() or st.session_state.processing_id != processing_id:
            return None

        if show_heatmap_only:
            plt.figure(figsize=(6, 6))
            plt.imshow(grad_magnitude, cmap=colormap)
            plt.axis("off")
            plt.title("ViT Gradient Visualization")

            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
            buf.seek(0)
            img_array = np.array(Image.open(buf))
            plt.close()
            return img_array
        else:
            import cv2

            colormap_mapping = {
                "jet": cv2.COLORMAP_JET,
                "hot": cv2.COLORMAP_HOT,
                "viridis": cv2.COLORMAP_VIRIDIS,
                "plasma": cv2.COLORMAP_PLASMA,
            }
            cv_colormap = colormap_mapping.get(colormap, cv2.COLORMAP_JET)

            grad_8bit = (grad_magnitude * 255).astype(np.uint8)
            grad_colored = cv2.applyColorMap(grad_8bit, cv_colormap)
            grad_colored = cv2.cvtColor(grad_colored, cv2.COLOR_BGR2RGB)
            grad_colored = grad_colored.astype(np.float32) / 255.0

            blended = (
                orig_img_np * (1 - explanation_alpha) + grad_colored * explanation_alpha
            )
            blended = np.clip(blended, 0, 1)

            return (blended * 255).astype(np.uint8)

    except Exception as e:
        logger.error(f"Error generating ViT visualization: {str(e)}")
        return None


def generate_cam_explanations(
    model, model_name, tensor, predicted_class, orig_img_np, methods, processing_id
):
    results = {}

    if model_name == "ViT-Base-16":
        if "Attention Visualization" in methods:
            attention_result = generate_vit_attention_visualization(
                model, tensor, orig_img_np, processing_id
            )
            if attention_result is not None and (
                st.session_state.processing_id == processing_id
            ):
                results["Attention Visualization"] = attention_result
        return results

    try:
        if check_should_stop() or st.session_state.processing_id != processing_id:
            return results

        target_layers = get_target_layers(model, model_name)
        targets = [ClassifierOutputTarget(predicted_class)]

        import cv2

        colormap_mapping = {
            "jet": cv2.COLORMAP_JET,
            "hot": cv2.COLORMAP_HOT,
            "viridis": cv2.COLORMAP_VIRIDIS,
            "plasma": cv2.COLORMAP_PLASMA,
        }
        cv_colormap = colormap_mapping.get(colormap, cv2.COLORMAP_JET)

        cam_methods = {
            "Grad-CAM": GradCAM,
            "Grad-CAM++": GradCAMPlusPlus,
            "Eigen-CAM": EigenCAM,
            "Ablation-CAM": AblationCAM,
        }

        for method_name in methods:
            if check_should_stop() or st.session_state.processing_id != processing_id:
                break

            if method_name in cam_methods:
                try:
                    cam_class = cam_methods[method_name]
                    cam = cam_class(
                        model=model,
                        target_layers=target_layers,
                    )

                    if (
                        check_should_stop()
                        or st.session_state.processing_id != processing_id
                    ):
                        break

                    grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]

                    if (
                        check_should_stop()
                        or st.session_state.processing_id != processing_id
                    ):
                        break

                    if show_heatmap_only:
                        plt.figure(figsize=(6, 6))
                        plt.imshow(grayscale_cam, cmap=colormap)
                        plt.axis("off")
                        plt.title(f"{method_name} Heatmap")

                        if (
                            check_should_stop()
                            or st.session_state.processing_id != processing_id
                        ):
                            plt.close()
                            break

                        buf = io.BytesIO()
                        plt.savefig(
                            buf, format="png", bbox_inches="tight", pad_inches=0
                        )
                        buf.seek(0)
                        img_array = np.array(Image.open(buf))
                        plt.close()

                        if st.session_state.processing_id == processing_id:
                            results[method_name] = img_array
                    else:
                        cam_image = show_cam_on_image(
                            orig_img_np,
                            grayscale_cam,
                            use_rgb=True,
                            colormap=cv_colormap,
                            image_weight=1 - explanation_alpha,
                        )

                        if st.session_state.processing_id == processing_id:
                            results[method_name] = cam_image

                    if st.session_state.processing_id == processing_id:
                        logger.info(f"Generated {method_name} successfully")

                except Exception as e:
                    logger.error(f"Error generating {method_name}: {str(e)}")
                    if st.session_state.processing_id == processing_id:
                        st.warning(f"Could not generate {method_name}: {str(e)}")

    except Exception as e:
        logger.error(f"Error in CAM generation: {str(e)}")
        if st.session_state.processing_id == processing_id:
            st.error(f"Error generating CAM visualizations: {str(e)}")

    return results


def generate_lime_explanation(model, model_name, orig_img_np, processing_id):
    try:
        if check_should_stop() or st.session_state.processing_id != processing_id:
            return None

        transform = get_transform(model_name)

        def batch_predict(images):
            if check_should_stop() or st.session_state.processing_id != processing_id:
                return np.zeros((len(images), NUM_CLASSES))

            model.eval()
            batch_tensors = []

            for img in images:
                if (
                    check_should_stop()
                    or st.session_state.processing_id != processing_id
                ):
                    return np.zeros((len(images), NUM_CLASSES))

                if img.max() <= 1.0:
                    img_uint8 = (img * 255).astype(np.uint8)
                else:
                    img_uint8 = img.astype(np.uint8)

                pil_img = Image.fromarray(img_uint8)
                tensor = transform(pil_img)
                batch_tensors.append(tensor)

            if check_should_stop() or st.session_state.processing_id != processing_id:
                return np.zeros((len(images), NUM_CLASSES))

            if not batch_tensors:
                return np.zeros((1, NUM_CLASSES))

            batch = torch.stack(batch_tensors).to(DEVICE)

            with torch.no_grad():
                try:
                    logits = model(batch)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                    return probs
                except Exception as e:
                    print(f"Error in batch prediction: {e}")
                    return np.zeros((len(images), NUM_CLASSES))

        if check_should_stop() or st.session_state.processing_id != processing_id:
            return None

        explainer = lime_image.LimeImageExplainer()

        explanation = explainer.explain_instance(
            orig_img_np,
            batch_predict,
            top_labels=1,
            hide_color=0,
            num_samples=50,
            random_seed=42,
        )

        if check_should_stop() or st.session_state.processing_id != processing_id:
            return None

        img_uint8 = (
            (orig_img_np * 255).astype(np.uint8)
            if orig_img_np.max() <= 1.0
            else orig_img_np.astype(np.uint8)
        )
        pil_img = Image.fromarray(img_uint8)
        tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred_output = model(tensor)
            if isinstance(pred_output, tuple):
                pred_output = pred_output[0]
            pred_class = pred_output.argmax().item()

        if check_should_stop() or st.session_state.processing_id != processing_id:
            return None

        lime_img, mask = explanation.get_image_and_mask(
            label=pred_class,
            positive_only=True,
            hide_rest=False,
            num_features=10,
            min_weight=0.01,
        )

        if check_should_stop() or st.session_state.processing_id != processing_id:
            return None

        lime_result = mark_boundaries(lime_img, mask)

        if st.session_state.processing_id == processing_id:
            logger.info("Generated LIME explanation successfully")
            return lime_result
        else:
            return None

    except Exception as e:
        logger.error(f"Error generating LIME: {str(e)}")
        print(f"LIME error details: {e}")
        if st.session_state.processing_id == processing_id:
            st.warning(f"Could not generate LIME explanation: {str(e)}")
        return None


initialize_session_state()

input_image = None
image_source = "None"

if uploaded_file:
    input_image = Image.open(uploaded_file).convert("RGB")
    image_source = f"Uploaded: {uploaded_file.name}"
elif sample_choice != "None":
    sample_path = SAMPLE_IMAGES[sample_choice]
    if os.path.exists(sample_path):
        input_image = Image.open(sample_path).convert("RGB")
        image_source = f"Sample: {sample_choice}"
    else:
        st.warning(f"Sample image not found: {sample_path}")

if input_image is not None:
    new_image_hash = get_image_hash(input_image, selected_model, selected_methods)

    if st.session_state.current_image_hash != new_image_hash:
        st.session_state.stop_processing.set()

        clear_processing_memory()

        st.session_state.current_image_hash = new_image_hash
        st.session_state.processing_id = new_image_hash
        st.session_state.stop_processing.clear()

if input_image is not None:
    try:
        with st.spinner(f"Loading {selected_model} model..."):
            model = load_selected_model(selected_model)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Input Image")
            st.image(input_image, caption=image_source, width=300)
            st.info(f"**Source:** {image_source}")

        with col2:
            st.subheader("Prediction Results")

            with st.spinner("Making prediction..."):
                probs, tensor, img_size = predict_image(
                    input_image, model, selected_model
                )
                top_idxs = np.argsort(probs)[::-1][:3]

            predicted_class_name = CLASS_NAMES[top_idxs[0]]
            predicted_confidence = probs[top_idxs[0]] * 100

            st.markdown(
                f"""
            <div class="prediction-box">
            <h3>Final Prediction: {predicted_class_name}</h3>
            <h4>Confidence: {predicted_confidence:.2f}%</h4>
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.subheader("Top-3 Predictions")

            metric_cols = st.columns(3)
            for i, idx in enumerate(top_idxs):
                with metric_cols[i]:
                    confidence = probs[idx] * 100
                    st.metric(
                        label=f"#{i+1} {CLASS_NAMES[idx]}", value=f"{confidence:.2f}%"
                    )

            fig, ax = plt.subplots(figsize=(10, 6))
            classes = [CLASS_NAMES[idx] for idx in top_idxs]
            confidences = [probs[idx] * 100 for idx in top_idxs]

            bars = ax.barh(
                classes, confidences, color=["#1f77b4", "#ff7f0e", "#2ca02c"]
            )
            ax.set_xlabel("Confidence (%)")
            ax.set_title("Top-3 Prediction Confidence")
            ax.set_xlim(0, 100)

            for bar, conf in zip(bars, confidences):
                ax.text(
                    bar.get_width() + 1,
                    bar.get_y() + bar.get_height() / 2,
                    f"{conf:.1f}%",
                    va="center",
                    fontweight="bold",
                )

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        if selected_methods:
            st.markdown("---")
            st.subheader("Explainable AI Visualizations")

            predicted_class = top_idxs[0]
            orig_img_np = (
                np.array(input_image.resize(img_size)).astype(np.float32) / 255.0
            )

            with st.spinner("Generating explanations..."):
                current_processing_id = st.session_state.processing_id
                all_results = {}

                cam_methods = [m for m in selected_methods if m != "LIME"]
                if (
                    cam_methods
                    and st.session_state.processing_id == current_processing_id
                ):
                    cam_results = generate_cam_explanations(
                        model,
                        selected_model,
                        tensor,
                        predicted_class,
                        orig_img_np,
                        cam_methods,
                        current_processing_id,
                    )
                    if st.session_state.processing_id == current_processing_id:
                        all_results.update(cam_results)

                if (
                    "LIME" in selected_methods
                    and st.session_state.processing_id == current_processing_id
                ):
                    lime_result = generate_lime_explanation(
                        model, selected_model, orig_img_np, current_processing_id
                    )
                    if (
                        lime_result is not None
                        and st.session_state.processing_id == current_processing_id
                    ):
                        all_results["LIME"] = lime_result

            if all_results:
                st.info(
                    f"All explanations shown for the same image and predicted class: **{predicted_class_name}**"
                )

                num_methods = len(all_results)
                cols_per_row = 3

                method_items = list(all_results.items())
                for i in range(0, num_methods, cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, col in enumerate(cols):
                        idx = i + j
                        if idx < num_methods:
                            method_name, image = method_items[idx]
                            with col:
                                st.image(
                                    image,
                                    caption=f"{method_name} - {predicted_class_name}",
                                    use_container_width=True,
                                )
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 1, 1])

                with col2:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    zip_filename = f"explanations_{selected_model.replace(' ', '_').replace('-', '_').lower()}_{timestamp}.zip"

                    buf = io.BytesIO()
                    with zipfile.ZipFile(buf, "w") as z:
                        orig_buf = io.BytesIO()
                        input_image.save(orig_buf, format="PNG")
                        z.writestr("original_image.png", orig_buf.getvalue())

                        summary = f"""Guava Leaf Disease Classification Results
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model: {selected_model}
Image Source: {image_source}

PREDICTIONS:
Final Prediction: {predicted_class_name} ({predicted_confidence:.2f}%)

Top-3 Results:
"""
                        for i, idx in enumerate(top_idxs):
                            conf = probs[idx] * 100
                            summary += f"{i+1}. {CLASS_NAMES[idx]}: {conf:.2f}%\n"

                        summary += (
                            f"\nXAI Methods Used: {', '.join(selected_methods)}\n"
                        )
                        z.writestr("prediction_summary.txt", summary)

                        for name, image in all_results.items():
                            if image.dtype != np.uint8:
                                img_array = (image * 255).astype(np.uint8)
                            else:
                                img_array = image

                            img_pil = Image.fromarray(img_array)
                            img_bytes = io.BytesIO()
                            img_pil.save(img_bytes, format="PNG")
                            z.writestr(
                                f"{name.replace(' ', '_').replace('+', 'plus')}_explanation.png",
                                img_bytes.getvalue(),
                            )

                    st.download_button(
                        "Download All Results (ZIP)",
                        data=buf.getvalue(),
                        file_name=zip_filename,
                        mime="application/zip",
                        help="Download predictions, explanations, and summary report",
                    )
            else:
                st.warning(
                    "No explanations could be generated. Please try different XAI methods."
                )

    except Exception as e:
        logger.error(f"Error in main application: {str(e)}")
        st.markdown(
            f"""
        <div class="error-box">
        <h4>Application Error</h4>
        <p>An error occurred while processing your request:</p>
        <p><code>{str(e)}</code></p>
        <p>Please check your model files and try again.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
        st.error("Please check your model files and try again.")

else:
    st.markdown(
        """
    <div class="success-box">
    <h3>Welcome to the Guava Leaf Disease Classifier!</h3>
    <p>Please upload an image or select a sample from the sidebar to get started.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
