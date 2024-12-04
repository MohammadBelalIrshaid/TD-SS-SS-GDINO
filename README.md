# Threat Detection in X-ray Baggage Images Using Self-Supervised and Semi-Supervised Learning

## Project Overview
This project presents a robust pipeline for threat detection in X-ray baggage images using a combination of self-supervised, semi-supervised learning techniques, and advanced object detection. The system integrates **BYOL (Bootstrap Your Own Latent)** with the **Swin Transformer** for feature extraction, **K-Means clustering** for semi-supervised label generation, and **Grounding DINO** for threat classification and localization.

The pipeline addresses the challenge of limited labeled data by progressively refining feature extraction, classification, and detection capabilities across three stages:
1. **Self-Supervised Learning**: Extract robust visual representations using BYOL and Swin Transformer.
2. **Semi-Supervised Learning**: Classify images using K-Means clustering on 1,000 labeled images (500 threat, 500 non-threat) and generate pseudo-labels for unlabeled data.
3. **Object Detection with Grounding DINO**: Detect and assign bounding boxes using refined Swin Transformer weights and BERT-guided textual grounding.

## Methodology
- **Stage 1: Self-Supervised Learning (BYOL + Swin Transformer)**
  - BYOL learns representations by aligning predictions from an online network and a target network.
  - Swin Transformer extracts hierarchical and spatial features from X-ray images.
  - Output: Pre-trained Swin Transformer weights fine-tuned on visual patterns.

- **Stage 2: Semi-Supervised Learning (K-Means Clustering)**
  - Uses Swin Transformer to classify threat and non-threat images.
  - K-Means clustering generates pseudo-labels for unlabeled images based on extracted features.
  - Output: An enriched dataset with reliable pseudo-labels for further training.

- **Stage 3: Object Detection (Grounding DINO)**
  - Grounding DINO leverages pre-trained Swin weights and BERT textual annotations for precise detection.
  - Integrates textual grounding to refine visual feature detection and classification.
  - Output: Final classification and bounding boxes for detected threats.

## Results
- **Labeling Accuracy**: Outperformed ResNet-18 and Timm Swin in labeling the SIXray10 dataset, achieving a **PT ratio** of 0.11 and **NN ratio** of 0.92.
- **Classification Baseline**: Initial tests using ResNet-50 achieved:
  - **Overall Accuracy**: 55%
  - **Weighted Precision**: 80%
  - **Threat Class F1-Score**: 15%
- **Detection Performance**:
  - Achieved **Average Precision (AP)** of 0.83 at 0.5 IoU with 1000 proposals.
  - Best recall of **0.727** with large area proposals.

---

## System Requirements
- **Operating System**: Linux or Windows (64-bit)
- **Python Version**: 3.8 or higher
- **GPU**: NVIDIA GPU with CUDA 11.3+ (for optimal performance)
- **RAM**: At least 16 GB (recommended)
- **Disk Space**: At least 20 GB for datasets and model weights

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/threat-detection-xray.git
   cd threat-detection-xray

2. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
3. **Install PyTorch (with CUDA support):**
   ```bash
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117

