# Multimodal Video Sentiment Analysis System



## Overview
This project implements an end-to-end **multimodal sentiment analysis system** that processes video content to detect emotions and sentiments by analyzing three modalities: speech text, visual frames, and audio characteristics. The system employs deep learning to fuse these modalities into a unified model, demonstrating advanced skills in multimodal AI engineering.

---

## Model Architecture

### Multimodal Fusion Approach
The core innovation lies in the **late fusion architecture** that combines extracted features from three independent encoders:

#### 1. Text Encoder (BERT-based)
- **Architecture**: Frozen BERT-base model with projection head
- **Input**: Transcribed speech text from Whisper ASR
- **Processing**:
  - Extracts [CLS] token embeddings (768-dim)
  - Projects to 128-dim latent space
- **Key Features**:
  - Transfer learning from pre-trained language model
  - Handles semantic understanding of spoken content

#### 2. Video Encoder (3D CNN)
- **Architecture**: Modified R3D-18 (3D ResNet) with adaptive head
- **Input**: 30 uniformly sampled video frames (224x224px)
- **Processing**:
  - Temporal feature extraction with 3D convolutions
  - Final projection to 128-dim space
- **Key Features**:
  - Captures spatiotemporal patterns
  - Processes RGB values normalized [0-1]

#### 3. Audio Encoder (1D CNN)
- **Architecture**: Custom convolutional network
- **Input**: 64-bin Mel-spectrograms (300 time steps)
- **Processing**:
  - Stacked 1D convolutions with pooling
  - Global average pooling + projection to 128-dim
- **Key Features**:
  - Extracts paralinguistic features
  - Robust to variable-length audio

### Feature Fusion & Classification
- **Fusion Layer**:
  - Concatenates 128-dim features from all modalities (384-dim total)
  - Processes through dense layers with batch normalization
  - Output: 256-dim joint representation

- **Dual Classification Heads**:
  1. **Emotion Recognition** (7 classes):  
     `anger, disgust, fear, joy, neutral, sadness, surprise`
  2. **Sentiment Analysis** (3 classes):  
     `positive, negative, neutral`

---

## Technical Highlights

### Key Innovations
- **Temporal Alignment**: Processes video segments synchronized with ASR outputs
- **Multimodal Balance**: Equal feature dimensions (128-dim) prevent modality dominance
- **Efficient Inference**:
  - Frame sampling for computational efficiency
  - ONNX runtime compatibility (potential optimization)
  - Batch processing of modalities

### Training Characteristics
- **Loss Function**: Dual cross-entropy loss (emotion + sentiment)
- **Regularization**:
  - Dropout (20-30%)
  - Batch normalization
  - Frozen encoder backbones
- **Optimization**: AdamW with weight decay

---

## Application Features

### Interactive Dashboard
- **Video Upload & Preview**: MP4 file processing
- **Analysis Visualization**:
  - Emotion confidence timeline (Plotly)
  - Sentiment distribution radar chart
  - Segment-level confidence metrics
- **Technical Insights**:
  - Per-frame processing statistics
  - Modality contribution weights
  - Confidence threshold adjustments

### Pipeline Architecture
1. **Input Processing**:
   - Video segmentation with FFmpeg
   - Whisper speech recognition
2. **Feature Extraction**:
   - Parallel modality processing
   - GPU-accelerated inference
3. **Multimodal Fusion**:
   - Feature concatenation
   - Joint representation learning
4. **Postprocessing**:
   - Softmax normalization
   - Top-k predictions aggregation

---

## Technology Stack

**Core AI**:
- PyTorch (Model Development)
- Transformers (BERT Text Encoder)
- Whisper (Speech Recognition)
- OpenCV (Video Processing)

**Backend**:
- FFmpeg (Video Processing)
- TorchAudio (Audio Features)
- NumPy (Array Operations)

**Frontend**:
- Streamlit (Web Interface)
- Plotly (Data Visualization)
- CSS (UI Styling)


This README emphasizes architectural decisions and technical depth while maintaining readability. It highlights transferable skills important for resume presentations, with clear sectioning for easy scanning by technical recruiters. The placeholder image location can be replaced with an actual architecture diagram for enhanced visual impact.
