# 🛡️ DeepVoiceGuard: AI Voice Cloning & Audio Deepfake Detection System

A complete, high-accuracy machine learning and acoustic signal processing system designed to detect AI-cloned, synthesized (e.g., ElevenLabs, XTTS, Bark, VITS, Tortoise, DiffSinger), and manipulated human speech.

---

## 🌟 Key Features

- **Multi-Domain Feature Extraction**:
  - **Linear Frequency Cepstral Coefficients (LFCC)**: Extracts 60-channel triangular filterbank representations preserving high-frequency vocoder phase artifacts (standard in ASVspoof speech anti-spoofing).
  - **Mel-Frequency Cepstral Coefficients (MFCC)**: 20 MFCCs + 10 first-order deltas.
  - **Pitch (F0) & Harmonics**: Fundamental frequency tracking, micro-jitter analysis, and Harmonic-to-Noise Ratio (HNR).
  - **Sub-Band Spectral Energy Ratios**: Captures sharp vocoder high-frequency cutoffs (>7 kHz) and diffusion noise floors.
  - **160+ Extracted Acoustic Features**: Spectral centroid, rolloff, flatness, contrast, skewness, and kurtosis.

- **Multi-Model Hybrid Classifier Suite**:
  - **Tabular GBDT Ensemble**: Soft-voting ensemble combining **LightGBM**, **XGBoost**, **Random Forest**, and **Extra Trees**.
  - **LFCC-LCNN Deep Network**: PyTorch Convolutional Neural Network with Max-Feature-Map (MFM) activation units specifically designed for speech anti-spoofing.
  - **SpecResNet Attention Network**: 2D Spectrogram Residual Network with Adaptive Global Pooling.

- **Explainable Forensic Diagnostics**:
  - **Temporal Segment Anomaly Timeline**: Locates exact second-by-second intervals where cloning artifacts occur.
  - **Acoustic Radar Profile**: 6-axis comparison against biological human speech physics.
  - **Interactive Spectrogram Viewer**: High-resolution Log-Mel and Linear STFT spectrograms.

- **Full Application Ecosystem**:
  - **Streamlit Web Dashboard**: Live audio upload, microphone recording, preset demo testing, and batch scanner.
  - **Command-Line Interface (CLI)**: Single-file scan, batch directory scan, and model training.
  - **FastAPI REST API Server**: Programmatic audio verification endpoints for external integration.

---

## 🚀 Quickstart

### 1. Setup Environment & Pre-train Models
Run the automated quickstart setup to generate demo samples and train the model suite:
```bash
python run_demo.py
```

### 2. Launch the Streamlit Web Application
```bash
streamlit run app/streamlit_app.py
```

---

## 🖥️ Command-Line Interface (CLI)

### Scan a Single Audio File
```bash
python cli.py scan --file path/to/audio.wav
```
Optionally export the diagnostic report to JSON:
```bash
python cli.py scan --file path/to/audio.wav --json report.json
```

### Batch Audit an Entire Folder
```bash
python cli.py scan-batch --dir path/to/recordings/ --out audit_report.csv
```

### Run Calibration & Training Setup
```bash
python cli.py demo-setup --samples 50 --epochs 12
```

---

## 🌐 REST API Endpoints

Start the FastAPI server:
```bash
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
```

- **`POST /api/v1/scan`**: Accepts `.wav`, `.mp3`, `.flac`, `.ogg` audio and returns the verdict, cloned probability, and confidence.
- **`POST /api/v1/analyze`**: Returns full forensic diagnostic breakdown, segment timeline, and radar scores.
- **`GET /health`**: Service status and model loading state.

---

## 📁 Project Architecture

```
d:/ML project/
├── app/
│   └── streamlit_app.py          # Interactive Streamlit Web Dashboard
├── src/
│   ├── features/
│   │   ├── audio_loader.py       # Audio loading, resampling & VAD silence trimming
│   │   ├── lfcc.py               # Linear Frequency Cepstral Coefficients (LFCC)
│   │   ├── spectral_forensics.py # Pitch, Jitter, HNR, and Spectral Descriptors
│   │   └── extractor.py          # Unified Multi-Modal Feature Extraction Engine
│   ├── models/
│   │   ├── tabular_models.py     # LightGBM, XGBoost, Random Forest, ExtraTrees
│   │   ├── deep_models.py        # PyTorch LFCC-LCNN & SpecResNet Architectures
│   │   └── ensemble.py           # Unified DeepVoiceGuard Ensemble Classifier
│   ├── forensics/
│   │   └── analyzer.py           # Anomaly timeline, Spectrograms & Radar scoring
│   ├── data/
│   │   ├── synthetic_generator.py# Procedural speech & synthetic artifact simulator
│   │   └── dataset_loader.py     # Dataset loading & batch feature extraction
│   ├── training/
│   │   ├── metrics.py            # Equal Error Rate (EER), ROC-AUC, Precision, Recall
│   │   └── trainer.py            # End-to-end training and evaluation pipeline
│   └── api/
│       └── server.py             # FastAPI REST verification service
├── sample_data/
│   ├── real/                     # Verified authentic human speech samples
│   └── cloned/                   # Sample AI-cloned speech clips
├── saved_models/                 # Model checkpoints, scalers & training metrics
├── cli.py                        # Command-line interface tool
├── run_demo.py                   # Automated setup & verification script
└── requirements.txt              # Project dependencies
```

---

## 🔬 Evaluation Metrics & Benchmarks

DeepVoiceGuard evaluates detection models using standard speech anti-spoofing benchmarks:
- **Equal Error Rate (EER)**: Threshold where False Acceptance Rate equals False Rejection Rate.
- **ROC-AUC & PR-AUC**: Area under the Receiver Operating Characteristic and Precision-Recall curves.
- **Segment Anomaly Localization**: Fine-grained sub-clip forensic attribution.
