"""
FastAPI REST API Server for DeepVoiceGuard Voice Cloning Detection.
"""

import os
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

from ..models.ensemble import DeepVoiceGuard
from ..forensics.analyzer import ForensicAnalyzer

app = FastAPI(
    title="DeepVoiceGuard API",
    description="REST API for AI Voice Cloning & Audio Deepfake Detection",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
detector: Optional[DeepVoiceGuard] = None
forensic_analyzer = ForensicAnalyzer()

@app.on_event("startup")
def load_models():
    global detector
    tab_path = "saved_models/tabular_model.joblib"
    lcnn_path = "saved_models/lcnn_model.pt"
    spec_path = "saved_models/specresnet_model.pt"

    detector = DeepVoiceGuard(
        tabular_model_path=tab_path if os.path.exists(tab_path) else None,
        lcnn_model_path=lcnn_path if os.path.exists(lcnn_path) else None,
        specresnet_model_path=spec_path if os.path.exists(spec_path) else None
    )
    print("DeepVoiceGuard models initialized for API service.")

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "service": "DeepVoiceGuard",
        "models_loaded": {
            "tabular": detector.tabular_model is not None if detector else False,
            "lcnn": detector.lcnn_model is not None if detector else False,
            "specresnet": detector.specresnet_model is not None if detector else False
        }
    }

@app.post("/api/v1/scan")
async def scan_audio_endpoint(file: UploadFile = File(...)):
    """
    Scans uploaded audio file (.wav, .mp3, .flac, .ogg) and returns detection verdict.
    """
    if not file.filename.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')):
        raise HTTPException(status_code=400, detail="Unsupported audio file format. Please upload .wav, .mp3, .flac, or .ogg")

    try:
        content = await file.read()
        res = detector.scan_audio(io.BytesIO(content))
        return {
            "filename": file.filename,
            "verdict": res["verdict"],
            "is_cloned": res["is_cloned"],
            "cloned_probability": res["cloned_probability"],
            "real_probability": res["real_probability"],
            "confidence_score": res["confidence_score"],
            "risk_level": res["risk_level"],
            "audio_duration_seconds": res["audio_duration"],
            "num_segments_analyzed": res["num_segments"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio processing error: {str(e)}")

@app.post("/api/v1/analyze")
async def analyze_audio_endpoint(file: UploadFile = File(...)):
    """
    Performs full forensic deep-dive including segment anomaly timeline and radar metrics.
    """
    if not file.filename.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')):
        raise HTTPException(status_code=400, detail="Unsupported audio file format.")

    try:
        content = await file.read()
        res = detector.scan_audio(io.BytesIO(content))
        radar = forensic_analyzer.compute_forensic_radar(res["forensics"])
        summary = forensic_analyzer.generate_forensic_summary(res)

        return {
            "filename": file.filename,
            "verdict": res["verdict"],
            "cloned_probability": res["cloned_probability"],
            "risk_level": res["risk_level"],
            "confidence_score": res["confidence_score"],
            "forensic_radar_scores": radar,
            "forensic_findings": summary,
            "segment_timeline": res["segment_timeline"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")
