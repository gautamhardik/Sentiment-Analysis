import time
from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from app.schemas.prediction import (
    PredictRequest,
    PredictResponse,
    OverallPrediction,
    TimingInfo,
    PredictMetadata,
    ModelVersions
)
from app.services.model_loader import model_manager
from app.services.predictor import predict_lr, predict_lstm, predict_bert

router = APIRouter()

MODEL_VERSIONS = ModelVersions(
    lr="1.0.0",
    lstm="1.0.0",
    bert="1.0.0"
)

@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    start_time = time.perf_counter()
    models_result = {}
    
    pos_count = 0
    neg_count = 0
    total_count = 0
    
    try:
        if "lr" in request.models:
            lr_res = predict_lr(request.text)
            models_result["lr"] = lr_res
            total_count += 1
            if lr_res.label == "Positive": pos_count += 1
            else: neg_count += 1
            
        if "lstm" in request.models:
            lstm_res = predict_lstm(request.text)
            models_result["lstm"] = lstm_res
            total_count += 1
            if lstm_res.label == "Positive": pos_count += 1
            else: neg_count += 1
            
        if "bert" in request.models:
            bert_res = predict_bert(request.text)
            models_result["bert"] = bert_res
            total_count += 1
            if bert_res.label == "Positive": pos_count += 1
            else: neg_count += 1
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    if total_count == 0:
        raise HTTPException(status_code=400, detail="No valid models selected or available")
        
    majority_label = "Positive" if pos_count >= neg_count else "Negative"
    agreement = f"{max(pos_count, neg_count)}/{total_count}"
    
    total_ms = (time.perf_counter() - start_time) * 1000
    
    return PredictResponse(
        overall=OverallPrediction(label=majority_label, agreement=agreement),
        models=models_result,
        timing=TimingInfo(total_ms=total_ms),
        metadata=PredictMetadata(model_versions=MODEL_VERSIONS)
    )

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "backend": False, # Gateway sets this to true later
        "ml_service": True,
        "database": False, # Gateway sets this to true later
        "models_loaded": {
            "logistic_regression": model_manager.models_loaded.get("logistic_regression", False),
            "lstm": model_manager.models_loaded.get("lstm", False),
            "bert": model_manager.models_loaded.get("bert", False)
        },
        "version": "1.0.0"
    }

@router.get("/models")
async def get_models() -> Dict[str, Any]:
    return {
        "lr": {
            "name": "Logistic Regression (TF-IDF)",
            "type": "Machine Learning",
            "version": MODEL_VERSIONS.lr,
            "description": "Bag-of-words classifier using TF-IDF features"
        },
        "lstm": {
            "name": "Bi-LSTM",
            "type": "Deep Learning",
            "version": MODEL_VERSIONS.lstm,
            "description": "Bidirectional recurrent neural network"
        },
        "bert": {
            "name": "BERT (Fine-Tuned)",
            "type": "Transformer",
            "version": MODEL_VERSIONS.bert,
            "description": "Fine-tuned contextual embedding model"
        }
    }

@router.get("/model_metrics")
async def get_model_metrics() -> Dict[str, Any]:
    return {
        # Minimal mock payload for now, database will hold the real ones
        "status": "Not implemented here, served by DB via Gateway"
    }
