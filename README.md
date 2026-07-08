---
title: The Screening Room
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# The Screening Room

> **Three Critics. One Review. Three Perspectives.**

[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit-red?style=for-the-badge&logo=huggingface&logoColor=white)](https://hardik-25-sentiment-analysis.hf.space/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

An AI-powered movie review sentiment analysis platform that compares how three fundamentally different machine learning architectures — **Logistic Regression**, **Bi-LSTM**, and **Fine-Tuned BERT** — interpret the same text. Deployed as a production-grade multi-service application on Hugging Face Spaces.

---

## Table of Contents

- [Live Demo](#-live-demo)
- [Why This Project?](#-why-this-project)
- [System Architecture](#-system-architecture)
- [Key Features](#-key-features)
- [Technology Stack](#-technology-stack)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Experimental Results](#-experimental-results)
- [Key Insights](#-key-insights)
- [Application Showcase](#-application-showcase)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
- [Engineering Decisions](#-engineering-decisions)
- [Limitations](#-limitations)
- [Future Roadmap](#-future-roadmap)
- [Author](#-author)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## Live Demo

> **Try it now:** [hardik-25-sentiment-analysis.hf.space](https://hardik-25-sentiment-analysis.hf.space/)

Watch the complete prediction workflow from review submission to simultaneous inference across all three models. The demonstration highlights the application's production-style interface, real-time predictions, model comparison, confidence scores, latency reporting, and explainability features.

**Highlights shown in the demo**

- Review submission with "Featured Screenings" sample reviews
- Real-time prediction across three independent models
- Majority vote consensus with agreement percentage
- Confidence visualization with animated progress bars
- Latency comparison per model
- Explainability: top positive/negative word extraction (LR) and reasoning engine
- Model metrics dashboard with accuracy, precision, recall, and F1 scores

---

## Why This Project?

### Business Problem

Sentiment analysis is a foundational NLP task, but most implementations rely on a single model. This makes it difficult to understand **how different architectures arrive at their conclusions** and **which approach works best for different types of text**.

### Motivation

Movie reviews are notoriously difficult for sentiment analysis. They contain:

- **Mixed sentiment** — *"The cinematography was breathtaking, but the plot was a disaster."*
- **Sarcasm** — *"Sure, because what the world needed was another superhero reboot."*
- **Negation** — *"I can't say I disliked it, but I also can't say I liked it."*
- **Complex context** — *"A masterpiece that somehow manages to be both profound and profoundly boring."*

### What Makes It Different

| Aspect | Typical Approach | The Screening Room |
|--------|------------------|-------------------|
| Models | Single classifier | Three architectures running in parallel |
| Output | Binary label | Label + confidence + latency + reasoning |
| Comparison | None | Side-by-side model disagreement analysis |
| Architecture | Monolithic | Multi-service (Nginx + Express + FastAPI) |
| Deployment | Notebook | Production Docker container |

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            Hugging Face Space                                │
│                           (Single Docker Container)                          │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                              Nginx (:7860)                              │ │
│  │                    Static files + Reverse proxy                         │ │
│  └───────────┬──────────────────────────────────────────┬──────────────────┘ │
│              │                                          │                    │
│              ▼                                          ▼                    │
│  ┌─────────────────────┐              ┌──────────────────────────────┐      │
│  │    Express.js        │              │        FastAPI                │      │
│  │    Backend (:4000)   │─────────────▶│     ML Service (:8000)       │      │
│  │                      │              │                              │      │
│  │  • Validation        │              │  • Logistic Regression       │      │
│  │  • Rate limiting     │              │  • Bi-LSTM                   │      │
│  │  • Session mgmt      │              │  • Fine-Tuned BERT           │      │
│  │  • SQLite (Prisma)   │              │  • Explainer engine          │      │
│  └─────────────────────┘              └──────────────────────────────┘      │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                         Supervisor (Process Manager)                    │ │
│  │              Manages Nginx, Express, and FastAPI lifecycle              │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Request Lifecycle

```
User submits review
        │
        ▼
   Nginx (:7860)
   Static files + /api/* proxy
        │
        ▼
   Express (:4000)
   Validation → Rate limiting → Session
        │
        ▼
   FastAPI (:8000)
   Model-specific preprocessing
        │
        ├──▶ Logistic Regression (TF-IDF)  ──┐
        ├──▶ Bi-LSTM (Tokenizer + Pad)     ──┼──▶ Standardized Response
        └──▶ BERT (Tokenizer + Attention)  ──┘
                                                    │
                                                    ▼
                                          Majority Vote + Confidence
                                                    │
                                                    ▼
                                          Frontend renders results
```

---

## Key Features

### Machine Learning

| Feature | Details |
|---------|---------|
| **Three architectures** | Logistic Regression (bag-of-words), Bi-LSTM (sequential), BERT (transformer) |
| **Model-specific preprocessing** | Each model receives text prepared according to its own pipeline |
| **Confidence normalization** | Raw probabilities mapped to a consistent 0–1 scale across models |
| **Explainability** | Top positive/negative words (LR) + reasoning engine (all models) |
| **Model warmup** | Inference pre-warmed at startup to eliminate cold-start latency |

### Engineering

| Feature | Details |
|---------|---------|
| **Multi-service architecture** | Nginx, Express, FastAPI — each process handles its responsibility |
| **Process orchestration** | Supervisor manages all services in a single Docker container |
| **SQLite persistence** | Prediction history and model metrics stored via Prisma ORM |
| **Rate limiting** | Express middleware protects the prediction endpoint |
| **Resilient inference** | Per-model try/catch — one model failure doesn't crash the request |

### User Experience

| Feature | Details |
|---------|---------|
| **Cinema-themed design** | Gold accents, ticket-styled inputs, marquee borders |
| **Real-time comparison** | Side-by-side model outputs with confidence bars |
| **Majority vote** | Consensus verdict with agreement percentage |
| **Featured Screenings** | Pre-built sample reviews for quick testing |
| **Latency display** | Per-model inference time shown in the results |

---

## Technology Stack

### Layer-wise Breakdown

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | HTML, CSS, JavaScript | Static UI served by Nginx |
| **Reverse Proxy** | Nginx | Static files, request routing, port 7860 |
| **Backend API** | Express.js + Prisma | Validation, routing, session management, persistence |
| **ML Inference** | FastAPI + Uvicorn | High-performance Python API for model serving |
| **ML Models** | Scikit-learn, TensorFlow/Keras, PyTorch + Transformers | Three sentiment classifiers |
| **Database** | SQLite (via Prisma ORM) | Prediction history, model metrics |
| **Process Manager** | Supervisor | Manages all services in single container |
| **Container** | Docker | Consistent deployment across environments |
| **Deployment** | Hugging Face Spaces (Docker SDK) | Cloud hosting with single-port exposure |

### Python Dependencies

```
numpy, pandas, scikit-learn, scipy, nltk
tensorflow, keras, torch, transformers
fastapi, uvicorn, pydantic, joblib
```

### Node.js Dependencies

```
express, prisma, axios, cors, helmet
morgan, zod, express-rate-limit
```

---

## Machine Learning Pipeline

### Workflow Diagram

```
Raw Movie Review
       │
       ▼
┌──────────────────┐
│  Text Preprocessing │
│  (Model-Specific)   │
└──────────┬───────┘
           │
     ┌─────┼─────┐
     ▼     ▼     ▼
┌────────┐ ┌────────┐ ┌────────┐
│ TF-IDF │ │Tokenizer│ │BERT    │
│ Vector │ │ + Pad   │ │Tokenizer│
└───┬────┘ └───┬────┘ └───┬────┘
    ▼          ▼          ▼
┌────────┐ ┌────────┐ ┌────────┐
│Logistic│ │ Bi-LSTM│ │  BERT  │
│Regression│ │(64 units)│ │(110M) │
└───┬────┘ └───┬────┘ └───┬────┘
    ▼          ▼          ▼
┌────────────────────────────────┐
│    Standardized Response        │
│  {label, confidence, latency,  │
│   reasoning, keywords}         │
└───────────────┬────────────────┘
                ▼
┌────────────────────────────────┐
│    Majority Vote + Verdict      │
└────────────────────────────────┘
```

### Training Process

| Model | Training Data | Features | Epochs | Optimizer |
|-------|--------------|----------|--------|-----------|
| Logistic Regression | IMDb 25k train | TF-IDF (50k features) | N/A | LBFGS |
| Bi-LSTM | IMDb 25k train | Tokenized sequences (max_len=300) | 10 | Adam |
| BERT | IMDb 25k train | WordPiece tokens (max_length=128) | 3 | AdamW |

### Preprocessing Pipelines

| Model | Steps |
|-------|-------|
| **LR (TF-IDF)** | Lowercase → Expand contractions → Remove HTML → Remove URLs → Remove special chars → Stopword removal (preserve negations) → Lemmatize |
| **Bi-LSTM** | Lowercase → Expand contractions → Remove HTML → Remove URLs → Keep alphanumeric + basic punctuation |
| **BERT** | Remove HTML → Normalize whitespace (tokenizer handles the rest) |

---

## Experimental Results

### Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | Parameters | Model Size | Inference |
|-------|----------|-----------|--------|----------|------------|------------|-----------|
| **Logistic Regression** | 90.15% | 90.10% | 90.20% | 90.10% | ~50K | ~1.5 MB | 2 ms |
| **Bi-LSTM** | 84.67% | 85.00% | 84.50% | 84.70% | ~120K | ~5 MB | 22 ms |
| **Fine-Tuned BERT** | 91.68% | 93.32% | 89.79% | 91.52% | ~110M | ~420 MB | 72 ms |

### Accuracy vs. Latency Trade-off

```
Accuracy (%)
    │
 92 ┤                                    ● BERT (91.68%, 72ms)
    │
 90 ┤  ● LR (90.15%, 2ms)
    │
 88 ┤
    │
 86 ┤
    │
 84 ┤          ● Bi-LSTM (84.67%, 22ms)
    │
    └──────────┬──────────┬──────────┬──────
               0         25         50        75
                        Latency (ms)
```

### Key Observations

- **Logistic Regression** achieves 90.15% accuracy at only 2ms — an exceptional speed-to-accuracy ratio
- **BERT** wins on raw accuracy (91.68%) but costs 36x more latency than LR
- **Bi-LSTM** underperforms both — its sequential nature adds complexity without matching LR's bag-of-words efficiency on this dataset
- The 1.5% gap between LR and BERT may not justify the 35x latency increase for many real-world applications

---

## Key Insights

### Why Logistic Regression Performs So Well

Movie reviews are rich in **sentiment-laden keywords** ("amazing", "terrible", "waste", "masterpiece"). TF-IDF captures these directly. Since sentiment in reviews is often expressed through **explicit word choices** rather than subtle context, the bag-of-words approach is surprisingly competitive.

### Why BERT Wins

BERT excels on reviews with:
- **Negation** — *"I can't say I disliked it"* (LR sees "disliked" as negative; BERT understands the double negation)
- **Sarcasm** — *"Sure, because what the world needed was another reboot"* (LR reads "great" positively; BERT detects contextual irony)
- **Mixed sentiment** — *"The cinematography was breathtaking but the plot was a disaster"* (LR averages the signals; BERT weighs them through attention)

### Engineering Observations

- **Singleton model loading** at startup eliminates cold-start penalties for all subsequent requests
- **Per-model try/catch** in the prediction router means a BERT timeout doesn't prevent LR and LSTM from returning results
- **Model warmup** with dummy inputs forces TensorFlow graph compilation and PyTorch CUDA initialization before the first real request
- **Negation-preserving stopword removal** is critical — removing "not", "never", "can't" destroys the most important sentiment inversion signals

---

## Application Showcase

### Landing Page

The main interface presents a cinema-themed design with:
- Gold-accented ticket-style input area
- "Featured Screenings" section with pre-built sample reviews
- Category badges (Positive, Negative, Mixed, Sarcasm, Negation, Complex Context)
- Difficulty indicators for each sample review

### Prediction Results

When a review is analyzed:
- **Overall Verdict** banner with majority vote and agreement percentage
- **Three critic cards** showing each model's prediction, confidence bar, and latency
- **LR Explainability** panel with top positive and negative word extraction
- **Reasoning engine** generating natural-language explanations per model

### Model Metrics Dashboard

The "Inside the Screening Room" page displays:
- Accuracy, Precision, Recall, and F1 for each model
- Dataset information (IMDb Large Movie Review Dataset)
- Technology stack breakdown
- Architecture flow diagram

---

## Repository Structure

```
sentiment-analysis/
│
├── frontend/                    # Static frontend (HTML/CSS/JS)
│   ├── index.html               # Main analysis page
│   ├── inside.html              # Model details & project info page
│   ├── style.css                # Cinema-themed styling
│   ├── app.js                   # Analysis UI logic + Featured Screenings
│   ├── inside.js                # Inside page logic (metrics, accordions)
│   └── film_reel.png            # Logo asset
│
├── backend/                     # Express.js API Gateway
│   ├── src/
│   │   ├── server.js            # Express app entry point
│   │   ├── config/index.js      # Environment configuration
│   │   ├── routes/v1.routes.js  # API route definitions
│   │   ├── controllers/
│   │   │   └── predict.controller.js  # Prediction, health, metrics handlers
│   │   ├── services/
│   │   │   ├── mlService.js     # FastAPI client (axios)
│   │   │   └── dbService.js     # Prisma database operations
│   │   └── middleware/
│   │       ├── validation.js    # Zod request validation
│   │       ├── rateLimiter.js   # Express rate limiting
│   │       └── errorHandler.js  # Centralized error handling
│   ├── prisma/
│   │   ├── schema.prisma        # Database schema
│   │   └── seed.js              # Model metrics seed data
│   └── package.json
│
├── ml_service/                  # FastAPI ML Inference Service
│   ├── main.py                  # FastAPI app entry point
│   ├── app/
│   │   ├── config.py            # Model paths & settings
│   │   ├── api/
│   │   │   └── routes.py        # /predict, /health, /models endpoints
│   │   ├── schemas/
│   │   │   └── prediction.py    # Pydantic request/response models
│   │   ├── services/
│   │   │   ├── model_loader.py  # Singleton model manager (LR, LSTM, BERT)
│   │   │   ├── predictor.py     # Per-model prediction functions
│   │   │   └── explainer.py     # Reasoning engine + word importance
│   │   └── utils/
│   │       └── preprocessing.py # Model-specific text preprocessing
│   └── requirements.txt         # (not used in Docker, see root requirements.txt)
│
├── models_ml/                   # Logistic Regression artifacts
│   ├── sentiment_model.pkl      # Trained LR model
│   └── tfidf_vectorizer.pkl     # Fitted TF-IDF vectorizer
│
├── models_lstm/                 # Bi-LSTM artifacts
│   └── best_lstm_model.keras    # Trained Keras model
│
├── models_bert/                 # BERT artifacts
│   ├── tuned/                   # Fine-tuned BERT model
│   │   ├── model.safetensors    # Model weights
│   │   ├── config.json          # Model configuration
│   │   ├── tokenizer.json       # WordPiece tokenizer
│   │   └── threshold.pkl        # Optimized classification threshold
│   └── *.json                   # Training config, tokenizer config
│
├── data/                        # Preprocessing artifacts
│   ├── tokenizer.pkl            # Keras tokenizer for LSTM
│   └── config.pkl               # LSTM configuration
│
├── Dockerfile                   # Single-container build (Python + Node + Nginx)
├── supervisord.conf             # Process manager configuration
├── nginx.conf                   # Reverse proxy configuration
├── hf_start.sh                  # Hugging Face Spaces entrypoint
├── docker-compose.yml           # Local multi-service development
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## Getting Started

### Docker (Recommended)

```bash
docker-compose up --build
```

This starts four services:
- **Frontend:** http://localhost:8080
- **Backend API:** http://localhost:4000
- **ML Service:** http://localhost:8000
- **PostgreSQL:** localhost:5432

### Hugging Face Spaces (Production)

The app is deployed as a single Docker container on Hugging Face Spaces. All services (Nginx, Express, FastAPI) are managed by Supervisor behind port 7860.

**Live URL:** [hardik-25-sentiment-analysis.hf.space](https://hardik-25-sentiment-analysis.hf.space/)

### Local Development

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install backend dependencies
cd backend && npm install && cd ..

# 3. Start ML service
cd ml_service && uvicorn main:app --host 0.0.0.0 --port 8000 &

# 4. Start backend
cd backend && npm start &

# 5. Serve frontend (any static server)
npx serve frontend -p 8080
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/predict` | Analyze a movie review |
| `GET` | `/api/v1/health` | System health check |
| `GET` | `/api/v1/models` | List available models |
| `GET` | `/api/v1/metrics` | Get model performance metrics |
| `GET` | `/api/v1/history/:sessionId` | Get prediction history |

**Example request:**

```bash
curl -X POST http://localhost:4000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was absolutely brilliant!", "models": ["lr", "lstm", "bert"]}'
```

---

## Engineering Decisions

### Why FastAPI for ML?

FastAPI provides native async support, automatic request validation via Pydantic, and seamless integration with Python's ML ecosystem (NumPy, PyTorch, TensorFlow). It serves as a natural boundary between application logic and model inference.

### Why Express for the Backend?

Express separates concerns by handling validation, rate limiting, session management, and database operations. This keeps the ML service focused purely on inference and allows the backend to be swapped or scaled independently.

### Why SQLite + Prisma?

For a portfolio project deployed on Hugging Face Spaces (ephemeral storage), SQLite provides zero-configuration persistence. Prisma's type-safe queries and schema migrations demonstrate production-style ORM practices without requiring a external database service.

### Why Three Models?

The goal is not only to predict sentiment but to **compare how fundamentally different NLP approaches interpret the same review**. This reveals:
- When bag-of-words fails (negation, sarcasm)
- When transformers justify their computational cost
- When simpler models are "good enough"

### Why Supervisor?

Hugging Face Spaces exposes a single port. Supervisor manages three processes (Nginx, Express, FastAPI) in one container, each with automatic restart and log separation — mimicking a minimal production orchestration setup.

---

## Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|------------|
| **No GPU in HF Spaces** | BERT inference is CPU-bound (~72ms) | Model warmup at startup; threshold optimization |
| **SQLite is ephemeral** | Data lost on Space restart | Metrics seeded at build time; prediction history is session-based |
| **Single-container constraint** | No horizontal scaling | Sufficient for portfolio demo; architecture supports splitting |
| **Binary classification only** | No neutral/multi-class support | Future roadmap includes multi-class extension |
| **English-only** | No multilingual support | Future roadmap includes multilingual models |

---

## Future Roadmap

| Priority | Feature | Status |
|----------|---------|--------|
| High | Multilingual sentiment analysis | Planned |
| High | Batch analysis (multiple reviews) | Planned |
| Medium | Additional transformer models (RoBERTa, DistilBERT) | Planned |
| Medium | Mobile-responsive interface | Planned |
| Low | Advanced model interpretation (SHAP, attention visualization) | Research |
| Low | Real-time streaming predictions | Research |

---

## Author

**Hardik** — [GitHub](https://github.com/Hardik-25)

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- **IMDb Large Movie Review Dataset** — Andrew L. Maas et al., Stanford AI Lab ([source](https://ai.stanford.edu/~amaas/data/sentiment/))
- **Hugging Face** — For the Transformers library and Spaces deployment platform
- **FastAPI** — For the high-performance Python web framework
- **Express.js** — For the minimalist Node.js backend framework
- **Prisma** — For the type-safe ORM and database toolkit
