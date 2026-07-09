---
title: The Screening Room
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

<div align="center">

# 🎬 The Screening Room

### Three Critics. One Review. Three Perspectives.

[![Live Demo](https://img.shields.io/badge/Live_Demo-Hugging_Face_Spaces-blue?style=for-the-badge&logo=huggingface&logoColor=white)](https://hardik-25-sentiment-analysis.hf.space/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

An AI-powered movie review sentiment analysis platform that compares how three fundamentally different machine learning architectures — **Logistic Regression**, **Bi-LSTM**, and **Fine-Tuned BERT** — interpret the same text. Deployed as a multi-service application in a single Docker container on Hugging Face Spaces.

<br>

<img src="assets/hero-landing.png" alt="The Screening Room landing page" width="100%">

</div>

---

## Table of Contents

- [Live Demo](#live-demo)
- [Why This Project?](#why-this-project)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Experimental Results](#experimental-results)
- [Key Insights](#key-insights)
- [Application Showcase](#application-showcase)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Engineering Decisions](#engineering-decisions)
- [Limitations](#limitations)
- [Future Roadmap](#future-roadmap)
- [Author](#author)

---
## Demo Video

https://github.com/user-attachments/assets/5c47cc05-40f5-4e9a-ad10-e32d959c6cc8


---
## Live Demo

> **Try it now:** [hardik-25-sentiment-analysis.hf.space](https://hardik-25-sentiment-analysis.hf.space/)

**Highlights shown in the demo**

- Review submission with "Featured Screenings" sample reviews
- Real-time prediction across three independent models
- Majority vote consensus with agreement percentage
- Confidence visualization with animated progress bars
- Latency comparison per model
- Explainability: top positive/negative word extraction (Logistic Regression) and per-model reasoning
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
|---|---|---|
| Models | Single classifier | Three architectures running in parallel |
| Output | Binary label | Label + confidence + latency + reasoning |
| Comparison | None | Side-by-side model disagreement analysis |
| Architecture | Monolithic script | Multi-service (Nginx + Express + FastAPI) |
| Deployment | Notebook | Dockerized, process-managed container |

---

## System Architecture

```
Hugging Face Space (Docker SDK)

  Nginx (:7860) — static files + reverse proxy
        │
        ▼
  Express.js (:4000)                FastAPI (:8000)
  validation, rate limiting,   ──▶  model-specific preprocessing
  session mgmt, SQLite (Prisma)     Logistic Regression, Bi-LSTM, BERT

  Supervisor manages Nginx / Express / FastAPI process lifecycle
```

### Request Lifecycle

```
User submits review
        │
        ▼
   Nginx (:7860)  — static files + /api/* proxy
        │
        ▼
   Express (:4000)  — validation → rate limiting → session
        │
        ▼
   FastAPI (:8000)  — model-specific preprocessing
        │
        ├──▶ Logistic Regression (TF-IDF)   ─┐
        ├──▶ Bi-LSTM (tokenizer + pad)       ─┼──▶ Standardized Response
        └──▶ BERT (tokenizer + attention)    ─┘
                                                    │
                                                    ▼
                                          Majority Vote + Confidence
                                                    │
                                                    ▼
                                          Frontend renders results
```

---

## Key Features

**Machine Learning**

| Feature | Details |
|---|---|
| Three architectures | Logistic Regression (bag-of-words), Bi-LSTM (sequential), BERT (transformer) |
| Model-specific preprocessing | Each model receives text prepared according to its own pipeline |
| Confidence normalization | Raw probabilities mapped to a consistent 0–1 scale across models |
| Explainability | Top positive/negative words (LR) + per-model reasoning text |

**Engineering**

| Feature | Details |
|---|---|
| Multi-service architecture | Nginx, Express, FastAPI — each process owns one responsibility |
| Process orchestration | Supervisor manages all three services in a single Docker container |
| SQLite persistence | Prediction history and model metrics stored via Prisma ORM |
| Rate limiting | Express middleware protects the prediction endpoint |
| Per-model fault isolation | One model failing doesn't prevent the other two from returning results |

**User Experience**

| Feature | Details |
|---|---|
| Cinema-themed design | Gold accents, ticket-styled input, marquee borders |
| Real-time comparison | Side-by-side model outputs with confidence bars |
| Majority vote | Consensus verdict with agreement ratio (e.g. "2/3") |
| Featured Screenings | Pre-built sample reviews spanning positive/negative/mixed/sarcasm cases |
| Latency display | Per-model inference time shown alongside every prediction |

---

## Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| Frontend | HTML, CSS, JavaScript | Static UI served by Nginx |
| Reverse Proxy | Nginx | Static files, request routing, single exposed port (7860) |
| Backend API | Express.js + Prisma | Validation, routing, session management, persistence |
| ML Inference | FastAPI + Uvicorn | Async Python API for model serving |
| ML Models | scikit-learn, TensorFlow/Keras, PyTorch + Transformers | Three sentiment classifiers |
| Database | SQLite (via Prisma ORM) | Prediction history, model metrics |
| Process Manager | Supervisor | Manages Nginx/Express/FastAPI lifecycle in one container |
| Container | Docker | Consistent deployment across environments |
| Deployment | Hugging Face Spaces (Docker SDK) | Cloud hosting behind a single port |

```
# Python
numpy · pandas · scikit-learn · scipy · nltk
tensorflow · keras · torch · transformers
fastapi · uvicorn · pydantic · joblib

# Node.js
express · prisma · axios · cors · helmet
morgan · zod · express-rate-limit
```

---

## Machine Learning Pipeline

```
Raw Movie Review
       │
       ▼
Model-Specific Preprocessing
       │
   ┌───┼────┐
   ▼   ▼    ▼
TF-IDF  Tokenizer  BERT Tokenizer
   │      +Pad          │
   ▼      ▼             ▼
Logistic  Bi-LSTM      BERT
Regression (64 units)  (110M params)
   │      │             │
   └──────┼─────────────┘
          ▼
Standardized Response
{label, confidence, latency, reasoning, keywords}
          │
          ▼
Majority Vote + Verdict
```

**Training**

| Model | Training Data | Features | Epochs | Optimizer |
|---|---|---|---|---|
| Logistic Regression | IMDb 25k train | TF-IDF (50k features) | N/A | LBFGS |
| Bi-LSTM | IMDb 25k train | Tokenized sequences (max_len=300) | 10 | Adam |
| BERT | IMDb 25k train | WordPiece tokens (max_length=128) | 3 | AdamW |

**Preprocessing**

| Model | Steps |
|---|---|
| LR (TF-IDF) | Lowercase → expand contractions → strip HTML/URLs → remove special chars → stopword removal (negations preserved) → lemmatize |
| Bi-LSTM | Lowercase → expand contractions → strip HTML/URLs → keep alphanumeric + basic punctuation |
| BERT | Strip HTML → normalize whitespace (tokenizer handles the rest) |

**Dataset:** [IMDb Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) (Maas et al., Stanford AI Lab) — 50,000 reviews, balanced 25k/25k, cited from its original academic source.

---

## Experimental Results

| Model | Accuracy | Precision | Recall | F1 | Parameters | Model Size | Inference |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 90.15% | 90.10% | 90.20% | 90.10% | ~50K | ~1.5 MB | 2 ms |
| Bi-LSTM | 84.67% | 85.00% | 84.50% | 84.70% | ~120K | ~5 MB | 22 ms |
| Fine-Tuned BERT | 91.68% | 93.32% | 89.79% | 91.52% | ~110M | ~420 MB | 72 ms |

*Metrics reported on the held-out IMDb test split using the final trained models.*

**Accuracy vs. latency**

```
Accuracy (%)
 92 |                                    * BERT (91.68%, 72ms)
 90 |  * LR (90.15%, 2ms)
 88 |
 86 |
 84 |          * Bi-LSTM (84.67%, 22ms)
    +----------+----------+----------+------
               0         25         50        75
                        Latency (ms)
```

**Observations**

- Logistic Regression reaches 90.15% accuracy at 2ms — a strong speed-to-accuracy ratio for a linear model
- BERT wins on raw accuracy (91.68%) but costs **36x** more latency than LR (72ms vs. 2ms)
- Bi-LSTM underperforms both on this dataset — its added sequential complexity doesn't outweigh LR's bag-of-words efficiency here
- The ~1.5-point accuracy gap between LR and BERT may not justify a 36x latency cost for latency-sensitive applications — this tradeoff is discussed further below

---

## Key Insights

**Why Logistic Regression performs so well:** movie reviews are rich in explicit sentiment-laden keywords ("amazing," "terrible," "masterpiece"), which TF-IDF captures directly. Since much of the sentiment signal in this dataset is lexical rather than contextual, a bag-of-words model is surprisingly competitive.

**Why BERT wins where it wins:** BERT's advantage concentrates on reviews with negation ("I can't say I disliked it"), sarcasm, and mixed sentiment — cases where isolated keyword weight is actively misleading and contextual understanding changes the answer.

**Engineering observations:**
- Singleton model loading at startup avoids per-request load overhead for all three models
- Per-model fault isolation means a slow or failed BERT call doesn't block LR and LSTM from returning results
- `tf.function`-compiled inference (see [Engineering Decisions](#engineering-decisions)) cut Bi-LSTM latency by roughly 50–70x versus the naive `.predict()` call

---

## Application Showcase

### Landing Page

Cinema-themed ticket-style input, "Featured Screenings" sample reviews tagged by category and difficulty (positive, negative, mixed, sarcasm, negation, complex context).

<img src="assets/landing-page.png" alt="The Screening Room landing page with Featured Screenings" width="100%">

### Prediction Results

Overall verdict banner with majority vote and agreement ratio; three critic cards with per-model prediction, confidence bar, and latency; LR explainability panel with top positive/negative words; natural-language reasoning per model.

<img src="assets/prediction-results.png" alt="Prediction results with three-model comparison" width="100%">

### Inside the Screening Room

Accuracy/precision/recall/F1 per model, dataset provenance, technology stack, and architecture diagram.

<img src="assets/inside-screening-room.png" alt="Model metrics dashboard and architecture panel" width="100%">

### A Representative Test Case

> *"A visually stunning masterpiece that falls completely flat in the final act. The cinematography is brilliant, but the terrible script and rushed ending ruined the entire movie for me."*

All three models correctly classified this as negative, but with very different confidence — Logistic Regression (76.4%) and Bi-LSTM (67.2%) landed close to the decision boundary, while BERT (99.0%) confidently weighted the negative pivot over the earlier praise. This is the kind of disagreement the project is built to surface.

<img src="assets/model-disagreement-example.png" alt="Example of model disagreement on a mixed-sentiment review" width="100%">

---

## Repository Structure

```
sentiment-analysis/
├── frontend/
│   ├── index.html
│   ├── inside.html
│   ├── style.css
│   ├── app.js
│   └── inside.js
│
├── backend/
│   ├── src/
│   │   ├── server.js
│   │   ├── config/index.js
│   │   ├── routes/v1.routes.js
│   │   ├── controllers/predict.controller.js
│   │   ├── services/mlService.js
│   │   ├── services/dbService.js
│   │   └── middleware/  (validation, rateLimiter, errorHandler)
│   └── prisma/  (schema.prisma, seed.js)
│
├── ml_service/
│   ├── main.py
│   └── app/
│       ├── config.py
│       ├── api/routes.py
│       ├── schemas/prediction.py
│       ├── services/  (model_loader, predictor, explainer)
│       └── utils/preprocessing.py
│
├── models_ml/          # Logistic Regression + TF-IDF vectorizer
├── models_lstm/         # Bi-LSTM + tokenizer
├── models_bert/tuned/    # Fine-tuned BERT + threshold.pkl
├── data/                  # Shared tokenizer/config artifacts
│
├── assets/                # README screenshots and demo media
├── Dockerfile             # Single-container build (Python + Node + Nginx)
├── supervisord.conf
├── nginx.conf
├── hf_start.sh            # Hugging Face Spaces entrypoint
├── docker-compose.yml     # Local development only
├── requirements.txt
└── README.md
```

---

## Getting Started

**Two separate environments — don't mix them up:**
- **Local development** (`docker-compose.yml`): Frontend, Express, FastAPI, and a full **PostgreSQL** instance as separate containers.
- **Production (Hugging Face Spaces)**: a single Docker container running Nginx + Express + FastAPI under Supervisor, using **SQLite** for persistence since HF Spaces storage is ephemeral and a full Postgres instance isn't part of that deployment.

### Local development (Docker Compose)

```bash
docker-compose up --build
```

- Frontend: http://localhost:8080
- Backend API: http://localhost:4000
- ML Service: http://localhost:8000
- PostgreSQL: localhost:5432

### Local development (without Docker)

```bash
pip install -r requirements.txt
cd backend && npm install && cd ..

cd ml_service && uvicorn main:app --host 0.0.0.0 --port 8000 &
cd backend && npm start &
npx serve frontend -p 8080
```

### Production

Live at [hardik-25-sentiment-analysis.hf.space](https://hardik-25-sentiment-analysis.hf.space/) — single Docker container, Supervisor-managed, exposed on port 7860.

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/predict` | Analyze a movie review |
| `GET` | `/api/v1/health` | System health check |
| `GET` | `/api/v1/models` | List available models |
| `GET` | `/api/v1/model_metrics` | Model performance metrics |
| `GET` | `/api/v1/history/:sessionId` | Prediction history for a session |

```bash
curl -X POST http://localhost:4000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was absolutely brilliant!", "models": ["logreg", "lstm", "bert"]}'
```

---

## Engineering Decisions

**Why FastAPI for the ML service?** Native async support and Pydantic validation fit naturally with a Python ML stack, without a translation layer between the web framework and NumPy/PyTorch/TensorFlow.

**Why a separate Express gateway?** Separating validation, rate limiting, session handling, and persistence from inference keeps the ML service focused purely on prediction and lets either layer scale or be replaced independently.

**Why SQLite + Prisma in production, but PostgreSQL locally?** HF Spaces storage is ephemeral and doesn't include a managed database service, so SQLite gives zero-configuration persistence there. Locally, PostgreSQL is used to exercise the same schema against a real relational database during development. Prisma's schema stays the same across both — only the underlying engine changes.

**Why three models instead of the best one?** The point isn't a leaderboard winner — it's showing *how* different architectures read the same text differently, which is a more honest and more interesting engineering story than reporting a single number.

**Why does Logistic Regression's explainability panel differ from Bi-LSTM's and BERT's?** TF-IDF coefficients are a real, faithful measure of feature importance, so LR shows word-level attribution. Bi-LSTM and BERT don't have an equivalently cheap, faithful attribution method by default (raw attention weights are not a validated importance measure), so their panels explain *how* the architecture processes text instead of fabricating word-level scores that would misrepresent the model.

**Why did Bi-LSTM's latency drop from 1500ms+ to ~22ms?** Keras's high-level `.predict()` rebuilds parts of the computation graph on every call. Wrapping the forward pass in `@tf.function(reduce_retracing=True)` and invoking the model directly precompiles a static graph, cutting per-request latency by roughly 50–70x.

**Why Supervisor?** HF Spaces exposes a single port. Supervisor runs Nginx, Express, and FastAPI as three managed processes in one container, each with independent restart and log streams — a minimal process-orchestration setup appropriate for a single-container deployment.

---

## Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| No GPU on HF Spaces | BERT inference is CPU-bound (~72ms) | Singleton loading avoids repeated load cost; threshold tuned for the CPU-served model |
| SQLite is ephemeral in production | Data resets on Space restart | Model metrics are re-seeded at build time; prediction history is session-scoped, not meant to be durable |
| Single-container deployment | No horizontal scaling | Sufficient for a portfolio demo; the service boundaries already support splitting into separate containers later |
| Binary classification only | No neutral/multi-class support | Listed under Future Roadmap |
| English-only | No multilingual support | Listed under Future Roadmap |

---

## Future Roadmap

| Priority | Feature | Status |
|---|---|---|
| High | Multilingual sentiment analysis | Planned |
| High | Batch analysis (multiple reviews at once) | Planned |
| Medium | Additional transformer models (RoBERTa, DistilBERT) | Planned |
| Medium | Mobile-responsive layout | Planned |
| Low | Validated token-level attribution (e.g. Integrated Gradients) | Research |
| Low | Real-time streaming predictions | Research |

---

## Author

**Hardik** 

---

## License

MIT — see [LICENSE](LICENSE)

## Acknowledgements

- [IMDb Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) — Andrew L. Maas et al., Stanford AI Lab
- [Hugging Face](https://huggingface.co/) — Transformers library and Spaces hosting
- [FastAPI](https://fastapi.tiangolo.com/) · [Express.js](https://expressjs.com/) · [Prisma](https://www.prisma.io/)
