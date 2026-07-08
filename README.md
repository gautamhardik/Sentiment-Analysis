---
title: Sentiment Analysis
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# The Screening Room

> **Three Critics. One Review. Three Perspectives.**

The Screening Room is an AI-powered movie review analysis platform that compares how three fundamentally different machine learning architectures interpret the same text.

1. **The Statistician (Logistic Regression + TF-IDF):** Traditional bag-of-words approach.
2. **The Sequentialist (Bi-LSTM):** Deep learning with sequential reading.
3. **The Contextualist (Fine-Tuned BERT):** Transformer with bidirectional context.

## Architecture

- **Frontend:** Static HTML/CSS/JS served via Nginx (port 7860 on HF Spaces)
- **Backend API Gateway:** Express.js for validation, routing, orchestration (port 4000)
- **ML Inference Service:** FastAPI Python for ML model inference (port 8000)
- **Orchestration:** Supervisor manages all processes in a single container

## Deploy on Hugging Face Spaces

This app uses a **Docker Space** to run all services (Nginx + Express + FastAPI) behind a single port.

## Local Development

```bash
docker-compose up --build
```
