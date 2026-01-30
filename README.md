🎙️ VoxGuard-AI
📌 Overview

The Voice Spoofing Detection System is a local, AI-based application designed to distinguish between genuine human speech and spoofed / synthetic audio (such as deepfake voices generated using TTS or voice conversion models).

With the rapid growth of AI-generated speech, voice spoofing poses serious risks to:

Voice-based authentication systems

Call center security

Digital identity verification

Financial fraud prevention

This project demonstrates a complete end-to-end deep learning pipeline, from audio ingestion to real-time prediction, exposed through a web API and integrated with a frontend interface.

🧠 Problem Statement

Modern speech synthesis and voice cloning technologies can generate highly realistic human-like voices, making it increasingly difficult to distinguish between real and fake audio.

Objective:
Build a system that can automatically analyze an input audio file and classify it as:

Genuine (real human speech)

Spoofed (synthetic / manipulated speech)

The system should:

Run locally

Provide real-time predictions

Be extensible to support advanced pretrained models

⚙️ System Architecture

User uploads an audio file via the frontend

Backend API receives the file

Audio is preprocessed and converted into MFCC features

Features are passed to a deep learning classifier

The system returns:

Predicted label (Genuine / Spoofed)

Confidence score

voice_spoofing_detection/
├── backend/
│   ├── app.py              # FastAPI application
│   ├── model.py            # PyTorch model definition
│   ├── audio_utils.py      # Audio preprocessing & feature extraction
│   ├── requirements.txt    # Python dependencies
│   └── weights/
│       └── best_model.pth  # Pre-trained model weights
├── README.md               # This file
└── test_pipeline.py        # Testing script
```

🛠️ Tech Stack (Minimal & Purpose-Driven)
Backend

Python 3.9+

FastAPI – high-performance REST API framework

Uvicorn – ASGI server

PyTorch – deep learning framework

Librosa – audio processing and MFCC extraction

NumPy – numerical computation

Frontend

v0 (Vercel) – UI generation and deployment

HTML / CSS / JavaScript

🚀 Setup & Running (Local)
🔹 Prerequisites

Python 3.9 or higher

pip (Python package manager)

Git (optional)

1️⃣ Backend Setup

Navigate to the backend directory:

cd backend


Create a virtual environment:

python -m venv venv


Activate the virtual environment:

Windows

.\venv\Scripts\activate


Mac / Linux

source venv/bin/activate


Install dependencies:

pip install -r requirements.txt

2️⃣ Run the Server

Start the FastAPI server:

uvicorn app:app --reload


The backend will be available at:

http://127.0.0.1:8000


Swagger API documentation:

http://127.0.0.1:8000/docs

🔌 API Endpoints
🔹 Health Check

GET /health
Checks if the API and model are running.

🔹 Audio Analysis

POST /analyze-audio

Input

Form-data

file: audio file (.wav recommended)

Output

{
  "label": "Genuine",
  "confidence": 0.54,
  "details": {
    "genuine_prob": 0.54,
    "spoofed_prob": 0.46
  }
}

🌐 Frontend Integration

Frontend UI:

https://v0-voice-spoofing-detection-ui.vercel.app/

Connecting the Frontend

Since the frontend runs on HTTPS and the backend runs locally on HTTP, one of the following approaches must be used:

✅ Option 1: Allow Mixed Content (Quick Demo)

Run backend locally

Open frontend URL

Allow insecure content in browser settings (shield icon)
