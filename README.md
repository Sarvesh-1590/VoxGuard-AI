# Voice Spoofing Detection System

A local AI-based system to detect legitimate vs. spoofed (deepfake) audio using Deep Learning.

## Project Structure

```
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

## Setup & Running (Local)

### Prerequisites
- Python 3.9+ installed
- Git (optional, for version control)

### 1. Set up the Backend

Navigate to the backend directory:
```bash
cd backend
```

Create a virtual environment:
```bash
python -m venv venv
```

Activate the virtual environment:
- **Windows**:
  ```powershell
  .\venv\Scripts\activate
  ```
- **Mac/Linux**:
  ```bash
  source venv/bin/activate
  ```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the Server

Start the FastAPI server:
```bash
uvicorn app:app --reload
```
The API will be available at `http://127.0.0.1:8000`.

### 3. API Endpoints

- **GET /health**: Check if the API and model are running.
- **POST /analyze-audio**: Upload an audio file for analysis.
  - **Input**: Form-data with key `file` (audio file).
  - **Output**: JSON with `label` (Genuine/Spoofed) and `confidence`.

## Frontend Integration

This project is designed to work with the **VoxGuard AI** frontend.

**Frontend URL**: [https://v0-voice-spoofing-detection-ui.vercel.app/](https://v0-voice-spoofing-detection-ui.vercel.app/)

### Connecting the Frontend
The frontend is hosted on Vercel (HTTPS), while your backend runs locally (HTTP). To allow them to communicate, you have two options:

#### Option 1: Live Frontend (Requires Mixed Content Allowance)
1.  Run the backend: `uvicorn app:app --reload`
2.  Open the [Frontend URL](https://v0-voice-spoofing-detection-ui.vercel.app/).
3.  **Important**: Since the backend is HTTP and frontend is HTTPS, your browser might block the request. You may need to:
    -   Click the "shield" icon in the URL bar and "Allow Unsafe Scripts" or "Allow Insecure Content".
    -   OR use a tunneling service like [ngrok](https://ngrok.com/) to expose your local server as HTTPS:
        ```bash
        ngrok http 8000
        ```
        Then, if the frontend supports changing the API URL, point it to the ngrok URL.

#### Option 2: Run Frontend Locally (Recommended)
If you have the frontend code (export from v0), run it locally to avoid HTTPS/HTTP issues.
