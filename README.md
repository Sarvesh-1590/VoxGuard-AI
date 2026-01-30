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

The backend is configured with CORS to allow connections from any origin (`*`), making it compatible with frontend tools like v0.dev.
Connect your frontend to `http://127.0.0.1:8000/analyze-audio`.
