from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import audio_utils
from model import ModelHandler
import io
import shutil
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Voice Spoofing Detection API",
    description="Local API for detecting genuine vs spoofed audio using Deep Learning (PyTorch).",
    version="1.0.0"
)

# CORS Middleware (Allow all for local development flexibility)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Model Handler (will load weights or initialize random on startup)
model_handler = ModelHandler()

@app.get("/health")
async def health_check():
    """
    Health check endpoint to ensure API is running.
    """
    status = "healthy"
    if model_handler.model is None:
        status = "degraded (model not loaded)"
    return {"status": status, "service": "Voice Spoofing Detector"}

@app.post("/analyze-audio")
async def analyze_audio(file: UploadFile = File(...)):
    """
    Analyzes an uploaded audio file for signs of spoofing.
    
    Expected input: Audio file (.wav, .mp3, .flac)
    Returns: JSON with label (Genuine/Spoofed) and confidence score.
    """
    logger.info(f"Received file: {file.filename}, content_type: {file.content_type}")
    
    # 1. Validate file type (basic check)
    allowed_types = ["audio/wav", "audio/mpeg", "audio/x-wav", "audio/flac", "application/octet-stream"] 
    # specific mimetypes vary often, so we'll try to process anything and fail gracefully if librosa can't read it
    
    try:
        # 2. Read file content
        file_content = await file.read()
        file_obj = io.BytesIO(file_content)
        
        # 3. Load and Preprocess Audio
        try:
            raw_audio = audio_utils.load_audio(file_obj)
        except Exception as e:
            logger.error(f"Audio loading failed: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid audio file. Could not load audio data. Error: {str(e)}")
            
        processed_audio = audio_utils.preprocess_audio(raw_audio)
        
        # 4. Extract Features
        features = audio_utils.extract_features(processed_audio)
        mfcc_features = features['mfcc']
        
        # 5. Inference
        result = model_handler.predict(mfcc_features)
        
        logger.info(f"Analysis complete for {file.filename}: {result}")
        
        return JSONResponse(content={
            "filename": file.filename,
            "analysis": result
        })
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Internal processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error processing audio: {str(e)}")

# Safe startup event to show where to access docs
@app.on_event("startup")
async def startup_event():
    logger.info("Starting up Voice Spoofing Detection API...")
    logger.info("Docs available at /docs")

if __name__ == "__main__":
    import uvicorn
    # Use standard host/port for local dev
    uvicorn.run(app, host="127.0.0.1", port=8000)
