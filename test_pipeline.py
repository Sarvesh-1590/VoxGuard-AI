import sys
import os
import numpy as np
import io
import soundfile as sf
import logging

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

try:
    import audio_utils
    from model import ModelHandler
except ImportError as e:
    print(f"Error importing backend modules: {e}")
    print("Make sure you have installed requirements: pip install -r backend/requirements.txt")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Verification")

def create_dummy_audio(filename="test_audio.wav", duration=3.0, sr=16000):
    """Generates a dummy sine wave audio file."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    x = 0.5 * np.sin(2 * np.pi * 440 * t) # 440Hz sine wave
    sf.write(filename, x, sr)
    return filename

def test_pipeline():
    logger.info("Starting Pipeline Verification...")
    
    # 1. Create Dummy Audio
    audio_file = create_dummy_audio()
    logger.info(f"Created dummy audio: {audio_file}")
    
    try:
        # 2. Simulate Loading
        logger.info("Step 1: Loading Audio...")
        try:
            # We treat the file path as a file-like object for consistency validation or just pass path
            # librosa handles paths fine. `audio_utils.load_audio` expects file_like usually coming from API
            # but let's see if we can pass open file
            with open(audio_file, 'rb') as f:
                # We need to read it into BytesIO to mimic UploadFile.file behaviour slightly
                # or just modify load_audio to accept paths.
                # The current load_audio uses librosa.load which accepts both.
                raw_audio = audio_utils.load_audio(f)
            logger.info(f"Audio loaded. Shape: {raw_audio.shape}")
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            return

        # 3. Preprocess
        logger.info("Step 2: Preprocessing...")
        processed_audio = audio_utils.preprocess_audio(raw_audio)
        logger.info(f"Audio preprocessed. Shape: {processed_audio.shape}")
        
        # 4. Feature Extraction
        logger.info("Step 3: Feature Extraction...")
        features = audio_utils.extract_features(processed_audio)
        mfcc = features['mfcc']
        logger.info(f"MFCCs extracted. Shape: {mfcc.shape}")
        
        # 5. Model Inference
        logger.info("Step 4: Model Inference...")
        handler = ModelHandler() # Should initialize with random weights if no file found
        result = handler.predict(mfcc)
        
        logger.info("-" * 30)
        logger.info("SUCCESS: Pipeline executed successfully.")
        logger.info(f"Prediction Result: {result}")
        logger.info("-" * 30)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
    finally:
        # Cleanup
        if os.path.exists(audio_file):
            os.remove(audio_file)
            logger.info("Cleaned up test file.")

if __name__ == "__main__":
    test_pipeline()
