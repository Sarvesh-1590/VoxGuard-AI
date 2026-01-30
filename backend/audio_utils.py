import librosa
import numpy as np
import io
import soundfile as sf

# Constants
SAMPLE_RATE = 16000
DURATION = 5  # Duration in seconds to analyze
N_MFCC = 40
HOP_LENGTH = 512
N_FFT = 2048

def load_audio(file_like) -> np.ndarray:
    """
    Loads audio from a file-like object and resamples it to SAMPLE_RATE.
    
    Args:
        file_like: A file-like object (e.g., BytesIO) containing audio data.
        
    Returns:
        np.ndarray: The audio time series.
    """
    try:
        # Load audio using soundfile first to handle file-like objects better if librosa struggles
        # But librosa.load supports file paths or file-like objects in newer versions
        # We'll use librosa directly for consistency, forcing mono and specific samplerate
        y, sr = librosa.load(file_like, sr=SAMPLE_RATE, mono=True)
        return y
    except Exception as e:
        raise ValueError(f"Failed to load audio: {str(e)}")

def preprocess_audio(y: np.ndarray) -> np.ndarray:
    """
    Preprocesses audio: Trims silence and ensures fixed length.
    
    Args:
        y (np.ndarray): Audio time series.
        
    Returns:
        np.ndarray: Preprocessed audio fixed to DURATION * SAMPLE_RATE.
    """
    # 1. Trim leading/trailing silence
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    
    # 2. Fix length (pad or truncate)
    target_length = int(DURATION * SAMPLE_RATE)
    if len(y_trimmed) < target_length:
        # Pad with zeros
        padding = target_length - len(y_trimmed)
        y_fixed = np.pad(y_trimmed, (0, padding), 'constant')
    else:
        # Truncate
        y_fixed = y_trimmed[:target_length]
        
    return y_fixed

def extract_features(y: np.ndarray) -> dict:
    """
    Extracts features from the audio for the model.
    Using MFCCs as the primary feature set for this baseline.
    
    Args:
        y (np.ndarray): Audio time series.
        
    Returns:
        dict: Dictionary containing extracted features.
              Mainly 'mfcc' which will be used for inference.
    """
    # Compute MFCCs
    mfcc = librosa.feature.mfcc(
        y=y, 
        sr=SAMPLE_RATE, 
        n_mfcc=N_MFCC, 
        n_fft=N_FFT, 
        hop_length=HOP_LENGTH
    )
    
    # Also extract some spectral features for potential metadata or advanced models
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH)
    
    return {
        "mfcc": mfcc, # Shape: (n_mfcc, time_steps)
        "spectral_centroid": spectral_centroid
    }
