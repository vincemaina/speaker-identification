import numpy as np
import scipy.signal
import scipy.fftpack
import scipy.stats
from scipy.io import wavfile
from flask import Flask
app = Flask(__name__)

# Constants
SAMPLE_RATE = 16000
LOW_PERCENTILE = 10
HIGH_PERCENTILE = 90

def extract_features(audio_file):
    """Extracts spectral features from an audio file without using Librosa."""
    
    # ✅ Load audio file
    sr, y = wavfile.read(audio_file)
    
    # Ensure audio is mono (convert stereo to mono if needed)
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)
    
    # Compute FFT (Short-Time Fourier Transform)
    f, t, Sxx = scipy.signal.spectrogram(y, fs=sr, nperseg=1024)

    # ✅ Spectral Centroid: Weighted mean of the frequencies
    spectral_centroid = np.sum(f[:, None] * Sxx, axis=0) / np.sum(Sxx, axis=0)

    # ✅ Spectral Bandwidth: Standard deviation of the spectrum
    spectral_bandwidth = np.sqrt(np.sum(((f[:, None] - spectral_centroid[None, :])**2) * Sxx, axis=0) / np.sum(Sxx, axis=0))

    # ✅ Spectral Flatness: Measures how noise-like the signal is
    spectral_flatness = scipy.stats.gmean(Sxx + 1e-10, axis=0) / np.mean(Sxx, axis=0)

    # ✅ Spectral Rolloff: Frequency below which 85% of the spectrum’s power resides
    spectral_rolloff = np.array([f[np.argmax(np.cumsum(Sxx[:, i]) >= 0.85 * np.sum(Sxx[:, i]))] for i in range(Sxx.shape[1])])

    # ✅ Spectral Contrast: Difference between peaks and valleys of the spectrum
    spectral_contrast = np.abs(np.max(Sxx, axis=0) - np.min(Sxx, axis=0))

    # ✅ Compute feature statistics (mean, std, percentiles)
    stats = {
        "mean_spectral_centroid": np.mean(spectral_centroid),
        "mean_spectral_bandwidth": np.mean(spectral_bandwidth),
        "mean_spectral_flatness": np.mean(spectral_flatness),
        "mean_spectral_rolloff": np.mean(spectral_rolloff),
        "mean_spectral_contrast": np.mean(spectral_contrast),
        
        "std_spectral_centroid": np.std(spectral_centroid),
        "std_spectral_bandwidth": np.std(spectral_bandwidth),
        "std_spectral_flatness": np.std(spectral_flatness),
        "std_spectral_rolloff": np.std(spectral_rolloff),
        "std_spectral_contrast": np.std(spectral_contrast),
        
        "low_spectral_centroid": np.percentile(spectral_centroid, LOW_PERCENTILE),
        "low_spectral_bandwidth": np.percentile(spectral_bandwidth, LOW_PERCENTILE),
        "low_spectral_flatness": np.percentile(spectral_flatness, LOW_PERCENTILE),
        "low_spectral_rolloff": np.percentile(spectral_rolloff, LOW_PERCENTILE),
        "low_spectral_contrast": np.percentile(spectral_contrast, LOW_PERCENTILE),
        
        "high_spectral_centroid": np.percentile(spectral_centroid, HIGH_PERCENTILE),
        "high_spectral_bandwidth": np.percentile(spectral_bandwidth, HIGH_PERCENTILE),
        "high_spectral_flatness": np.percentile(spectral_flatness, HIGH_PERCENTILE),
        "high_spectral_rolloff": np.percentile(spectral_rolloff, HIGH_PERCENTILE),
        "high_spectral_contrast": np.percentile(spectral_contrast, HIGH_PERCENTILE),
    }

    return stats

@app.route("/api/python")
def hello_world():
    return "<p>Hello, World!</p>"
