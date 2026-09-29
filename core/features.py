"""
Feature Extraction Module for Real-Time Deepfake Voice Detection
Extracts:
1. Log-Mel Spectrogram (Magnitude spectral energy across perceptual frequencies)
2. Modified Group Delay (MGD) (Phase-derivative feature revealing vocoder phase distortion)
"""

import numpy as np
import librosa
import scipy.signal

SAMPLE_RATE = 16000
N_FFT = 512          # 32 ms window at 16kHz (satisfies 20-30ms requirement)
HOP_LENGTH = 160     # 10 ms hop length
N_MELS = 64          # Mel filterbanks
CHUNK_DURATION = 1.0 # 1 second decision chunk
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION) # 16000 samples

def compute_log_mel_spectrogram(y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS):
    """
    Computes Log-Mel Spectrogram using librosa.
    Output shape: (n_mels, time_frames)
    """
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=20,
        fmax=8000,
        power=2.0
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    # Normalize to zero mean, unit variance for neural network stability
    mean = np.mean(log_mel)
    std = np.std(log_mel) + 1e-6
    return (log_mel - mean) / std

def compute_modified_group_delay(y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS):
    """
    Computes Modified Group Delay (MGD) phase representation.
    Group Delay tau(w) = -d/dw(arg(X(w)))
    Vocoders (HiFi-GAN, MelGAN, Tacotron, VITS) fail to reproduce authentic
    phase consistency, leaving abnormal group delay patterns.
    """
    # Pad signal if necessary
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))
    
    # x(n) and n * x(n)
    n = np.arange(len(y))
    ny = n * y

    # STFT of y and n*y
    X = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hann')
    Y = librosa.stft(ny, n_fft=n_fft, hop_length=hop_length, window='hann')

    X_real, X_imag = np.real(X), np.imag(X)
    Y_real, Y_imag = np.real(Y), np.imag(Y)

    # Group delay formula: (X_R * Y_R + X_I * Y_I) / (|X|^2 + eps)
    numerator = X_real * Y_real + X_imag * Y_imag
    denominator = (X_real ** 2 + X_imag ** 2) ** 0.5 + 1e-6

    # Modified group delay with gamma exponent to prevent peak blowing
    gamma = 0.4
    mgd = numerator / (denominator ** (2 * gamma))
    
    # Compress frequency axis using Mel scale to match mel spectrogram dimension
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=20, fmax=8000)
    mgd_mel = np.dot(mel_basis, np.abs(mgd))
    
    # Log compression and normalization
    mgd_mel = np.log1p(np.maximum(mgd_mel, 0))
    mean = np.mean(mgd_mel)
    std = np.std(mgd_mel) + 1e-6
    return (mgd_mel - mean) / std

def extract_combined_features(audio_data, sr=SAMPLE_RATE, target_frames=100):
    """
    Extracts 2-channel feature map:
    Channel 0: Log-Mel Spectrogram
    Channel 1: Modified Group Delay (Phase)
    Returns: numpy array of shape (2, n_mels, target_frames)
    """
    # Convert to mono if stereo
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Resample if sample rate differs
    if sr != SAMPLE_RATE:
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE

    # Ensure length is at least 1 chunk (16000 samples)
    if len(audio_data) < CHUNK_SAMPLES:
        audio_data = np.pad(audio_data, (0, CHUNK_SAMPLES - len(audio_data)), mode='wrap')
    elif len(audio_data) > CHUNK_SAMPLES:
        # Use first chunk or central chunk
        start = (len(audio_data) - CHUNK_SAMPLES) // 2
        audio_data = audio_data[start : start + CHUNK_SAMPLES]

    log_mel = compute_log_mel_spectrogram(audio_data, sr=sr)
    mgd = compute_modified_group_delay(audio_data, sr=sr)

    # Pad or trim to target_frames (default 100 frames = 1.0 second with hop=160)
    def fix_frames(feat, target):
        if feat.shape[1] < target:
            return np.pad(feat, ((0, 0), (0, target - feat.shape[1])), mode='constant')
        return feat[:, :target]

    log_mel = fix_frames(log_mel, target_frames)
    mgd = fix_frames(mgd, target_frames)

    # Shape: (2, 64, target_frames)
    combined = np.stack([log_mel, mgd], axis=0).astype(np.float32)
    return combined
