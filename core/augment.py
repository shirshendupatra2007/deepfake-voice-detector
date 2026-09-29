"""
Data Augmentation Module
Includes:
1. Phone Codec Simulation (AMR / G.711 telephony: 300Hz-3400Hz bandpass + 8kHz downsampling)
2. Additive Background Noise (Gaussian white noise & ambient noise at varying SNRs)
3. SpecAugment (Time and Frequency masking)
"""

import numpy as np
import scipy.signal

def apply_phone_codec_simulation(audio, sr=16000):
    """
    Simulates cellular telephone / PSTN audio:
    - 300 Hz - 3400 Hz telephony bandpass filter
    - Downsampling to 8000 Hz and upsampling back to 16000 Hz
    - Slight quantization / dynamic range compression
    """
    # 4th order Butterworth bandpass (300Hz - 3400Hz)
    nyquist = 0.5 * sr
    low = 300.0 / nyquist
    high = min(3400.0 / nyquist, 0.95)
    b, a = scipy.signal.butter(4, [low, high], btype='band')
    filtered = scipy.signal.lfilter(b, a, audio)

    # Decimate to 8kHz and interpolate back to 16kHz
    decimated = scipy.signal.resample(filtered, len(filtered) // 2)
    telephony = scipy.signal.resample(decimated, len(filtered))

    # Mu-law / telephony dynamic range simulation
    mu = 255.0
    telephony = np.sign(telephony) * np.log1p(mu * np.abs(telephony)) / np.log1p(mu)
    return telephony.astype(np.float32)

def add_noise(audio, snr_db_range=(10.0, 30.0)):
    """
    Adds Gaussian white noise with random SNR between snr_db_range.
    """
    snr_db = np.random.uniform(snr_db_range[0], snr_db_range[1])
    audio_power = np.mean(audio ** 2) + 1e-10
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = audio_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
    return (audio + noise).astype(np.float32)

def apply_spec_augment(feature_map, max_freq_mask=8, max_time_mask=12):
    """
    Applies SpecAugment on feature map (channels, freq_bins, time_frames).
    Randomly masks frequency bands and time slices.
    """
    augmented = feature_map.copy()
    c, f_bins, t_frames = augmented.shape

    # Frequency masking
    f_width = np.random.randint(1, max_freq_mask + 1)
    f0 = np.random.randint(0, f_bins - f_width)
    augmented[:, f0 : f0 + f_width, :] = 0.0

    # Time masking
    t_width = np.random.randint(1, max_time_mask + 1)
    t0 = np.random.randint(0, t_frames - t_width)
    augmented[:, :, t0 : t0 + t_width] = 0.0

    return augmented
