"""
Audio Processor and Acoustic Forensics Analyzer
Extracts:
1. Feature tensor: Log-Mel Spectrogram + Modified Group Delay Phase (2, 64, 100)
2. Biometric Acoustics:
   - Pitch Jitter & Micro-prosody (natural human vocal fold vibration vs synthetic AI constancy)
   - Phase Discontinuity Index (neural vocoder Griffin-Lim/HiFi-GAN artifact)
   - High-Frequency Vocoder Brickwall Cutoff (7-8 kHz energy drop)
   - Spectral Flatness & Harmonic Regularity
3. Computes calibrated Forensic Spoof Probability (0..1)
"""

import time
import io
import numpy as np
import soundfile as sf
import librosa
from core.features import extract_combined_features, SAMPLE_RATE, CHUNK_SAMPLES

import scipy.io.wavfile

def load_audio_from_bytes(audio_bytes):
    """
    Decodes audio bytes (WAV, WebM, Opus, MP3, OGG, FLAC, M4A) into 16kHz float32 numpy array.
    Supports native soundfile, scipy wavfile, and universal FFmpeg streaming decoding.
    """
    # 1. Try soundfile (handles standard WAV, MP3, OGG, FLAC)
    try:
        audio_stream = io.BytesIO(audio_bytes)
        data, sr = sf.read(audio_stream)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if sr != SAMPLE_RATE:
            data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLE_RATE)
        return data.astype(np.float32)
    except Exception:
        pass

    # 2. Try scipy.io.wavfile (handles PCM WAV)
    try:
        audio_stream = io.BytesIO(audio_bytes)
        sr, data = scipy.io.wavfile.read(audio_stream)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if sr != SAMPLE_RATE:
            data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLE_RATE)
        return data.astype(np.float32)
    except Exception:
        pass

    # 3. Universal Fallback via FFmpeg (decodes browser WebM/Opus, M4A, AAC, etc.)
    try:
        import subprocess
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe, '-i', 'pipe:0',
            '-f', 's16le', '-acodec', 'pcm_s16le',
            '-ac', '1', '-ar', str(SAMPLE_RATE),
            'pipe:1'
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        raw_pcm, _ = proc.communicate(input=audio_bytes)
        if len(raw_pcm) > 0:
            data = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32) / 32768.0
            return data
    except Exception as e:
        print(f"[AudioDecoder] FFmpeg fallback error: {e}")

    raise ValueError("Could not decode audio data: format not recognized.")

def compute_pitch_jitter(audio_data, sr=SAMPLE_RATE):
    """
    Computes pitch period micro-jitter (%):
    - Real humans cannot maintain perfect mechanical pitch: natural jitter is typically 0.8% - 3.5%.
    - AI synthesizers (ElevenLabs, OpenAI, Tacotron, VITS) have near-zero jitter (typically < 0.45%).
    """
    try:
        # Use YIN algorithm with pitch bounds for human speech (65Hz - 400Hz)
        f0 = librosa.yin(audio_data, fmin=65, fmax=400, sr=sr)
        valid = f0[~np.isnan(f0) & (f0 > 65) & (f0 < 400)]
        if len(valid) < 8:
            return 1.2 # Default neutral fallback if unvoiced
        
        periods = 1.0 / valid
        period_diffs = np.abs(np.diff(periods))
        jitter_pct = (np.mean(period_diffs) / (np.mean(periods) + 1e-6)) * 100.0
        return float(np.clip(jitter_pct, 0.05, 35.0))
    except Exception:
        return 1.2

def analyze_audio_forensics(audio_data, sr=SAMPLE_RATE):
    """
    Computes explainability forensics and an acoustic forensic spoof probability.
    """
    if len(audio_data) < 512:
        return {
            "phase_irregularity": 50.0,
            "vocoder_cutoff_detected": False,
            "pitch_jitter_pct": 1.2,
            "harmonic_regularity": 50.0,
            "mean_rolloff_hz": 4000.0,
            "forensic_spoof_prob": 0.5,
            "cues": ["Audio chunk too short for forensic evaluation."]
        }

    # 1. Pitch Jitter (Human micro-prosody)
    jitter = compute_pitch_jitter(audio_data, sr=sr)

    # 2. STFT Analysis for Phase & High-Frequency Cutoff
    D = librosa.stft(audio_data, n_fft=512, hop_length=160)
    mag, phase = np.abs(D), np.angle(D)

    # Phase irregularity via second difference of unwrapped phase
    phase_diff = np.diff(phase, axis=1)
    phase_diff2 = np.diff(phase_diff, axis=1)
    phase_std = float(np.std(phase_diff2))
    # Normalized 0..100
    phase_irregularity = float(np.clip((phase_std - 3.5) * 45.0, 15.0, 95.0))

    # 3. High-Frequency Vocoder Brickwall Cutoff
    # Vocoders (16kHz or 22kHz models) sharply cut off above 7.0 - 7.5 kHz
    total_energy = float(np.sum(mag)) + 1e-6
    high_bins = int(512 * 7000 / sr) # Bins above 7000 Hz
    high_energy = float(np.sum(mag[high_bins:, :]))
    high_ratio = high_energy / total_energy
    vocoder_cutoff = high_ratio < 0.0011

    # 4. Spectral Roll-off
    rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr, roll_percent=0.85)
    mean_rolloff = float(np.mean(rolloff))

    # 5. Spectral Flatness
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=audio_data)))
    harmonic_regularity = float(np.clip((1.0 - flatness * 2.5) * 100.0, 15.0, 95.0))

    # Calculate Acoustic Forensic Spoof Score (0.0 = Bonafide Real, 1.0 = Spoofed AI)
    spoof_score = 0.0

    # Jitter indicator:
    # If jitter < 0.6% -> highly likely AI synthetic voice
    # If jitter > 1.2% -> highly characteristic of real human vocal cord physiology
    if jitter < 0.35:
        spoof_score += 0.50
    elif jitter < 0.65:
        spoof_score += 0.35
    elif jitter > 2.0:
        spoof_score -= 0.35
    elif jitter > 1.0:
        spoof_score -= 0.20

    # Vocoder Cutoff indicator:
    if vocoder_cutoff:
        spoof_score += 0.30
    else:
        spoof_score -= 0.15

    # Phase irregularity indicator:
    if phase_irregularity > 65.0:
        spoof_score += 0.20
    elif phase_irregularity < 40.0:
        spoof_score -= 0.15

    # Base probability centered at 0.5
    forensic_prob = float(np.clip(0.5 + spoof_score, 0.03, 0.97))

    # Diagnostic cues
    cues = []
    if jitter < 0.65:
        cues.append(f"Unnaturally flat pitch micro-prosody (Jitter: {jitter:.2f}%): characteristic of AI acoustic generators lacking human vocal fold tremor.")
    else:
        cues.append(f"Authentic vocal fold perturbation detected (Micro-jitter: {jitter:.2f}%): matches natural human larynx physiology.")

    if vocoder_cutoff:
        cues.append("High-frequency boundary attenuation consistent with neural vocoder synthesis cutoffs.")
    else:
        cues.append("Acoustic energy decays continuously without artificial vocoder bandpass attenuation.")

    if phase_irregularity > 60.0:
        cues.append("Abnormal group delay dispersion observed in harmonic transitions (vocoder phase jitter).")
    else:
        cues.append("Continuous natural vocal tract phase trajectory observed.")

    return {
        "phase_irregularity": round(phase_irregularity, 1),
        "vocoder_cutoff_detected": bool(vocoder_cutoff),
        "pitch_jitter_pct": round(jitter, 2),
        "harmonic_regularity": round(harmonic_regularity, 1),
        "mean_rolloff_hz": round(mean_rolloff, 1),
        "forensic_spoof_prob": round(forensic_prob, 3),
        "cues": cues
    }

def process_audio_chunk(audio_data, sr=SAMPLE_RATE):
    """
    Extracts neural network input tensor and forensic metrics.
    """
    t0 = time.perf_counter()

    # Extract 2-channel features (Log-Mel + Modified Group Delay)
    features = extract_combined_features(audio_data, sr=sr, target_frames=100)
    batch_features = np.expand_dims(features, axis=0) # (1, 2, 64, 100)

    # Forensic Analysis
    forensics = analyze_audio_forensics(audio_data, sr=sr)

    extraction_time_ms = (time.perf_counter() - t0) * 1000.0

    # Downsampled waveform for UI
    step = max(1, len(audio_data) // 100)
    waveform_preview = [round(float(x), 4) for x in audio_data[::step][:100]]

    # Spectrogram preview (Log-Mel, 32x50)
    mel_channel = features[0]
    spec_preview = [
        [round(float(val), 2) for val in row[::2]]
        for row in mel_channel[::2]
    ]

    # MGD phase preview (32x50)
    mgd_channel = features[1]
    mgd_preview = [
        [round(float(val), 2) for val in row[::2]]
        for row in mgd_channel[::2]
    ]

    return {
        "feature_tensor": batch_features,
        "forensics": forensics,
        "extraction_time_ms": round(extraction_time_ms, 2),
        "waveform": waveform_preview,
        "spectrogram": spec_preview,
        "mgd_spectrum": mgd_preview
    }
