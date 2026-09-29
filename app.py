"""
FastAPI Server for Real-Time Deepfake Voice Detection
Serves:
- REST API for audio file and stream chunk analysis
- Preloaded sample audio files for hackathon judges
- Live benchmark metrics dashboard
- Modern, responsive Web UI
"""

import os
import json
import io
import webbrowser
import threading
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from core.audio_processor import load_audio_from_bytes, process_audio_chunk
from core.inference import DeepfakeVoiceInference

app = FastAPI(title="Deepfake Voice Detector API", version="1.0.0")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
BENCHMARK_PATH = os.path.join(BASE_DIR, "benchmark_results.json")

# Initialize Inference Engine
onnx_file = os.path.join(MODELS_DIR, "deepfake_detector.onnx")
pth_file = os.path.join(MODELS_DIR, "deepfake_detector.pth")
inference_engine = DeepfakeVoiceInference(onnx_path=onnx_file, pth_path=pth_file)

# Mount static, samples, and models folders
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")
app.mount("/models", StaticFiles(directory=MODELS_DIR), name="models")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    root_index = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(root_index):
        with open(root_index, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    index_file = os.path.join(STATIC_DIR, "index.html")
    with open(index_file, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/api/benchmarks")
async def get_benchmarks():
    """Returns verified benchmarks directly from disk."""
    if os.path.exists(BENCHMARK_PATH):
        with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Benchmark results not found")

@app.get("/api/samples/{filename}")
async def get_sample_audio(filename: str):
    """Serves quick-test sample audio files for judges."""
    file_path = os.path.join(SAMPLES_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/wav")
    raise HTTPException(status_code=404, detail="Sample audio not found")

@app.get("/api/samples-list")
async def get_samples_list():
    """Lists available sample audio clips."""
    samples = [
        {
            "id": "sample_real_studio.wav",
            "name": "Sample 1: Real Human Voice (Studio Mic)",
            "expected": "Real",
            "description": "Natural human speech with authentic vocal tract formants and smooth glottal phase."
        },
        {
            "id": "sample_real_phone.wav",
            "name": "Sample 2: Real Human Voice (Phone Call Codec)",
            "expected": "Real",
            "description": "Human speech passing through AMR-NB/G.711 telephony bandpass (300-3400Hz)."
        },
        {
            "id": "sample_ai_neural_tts.wav",
            "name": "Sample 3: AI Voice (Neural TTS Vocoder)",
            "expected": "Likely AI",
            "description": "Synthetic voice exhibiting phase jitter, high-frequency cutoff, and formant rigidity."
        },
        {
            "id": "sample_ai_voice_clone.wav",
            "name": "Sample 4: AI Voice (Voice Conversion Clone)",
            "expected": "Likely AI",
            "description": "AI-cloned voice with synthetic phase dispersion and unnatural pitch micro-prosody."
        }
    ]
    return samples

@app.post("/api/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    """
    Analyzes an uploaded audio file or recorded microphone blob.
    Extracts Log-Mel Spectrogram & Modified Group Delay, runs ONNX/PyTorch inference.
    """
    try:
        contents = await file.read()
        audio_data = load_audio_from_bytes(contents)

        if len(audio_data) == 0:
            raise HTTPException(status_code=400, detail="Empty audio content")

        # Process audio features & explainability forensics
        processed = process_audio_chunk(audio_data)

        # Run Neural Inference with Forensic Acoustic Calibration
        pred = inference_engine.predict(processed["feature_tensor"], forensic_info=processed["forensics"])

        total_latency = round(processed["extraction_time_ms"] + pred["inference_time_ms"], 2)

        return {
            "success": True,
            "verdict": pred["label"],
            "is_spoof": pred["is_spoof"],
            "confidence_percent": pred["confidence"],
            "probabilities": {
                "real_human": pred["real_prob"],
                "likely_ai": pred["spoof_prob"]
            },
            "latency": {
                "feature_extraction_ms": processed["extraction_time_ms"],
                "model_inference_ms": pred["inference_time_ms"],
                "total_system_latency_ms": total_latency
            },
            "backend": pred["backend"],
            "forensics": processed["forensics"],
            "visualizations": {
                "waveform": processed["waveform"],
                "spectrogram": processed["spectrogram"],
                "mgd_spectrum": processed["mgd_spectrum"]
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

def run_server(port=8000, open_browser=True):
    if open_browser:
        threading.Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")

if __name__ == '__main__':
    run_server()
