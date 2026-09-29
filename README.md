# AURA SHIELD // Real-Time Deepfake Voice Defense
### Hackathon Submission & Live Judge Demonstration System

A lightweight, real-time deepfake voice detection system engineered to detect AI voice clones (ElevenLabs, TTS, Voice Conversion, VITS, HiFi-GAN) within **1 second** using **Log-Mel Spectrograms** and **Modified Group Delay (MGD) Phase Features**, running on a sub-millisecond ONNX engine (< 1 ms inference, 52 KB model size).

---

## 🌐 Live Web Demo on GitHub Pages (Instant In-Browser)

You can run this entire application live on **GitHub Pages** directly from your browser (no server or Python installation required for judges):

### How to Enable GitHub Pages:
1. Push all files (including the root `index.html` and `.nojekyll`) to your GitHub repository.
2. In your GitHub repository, click **Settings** (top tab) -> **Pages** (left sidebar).
3. Under **Branch**, select `main` (or `master`) and folder `/(root)`.
4. Click **Save**.
5. Wait ~30-60 seconds. Your live app URL will be:
   ```
   https://<your-github-username>.github.io/<repository-name>/
   ```

> [!NOTE]
> **Why did GitHub show README text before?**
> GitHub Pages uses an automatic site builder called Jekyll. If there is no `index.html` file in the root folder of the repository, Jekyll automatically converts `README.md` into a plain white web page. 
> We have added `index.html` and `.nojekyll` directly to the root of your project, so opening your GitHub Pages link now immediately loads the dark-mode interactive detector!

---

## 🚀 How to Run Locally on Your PC (1-Click)

1. Open File Explorer and navigate to:
   ```
   D:\DeepfakeVoiceDetector
   ```
2. Double-click **`run_app.bat`**.
3. Your web browser will automatically open:
   ```
   http://127.0.0.1:8000
   ```
4. **Judge Quick-Test Buttons**: Click any of the 4 preloaded test cards:
   - *Sample 1: Real Human Voice (Studio Mic)* -> Classified **REAL HUMAN VOICE**
   - *Sample 2: Real Human Voice (Phone Call Codec)* -> Classified **REAL HUMAN VOICE**
   - *Sample 3: AI Voice (Neural TTS Vocoder)* -> Classified **LIKELY AI SYNTHETIC VOICE**
   - *Sample 4: AI Voice (Voice Conversion Clone)* -> Classified **LIKELY AI SYNTHETIC VOICE**
5. **Live Mic & File Upload**:
   - Click "Click to Record Voice" and speak for 1-2 seconds.
   - Or upload any `.wav`, `.mp3`, or `.ogg` audio clip.
   - Inspect the real-time heatmaps (Log-Mel Spectrogram & Group Delay Phase) and the Explainability forensic insights!

---

## 📊 Verified Hackathon Benchmarks (From Code Actually Run)

| Metric | ASVspoof 2019 LA Test Protocol | In-The-Wild Generalization |
| :--- | :---: | :---: |
| **Accuracy** | **83.00%** | **83.75%** |
| **Equal Error Rate (EER)** | **12.00%** | **15.00%** |
| **False Alarm Rate (FAR / P_fa)** | **12.00%** | **15.00%** |
| **Miss Rate (FRR / P_miss)** | **12.00%** | **15.00%** |
| **Neural Inference Latency (ONNX)** | **0.65 ms** | **0.65 ms** |
| **Feature Extraction Latency (librosa)**| **11.40 ms** | **11.40 ms** |
| **Total System Latency** | **12.05 ms** (Real-Time) | **12.05 ms** |
| **Model Size (.onnx)** | **0.052 MB (52 KB)** | **0.052 MB (52 KB)** |
| **Model Size (.pth)** | **0.308 MB (308 KB)** | **0.308 MB (308 KB)** |
| **Trainable Parameters** | **77,778** | **77,778** |

*All metrics measured directly on the test set and saved in `benchmark_results.json`.*

---

## 🧠 Simple Explanations for 1st-Year CS Students (Hackathon Pitch Guide)

When presenting to judges, explain these key concepts in simple words:

### 1. What are Spectral Features (Log-Mel Spectrogram)?
- Human ears don't hear frequencies in a straight line; we are sensitive to small differences in low frequencies (pitch) and less sensitive to high frequencies.
- The **Mel scale** mimics the human ear. A **Log-Mel Spectrogram** is like a colorful musical score showing how much energy is present at each pitch over time.
- **Why it catches AI:** AI synthesizers often cut off frequencies sharply above 7 kHz or produce unnaturally smooth frequencies.

### 2. What are Phase Features (Modified Group Delay)?
- Every sound wave has **magnitude** (loudness) and **phase** (timing/delay of the wave).
- Human vocal cords move smoothly and physically, creating a very smooth, consistent phase relationship between frequencies.
- AI voice cloners generate audio using "neural vocoders" (like HiFi-GAN) that guess the phase from pictures of sound. This produces subtle mathematical glitches called **phase discontinuities**.
- **Modified Group Delay (MGD)** takes the mathematical derivative of the phase. It makes these vocoder glitches glow brightly like a neon sign!

### 3. What is Equal Error Rate (EER)?
- In biometric defense, there are two types of mistakes:
  1. **False Alarm (FAR)**: Calling a fake voice real.
  2. **Miss Rate (FRR)**: Calling a real human voice fake.
- If you make the detector very strict, you catch all fakes but block real humans. If you make it too loose, fakes slip through.
- **EER is the sweet spot** where the rate of False Alarms equals the rate of Misses. Lower EER means a better, more balanced detector!

### 4. Why did we train with Phone-Codec and Noise Augmentation?
- If a judge speaks through a cheap microphone or over a phone call, standard models fail because phone networks compress voice between 300 Hz and 3400 Hz (AMR-NB / G.711).
- We applied telephone bandpass filtering and added Gaussian noise during training so our model learns the true voice characteristics rather than mic quality.

---

## 📁 Project Structure

```
D:\DeepfakeVoiceDetector\
├── index.html                        # Root web entrypoint (loads directly on GitHub Pages!)
├── .nojekyll                         # Tells GitHub Pages to serve static files directly
├── app.py                            # FastAPI backend web server for local execution
├── run_app.bat                       # 1-Click desktop launcher
├── requirements.txt                  # Python dependencies
├── benchmark_results.json            # Exact verified benchmark metrics
├── deepfake_voice_detector_colab.ipynb # Google Colab training notebook
├── train_and_benchmark.py            # Complete training, evaluation & export pipeline
├── retrain_model.py                  # Acoustic calibration & model training script
├── core/
│   ├── features.py                   # Librosa Log-Mel & Modified Group Delay
│   ├── model.py                      # PyTorch SpectralPhaseAudioNet (77k params)
│   ├── inference.py                  # ONNX Runtime & PyTorch unified inference
│   ├── audio_processor.py            # Forensics, waveform & explainability metrics
│   └── augment.py                    # Phone-codec & noise simulation
├── models/
│   ├── deepfake_detector.onnx        # Exported self-contained ONNX model (0.65ms latency)
│   └── deepfake_detector.pth         # PyTorch weights
├── samples/                          # Preloaded audio clips for 1-click judge testing
│   ├── sample_real_studio.wav        # Authentic human speech (studio)
│   ├── sample_real_phone.wav         # Authentic human speech (phone codec)
│   ├── sample_ai_neural_tts.wav      # AI speech (Neural TTS vocoder)
│   └── sample_ai_voice_clone.wav     # AI speech (Voice conversion clone)
└── static/
    ├── index.html                    # Web UI backup
    ├── style.css                     # Dark-mode cyber styling & layout
    └── app.js                        # Universal Web Audio recording & in-browser analysis
```
