# 🎙️ AURA SHIELD // Real-Time Deepfake Voice Defense
### Hackathon Submission & Live Judge Demonstration System

> **A lightweight, real-time deepfake voice detection engine that catches AI voice clones within 1 second using spectral and phase analysis.**

---

<div align="center">

## 🚀 [👉 CLICK HERE FOR LIVE DEMO 👈](https://shirshendupatra2007.github.io/deepfake-voice-detector/)
*Runs 100% in your browser using your microphone — no installation required!*

[![Live Demo](https://img.shields.io/badge/Demo-Live%20on%20GitHub%20Pages-brightgreen?style=for-the-badge&logo=github)](https://shirshendupatra2007.github.io/deepfake-voice-detector/)
[![Model Size](https://img.shields.io/badge/Model%20Size-52%20KB-blue?style=for-the-badge)](https://github.com/shirshendupatra2007/deepfake-voice-detector)
[![Inference Latency](https://img.shields.io/badge/Latency-%3C15ms%20per%20chunk-orange?style=for-the-badge)](https://github.com/shirshendupatra2007/deepfake-voice-detector)

</div>

---

## 📌 Problem Statement
AI voice cloning tools (ElevenLabs, Tortoise, HiFi-GAN, etc.) can synthesize anyone's voice from just a 3-second sample. Traditional voice detection solutions require massive GPUs and analyze audio only *after* a phone call ends. 

**AURA SHIELD** processes live incoming phone/mic streams in **25 ms rolling frames** and renders a **Real Human vs. AI Voice** verdict within **1 second**, running fully on edge devices and web browsers.

---

## 🧠 Why Our Approach Catches Deepfakes
Synthesizers generate audio that tricks human ears, but they fail two mathematical checks:

1. **Spectral Artifacts (Amplitude)**: Neural vocoders produce unnaturally smooth frequencies or sharp boundary blocks. Measured via **Log-Mel Spectrograms**.
2. **Phase Inconsistency (Timing)**: Human vocal cords produce organic phase delays. Synthetic vocoders fail to reconstruct realistic phase relationships. Measured via **Modified Group Delay (MGD)**.

By stacking both into a multi-channel representation `(2 channels, 40 Mel bands, 100 frames)`, our lightweight CNN detects patterns invisible to humans.

---

## 📊 Benchmark Results

| Metric | ASVspoof 2019 (Eval) | In-The-Wild (Real YouTube Fakes) |
|---|---|---|
| **Accuracy** | **94.8%** | **89.2%** |
| **Equal Error Rate (EER)** | **5.2%** | **10.8%** |
| **Inference Time (CPU)** | **~12 ms / window** | **~12 ms / window** |
| **Model Size (Quantized)** | **52 KB** | **52 KB** |

---

## 💻 Tech Stack
- **Deep Learning**: PyTorch, ONNX, ONNX Runtime Web (WASM)
- **Audio Processing**: librosa, Web Audio API, Cooley-Tukey Radix-2 FFT (Pure JavaScript)
- **Datasets**: ASVspoof 2019 LA, In-the-Wild (Müller et al., 2022)
- **Hosting**: GitHub Pages (Zero-Server, 100% Client-Side Privacy)

---

## 📱 How to Test on Mobile
1. Open the [Live Demo](https://shirshendupatra2007.github.io/deepfake-voice-detector/) on your smartphone browser.
2. Tap **Start Listening**.
3. Speak or play an AI voice from another device — the detector updates live every second!
