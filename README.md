
# 🧠 Audio Deepfake Detection Project – Conversational README

Welcome to the detailed development log and knowledge summary for our **Audio Deepfake Detection** project. This document captures our thought process, questions, and evolving understanding as we dive into the domain of detecting fake audio using machine learning.

---

## 👋 Getting Started: Project Concept

**User:** *"I'm working on audio deepfake detection as a project, it's very cool!"*

This exciting journey started with exploring deepfake audio — a field that combines speech processing, machine learning, and forensics. The goal: detect synthetic voices generated using AI tools like TTS (Text-to-Speech) or voice conversion.

---

## 🔍 First Step: What Features to Use?

**User:** *"Now I'm looking into features, which ones should I take?"*

We discussed and outlined the most relevant **signal processing features** used in state-of-the-art audio deepfake detection models. Here's what we documented:

### 📊 Categories of Features

1. **Spectral Features**
   - MFCCs
   - STFT (Spectrograms)
   - CQT
   - LPC coefficients
   - Spectrogram analysis

2. **Phase-Based Features**
   - Group delay
   - Phase spectrum
   - Relative phase

3. **Temporal Features**
   - Zero-crossing rate
   - Energy contour
   - Amplitude envelope

4. **Higher-Level Features**
   - Formant analysis
   - Pitch & jitter
   - Voice source
   - Modulation spectrum

### 📚 Key Research Papers

We listed key papers like:
- ASVspoof2019
- ADD2022
- RawNet2
- AASIST
- LCNN
- Self-supervised learning works
- One-class learning (Wang et al.)
- Sequence-based detection (Yang et al.)

---

## 🧪 Feature Extraction Plans

**User:** *"Let's make one using Librosa first for CPU, then one with Torchaudio for GPU."*

We plan to implement two feature extraction pipelines:
- `librosa` for fast prototyping and CPU-based workflows.
- `torchaudio` for GPU-based, scalable extraction suited for training.

---

## 📁 Dataset Understanding: ASVspoof2019

**User:** *"I'm working with ASVspoof2019."*  
**User:** *"Can we understand the dataset first?"*  
**User:** *"I only have LA, not PA. What’s PA then and when is it used?"*

### 📦 ASVspoof2019 Dataset

- Two subsets:
  - **LA (Logical Access):** Detects TTS and VC-based attacks.
  - **PA (Physical Access):** Detects replay attacks in real physical environments.

**Your Focus**: **Logical Access (LA)** — ideal for audio deepfake detection.

---

## ⚔️ Understanding CM vs ASV

**User:** *"What's CM vs ASV?"*  
**User:** *"Can we summarize this in text?"*

### ✅ Definitions

| Term | Full Form | Role |
|------|-----------|------|
| **CM** | Countermeasure | Detects spoofed (fake) audio |
| **ASV** | Automatic Speaker Verification | Confirms speaker identity |

### 🔄 Real-World Flow

```
[ CM ] → [ ASV ]
```
CM first checks for fakes, then ASV verifies the speaker’s identity.

---

## 📝 Summary Export

You asked for a summary of everything in Markdown (`.md`) format, which we created and saved as:

- [audio_deepfake_summary.md](sandbox:/mnt/data/audio_deepfake_summary.md)

Then, you asked for a **super-detailed `README.md`** version, which you’re reading now.

---

## 🚧 Next Steps

- Implement and benchmark **Librosa** and **Torchaudio** pipelines.
- Perform feature extraction on LA subset.
- Train baseline models (e.g., CNN, RawNet2).
- Test generalizability with unseen attacks.

---

## 📬 Want to Continue?

Let me know when you're ready to:
- Load the ASVspoof2019 metadata and audio.
- Begin with feature extraction using `librosa`.
- Or start training a model with `torchaudio` inputs.
