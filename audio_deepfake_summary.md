
# 🧠 Audio Deepfake Detection Project – Summary

## 📁 Dataset: ASVspoof2019

- You are working with the **Logical Access (LA)** subset.
- ASVspoof2019 contains two main subsets:
  - **LA (Logical Access)**: Focuses on detecting **AI-generated spoofed speech** (e.g., from TTS and voice conversion).
  - **PA (Physical Access)**: Focuses on detecting **replayed recordings** played through speakers and re-recorded in real environments.

### ✅ You're using:
- **LA** (Logical Access), which is **suitable for deepfake detection**.

---

## ⚔️ CM vs ASV: Key Differences

| Term | Full Form | Task | Role |
|------|-----------|------|------|
| **CM** | Countermeasure | Detects if audio is **bonafide** or **spoof** | Filters fake speech before ASV |
| **ASV** | Automatic Speaker Verification | Verifies speaker identity | Accepts/rejects identity claims |

### 🔁 Real-World Usage Flow

```
[ CM (spoof filter) ] → [ ASV (identity verification) ]
```

- **CM** filters fake inputs.
- **ASV** checks if the speaker is who they claim to be.

---

## 📊 Key Feature Categories for Deepfake Detection

### 1. **Spectral Features**
- MFCCs, STFT, CQCC, LPC, spectrograms

### 2. **Phase-Based Features**
- Group delay, phase spectrum, relative phase shift

### 3. **Temporal Features**
- Zero-crossing rate, energy contour, amplitude envelope

### 4. **Higher-Level Features**
- Pitch, jitter, formants, modulation spectral features

---

## 📚 Key Papers & Techniques

| Paper | Key Features | Highlights |
|-------|--------------|------------|
| ASVspoof2019 (Todisco) | LFCC, CQCC | Benchmark dataset |
| ADD 2022 | Mel-spectrograms, RawNet2 | Real-world generalization |
| RawNet2 | Raw audio | End-to-end deep learning |
| AASIST | Multi-scale spectrograms, phase | Graph attention networks |
| LCNN | Spectral features | Lightweight CNN |
| Tak et al. (2022) | Raw audio, SSL | Domain adaptation |
| Zhang et al. (2021) | Mel-spectrogram + attention | Cross-domain generalization |
| Wang et al. (2020) | Group delay, MFCC | One-class learning |
| Yang et al. (2022) | Temporal features | Sequential artifacts |
| Sahidullah et al. | Cochlear cepstrum | Physiological inspiration |

---

## 🛠️ Tools

- You plan to implement:
  - **Librosa** version (CPU-based feature extraction)
  - **Torchaudio** version (GPU acceleration)
