# 🎙️ Comparative Study of Speech-to-Text Models for Noisy Real-World Audio

<p align="center">

**Benchmarking Whisper Small, Faster-Whisper Small, and Wav2Vec2 Base 960h**

A reproducible ASR benchmark for customer-support voice applications under clean and noisy audio conditions.

</p>

---

## 📌 Overview

Automatic Speech Recognition (ASR) is an important component of modern voice-based customer-support systems. In a real support call, the audio is rarely perfect. Background conversations, different microphones, telephone compression, regional accents, and environmental noise can all affect transcription quality.

This project evaluates three ASR models under a common benchmarking setup:

- **Whisper Small**
- **Faster-Whisper Small**
- **Wav2Vec2 Base 960h**

The objective is not only to compare transcription accuracy, but also to understand the practical trade-offs between **accuracy, inference speed, memory usage, deployment complexity, and noisy-audio robustness**.

---

## 🎯 Project Objective

The main objectives of this project are to:

- Research modern speech-to-text architectures.
- Benchmark three selected ASR models under identical conditions.
- Evaluate performance on clean and noisy speech.
- Calculate Word Error Rate (WER).
- Measure inference latency and Real-Time Factor (RTF).
- Estimate memory requirements.
- Compare deployment complexity.
- Identify the most suitable candidate for a customer-support voice assistant.

---

## 🧠 Models Evaluated

| Model | Architecture | Main Characteristic |
|---|---|---|
| **Whisper Small** | Encoder-Decoder Transformer | General-purpose multilingual ASR |
| **Faster-Whisper Small** | Whisper + CTranslate2 | Optimized Whisper inference |
| **Wav2Vec2 Base 960h** | CNN + Transformer + CTC | English ASR using self-supervised pretraining |

### 1. Whisper Small

Whisper Small is part of OpenAI's Whisper model family. It uses an encoder-decoder Transformer architecture and was trained on a large multilingual collection of audio obtained from the web.

**Strengths**
- Strong general-purpose speech recognition
- Broad language coverage
- Good tolerance to different recording conditions
- Well-established ecosystem

**Limitations**
- Higher computational requirements than smaller ASR models
- CPU inference can be relatively slow
- Transformer decoding can increase latency

---

### 2. Faster-Whisper Small

Faster-Whisper uses the Whisper model through the **CTranslate2** inference engine. The objective is to improve inference efficiency without changing the underlying Whisper model's learned knowledge.

**Strengths**
- Faster inference than conventional Whisper implementations
- Lower memory requirements in optimized configurations
- Supports quantized inference
- Practical for production-oriented deployments

**Limitations**
- Requires an additional inference framework
- Performance depends on hardware and configuration
- Still based on Whisper's underlying decoding approach

---

### 3. Wav2Vec2 Base 960h

Wav2Vec2 follows a different approach from Whisper. It learns useful speech representations from audio and is subsequently fine-tuned for speech recognition using transcribed data.

The `facebook/wav2vec2-base-960h` checkpoint is intended for English speech recognition and is closely associated with the LibriSpeech dataset.

**Strengths**
- Established English ASR architecture
- Efficient model size
- Strong performance on suitable English speech
- Different architecture provides a useful comparison against Whisper

**Limitations**
- English-focused checkpoint
- Performance can change significantly when audio differs from training conditions
- Less naturally suited to multilingual applications
- Additional processing may be required for formatting/punctuation

---

# 🗂️ Dataset

The benchmark uses an open speech-recognition dataset containing audio recordings with corresponding reference transcripts.

The same recordings and reference transcripts are supplied to every model to maintain a consistent evaluation environment.

### Evaluation Conditions

The benchmark considers two primary conditions:

```text
                    ┌─────────────────┐
                    │  Speech Dataset │
                    └────────┬────────┘
                             │
                 ┌───────────┴───────────┐
                 ▼                       ▼
          ┌──────────────┐       ┌──────────────┐
          │ Clean Audio  │       │ Noisy Audio  │
          └──────┬───────┘       └──────┬───────┘
                 │                      │
                 └──────────┬───────────┘
                            ▼
                  ┌──────────────────┐
                  │ Three ASR Models │
                  └────────┬─────────┘
                           ▼
                  ┌──────────────────┐
                  │ Evaluation      │
                  │ WER / Latency   │
                  │ RTF / Memory    │
                  └──────────────────┘
