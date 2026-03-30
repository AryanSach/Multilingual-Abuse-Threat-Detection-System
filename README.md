#  Multilingual Abuse & Threat Detection System

An end-to-end NLP system designed to detect abusive language and threatening intent from text derived from audio streams (e.g., CCTV recordings).

This system supports **Hindi, Hinglish, and English**, making it suitable for real-world user-generated and noisy conversational data.

---

##  System Overview

The system is part of a larger pipeline:

CCTV Recordings  
→ Audio Extraction (WAV)  
→ Speech-to-Text Transcription  
→ NLP Models (Abuse + Threat Detection)  
→ Alerts / Output  

---

##  Features

-  **Token-level abuse detection** (NER-style)
-  **Sentence-level threat detection**
-  Multilingual support (Hindi + Hinglish + English)
-  Handles code-mixed text
-  Robust against noisy inputs (leet, misspellings)
-  Synthetic + real dataset training

---

##  Models Used

### 1. Abuse Detection (Token Classification)
- Model: **MuRIL (google/muril-base-cased)**
- Labels:
  - `O` (Normal)
  - `DIRECT` (Direct abuse)
  - `SEXUAL` (Sexual abuse)
  - `SLUR` (Slur)

### 2. Threat Detection (Sequence Classification)
- Model: **XLM-RoBERTa (xlm-roberta-base)**
- Labels:
  - `NOT_THREAT`
  - `THREAT`

---

##  Key Technical Highlights

- Custom **token-label alignment** using `word_ids()` for subword handling
- **Subword reconstruction** (merging `##` tokens into full words)
- Multilingual **synthetic dataset generation**
  - Morphological suffix expansion
  - Romanization (Hindi ↔ Hinglish)
  - Leetspeak decoding (@ → a, 3 → e, etc.)
  - Noise injection (misspellings, truncation)
- Hybrid training:
  - Real datasets (HASOC, Kaggle)
  - Synthetic data for robustness
- Text normalization:
  - Unicode NFKC normalization
  - Regex cleanup

---


