# 🎙️ NeuroVoice — Speech-to-Text AI

> **Multilingual speech recognition application powered by OpenAI Whisper, featuring a modern glassmorphism UI and zero OS dependency on Windows.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

---

## 📋 Executive Summary

**NeuroVoice** is a high-performance Speech-to-Text application that splits a heavy ML inference engine from a lightweight, browser-based frontend. It features a stunning dark-mode glassmorphism UI and natively bypasses Windows FFMPEG requirements through custom `librosa` array mapping — delivering seamless multilingual transcription across global and regional languages.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🎙️ **Whisper AI Engine** | OpenAI Whisper-based speech recognition with PyTorch backend |
| 🌍 **Multilingual Support** | Transcription across major global and regional languages |
| 🖥️ **Modern Glassmorphism UI** | Vanilla JS/HTML/CSS dark-mode frontend — deployable to Vercel |
| ⚡ **FastAPI Backend** | High-performance REST API connecting frontend to ML inference |
| 🔧 **Zero OS Dependencies** | Custom `librosa` array mapping bypasses Windows FFMPEG pipeline |

---

## 🏗️ Architecture

```
┌───────────────────────┐         ┌──────────────────────────┐
│    Frontend (Vercel)   │  HTTP  │     Backend (FastAPI)     │
│  ┌─────────────────┐   │◄──────►│  ┌────────────────────┐  │
│  │ Glassmorphism UI │   │        │  │ PyTorch + Whisper  │  │
│  │ Vanilla JS/HTML  │   │        │  │ Multilingual STT   │  │
│  └─────────────────┘   │        │  └────────────────────┘  │
└────────────────────────┘        └──────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** installed on your system
- **~2 GB disk space** for PyTorch and HuggingFace model downloads
- **RTX GPU recommended** (Whisper requires ~1.5–2 GB RAM minimum)

> ⚠️ **Hosting Note:** Render.com free tier provides only 512 MB RAM — insufficient for Whisper. Use Hugging Face Spaces (16 GB free) or a paid tier for public deployment.

### Installation (Local — Recommended)

```powershell
# Step 1: Clone the repository
git clone https://github.com/Aumnamaha/Neuro-Voice.git
cd Neuro-Voice

# Step 2: Create & activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Step 3: Install dependencies
pip install torch torchaudio
pip install fastapi uvicorn librosa numpy

# Step 4: Start the backend server
python backend/api.py
# Wait for: "Application startup complete" in terminal

# Step 5: Open frontend
# Navigate to frontend/ folder in File Explorer and double-click index.html
```

### Frontend Deployment (Vercel)

1. Push repository to GitHub
2. Go to [Vercel.com](https://vercel.com/) → Add New Project
3. Import your repository with these settings:
   - **Root Directory:** `frontend`
   - **Framework Preset:** Other
   - **Include files outside root:** OFF
4. Click **Save and Deploy**

### Backend Deployment Command

```bash
uvicorn api:app --host 0.0.0.0 --port 10000
```

---

## 📁 Project Structure

```
Neuro-Voice/
├── backend/                  # Python AI logic & FastAPI server
│   └── api.py                # Uvicorn entry point + ML inference
├── frontend/                 # Web UI (HTML/CSS/JS) — Vercel deployable
│   └── index.html            # Glassmorphism dark-mode interface
└── README.md                 # This file
```

---

## 🧪 Verification

```powershell
# Verify dependencies are installed
pip check

# Launch backend and verify (no errors in terminal)
python backend/api.py
# Expected: "Application startup complete" message
```

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| **Backend** | Python, FastAPI, Uvicorn, PyTorch |
| **AI/ML** | OpenAI Whisper (multilingual STT) |
| **Frontend** | Vanilla HTML/CSS/JS, Glassmorphism UI |
| **Deployment** | Vercel (frontend), Hugging Face Spaces / paid tier (backend) |

---

## 👤 Author

**Maintained by [Thirunagari Aum Namaha](https://github.com/Aumnamaha)** • [LinkedIn](https://www.linkedin.com/in/aumnamaha-thirunagari-9214782bb/)

---

## 📜 License

This project is licensed under the MIT License.
