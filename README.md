<p align="center">
  <h1 align="center">🧠 EmotionSense</h1>
  <p align="center">
    <strong>AI-Powered Facial Emotion Detection Platform</strong>
  </p>
  <p align="center">
    Real-time emotion recognition using deep learning, with webcam capture, emoji analysis, and text sentiment detection.
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/node.js-18%2B-green?logo=node.js&logoColor=white" alt="Node.js">
    <img src="https://img.shields.io/badge/OpenCV-4.11-red?logo=opencv&logoColor=white" alt="OpenCV">
    <img src="https://img.shields.io/badge/MongoDB-Optional-brightgreen?logo=mongodb&logoColor=white" alt="MongoDB">
    <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
  </p>
</p>

---

## 📖 Overview

**EmotionSense** is a full-stack AI application that detects and classifies human emotions from facial images in real time. It combines a **Python Flask** AI service powered by a **FER+ ONNX deep learning model** with a **Node.js/Express** backend and a sleek **vanilla JavaScript** dashboard frontend.

The system supports multiple input methods — image upload, webcam capture, emoji recognition, and keyword-based text sentiment analysis — making it a versatile tool for emotion research, UX testing, and educational purposes.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 👤 **Face Detection** | Real-time multi-face detection using OpenCV Haar Cascade classifiers |
| 😊 **Emotion Recognition** | Classifies 5 emotions (Happy, Sad, Angry, Neutral, Surprise) using FER+ ONNX model |
| 📷 **Webcam Capture** | Live camera feed with frame capture for instant analysis |
| 📤 **Image Upload** | Drag-and-drop or file picker with 10MB limit |
| 🎨 **Emoji Detection** | Recognizes 20+ emotion emojis from text and maps to emotion categories |
| 📝 **Text Sentiment** | Keyword-based emotion analysis with confidence scoring |
| 📊 **Analytics Dashboard** | Real-time statistics, emotion distribution charts, and recent activity |
| 📋 **Detection History** | Persistent history with MongoDB (optional — works without DB too) |
| 🌙 **Premium Dark UI** | Glassmorphism design with animated backgrounds and micro-interactions |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Browser (Frontend)                    │
│         HTML5 · CSS3 · Vanilla JavaScript                │
│     Dashboard │ Detection │ History │ Toasts             │
└───────────────────────┬─────────────────────────────────┘
                        │ HTTP / JSON
┌───────────────────────▼─────────────────────────────────┐
│              Node.js / Express Backend (:3000)           │
│   Routes: /detect · /detect/webcam · /emoji             │
│           /text-emotion · /stats · /history             │
│   Middleware: CORS · Multer · Morgan                     │
│   Database: MongoDB / Mongoose (optional)                │
└───────────────────────┬─────────────────────────────────┘
                        │ HTTP / multipart
┌───────────────────────▼─────────────────────────────────┐
│              Python Flask AI Service (:5000)             │
│   Face Detection: OpenCV Haar Cascade (3 classifiers)   │
│   Emotion Model: FER+ ONNX (emotion-ferplus-8.onnx)    │
│   Processing: NumPy · OpenCV DNN                         │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | HTML5, CSS3 (custom properties, glassmorphism), Vanilla JavaScript |
| **Backend** | Node.js, Express 5, Mongoose, Multer, Axios |
| **AI Service** | Python 3, Flask, OpenCV (Haar Cascade + DNN), NumPy |
| **ML Model** | FER+ ONNX — pre-trained emotion classification (64×64 grayscale input) |
| **Database** | MongoDB (optional — all features work without it) |

---

## 🚀 Installation

### Prerequisites

- **Node.js** v18+
- **Python** 3.10+
- **MongoDB** (optional — for persistent history)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Programmaster00/EmotionSen.git
cd EmotionSen
```

### Step 2: Download the AI Model

Download the FER+ ONNX emotion model and place it in the `ai-service/model/` directory:

```bash
# Create the model directory if it doesn't exist
mkdir -p ai-service/model

# Download the FER+ ONNX model (~35MB)
# Option A: Download from ONNX Model Zoo
curl -L -o ai-service/model/emotion-ferplus-8.onnx \
  "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
```

### Step 3: Install Backend Dependencies

```bash
cd backend
npm install
```

### Step 4: Install AI Service Dependencies

```bash
cd ../ai-service
pip install -r requirements.txt
```

### Step 5: Configure Environment

Copy the example environment file and update as needed:

```bash
cd ../backend
cp .env.example .env
```

Edit `backend/.env` with your settings:

```env
PORT=3000
MONGODB_URI=mongodb://127.0.0.1:27017/emotionsense
AI_SERVICE_URL=http://127.0.0.1:5000
NODE_ENV=development
```

> **Note:** MongoDB is optional. The application functions fully without it — detection history just won't persist between sessions.

### Step 6: Start the Services

**Terminal 1 — Backend Server:**
```bash
cd backend
npm start
# Runs on http://localhost:3000
```

**Terminal 2 — AI Service:**
```bash
cd ai-service
python app.py
# Runs on http://localhost:5000
```

### Step 7: Open the Dashboard

Navigate to **http://localhost:3000** in your browser.

### Quick Start (Windows)

Double-click `start-services.bat` to launch everything automatically.

### Quick Start (Linux / macOS)

```bash
bash start-services.sh
```

---

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `3000` | Backend server port |
| `MONGODB_URI` | `mongodb://127.0.0.1:27017/emotionsense` | MongoDB connection string |
| `AI_SERVICE_URL` | `http://127.0.0.1:5000` | Python AI service URL |
| `NODE_ENV` | `development` | Node environment |

### Database Options

- **Local MongoDB**: `mongod` (default)
- **MongoDB Atlas**: Update `MONGODB_URI` with your Atlas connection string
- **No Database**: All features work except persistent history

---

## 📖 Usage Guide

### 🏠 Dashboard
View real-time analytics — total detections, average confidence, most common emotion, and emotion distribution charts.

### 🔍 Detection
1. **Upload Mode**: Drag and drop an image or click to browse (JPEG, PNG, WebP, GIF up to 10MB)
2. **Webcam Mode**: Click "Start Camera" → "Capture" to freeze a frame
3. Click **"Analyze Emotion"** to process
4. View results: dominant emotion with emoji, confidence ring, and full emotion breakdown

### 📋 History
Browse all past detection results with timestamps, confidence scores, input methods, and face counts. Clear history with one click.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | System health check (backend, AI service, database) |
| `POST` | `/api/detect` | Detect emotion from uploaded image (multipart/form-data) |
| `POST` | `/api/detect/webcam` | Detect emotion from base64 webcam capture |
| `POST` | `/api/emoji` | Analyze emojis in text and map to emotions |
| `POST` | `/api/text-emotion` | Keyword-based text emotion analysis |
| `GET` | `/api/stats` | Get detection statistics and emotion distribution |
| `GET` | `/api/history` | Get detection history (paginated) |
| `DELETE` | `/api/history` | Clear all detection history |

### Example — Detect Emotion from Image

```bash
curl -X POST http://localhost:3000/api/detect \
  -F "image=@photo.jpg"
```

```json
{
  "success": true,
  "faces_detected": 1,
  "results": [{
    "face_id": 1,
    "dominant_emotion": "Happy",
    "confidence": 87.5,
    "emoji": "😊",
    "all_emotions": {
      "Happy": 87.5, "Sad": 5.2, "Angry": 2.1,
      "Neutral": 3.8, "Surprise": 1.4
    },
    "bounding_box": { "x": 150, "y": 120, "width": 180, "height": 220 }
  }]
}
```

---

## 📁 Project Structure

```
EmotionSense/
├── ai-service/                  # Python Flask AI Service
│   ├── app.py                   # Main AI service (face detection + emotion model)
│   ├── requirements.txt         # Python dependencies
│   └── model/
│       └── emotion-ferplus-8.onnx  # FER+ ONNX emotion model (download separately)
│
├── backend/                     # Node.js Express Backend
│   ├── server.js                # Express server entry point
│   ├── package.json             # Node.js dependencies
│   ├── .env.example             # Environment configuration template
│   ├── routes/
│   │   └── detection.js         # API route handlers
│   └── models/
│       └── Detection.js         # Mongoose schema for detection history
│
├── frontend/                    # Vanilla JS Web Dashboard
│   ├── index.html               # Main HTML (SPA with tab navigation)
│   ├── app.js                   # Frontend application logic
│   └── style.css                # Premium dark theme styles
│
├── start-services.bat           # Windows quick-start script
├── start-services.sh            # Linux/macOS quick-start script
├── .gitignore
└── README.md
```

---

## 📸 Screenshots

> Screenshots can be added here to showcase the dashboard, detection interface, and results panel.

<!-- 
![Dashboard](screenshots/dashboard.png)
![Detection Results](screenshots/detection.png)
![History](screenshots/history.png)
-->

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/your-feature`
3. **Commit** your changes: `git commit -m "Add your feature"`
4. **Push** to your branch: `git push origin feature/your-feature`
5. **Open** a Pull Request

### Guidelines

- Follow existing code style and conventions
- Add comments for complex logic
- Test your changes before submitting
- Update documentation if adding new features

---

## 📄 License

This project is licensed under the **MIT License** — free to use for education, research, and commercial purposes.

---



<p align="center">
  Built with ❤️ using Python, Node.js, and OpenCV
</p>
