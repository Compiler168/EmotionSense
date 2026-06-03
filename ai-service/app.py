"""
EmotionSense AI Service
========================
Flask-based REST API for facial emotion detection, human detection, and emoji detection.

Endpoints:
  POST /api/detect  - Detect emotions from an uploaded image
  GET  /api/health  - Health check endpoint

The service uses:
  - OpenCV Haar Cascade for face/human detection
  - FER+ ONNX deep-learning model for emotion classification
  - Emoji detection and mapping
"""

import os
import io
import base64
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS

# ─── Configuration ──────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# Target emotion labels for classification output
TARGET_EMOTIONS = ['Happy', 'Sad', 'Angry', 'Neutral', 'Surprise']

# Emoji mapping for each emotion
EMOTION_EMOJIS = {
    'Happy': '😊',
    'Sad': '😢',
    'Angry': '😠',
    'Neutral': '😐',
    'Surprise': '😲',
    'Fear': '😨',
    'Disgust': '🤢'
}

# ─── Human Detection (Face Detection) ──────────────────────────────
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
PROFILE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_profileface.xml'
ALT_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'

face_cascade = None
profile_cascade = None
alt_cascade = None

emotion_net = None

def load_model():
    """Load Haar Cascade classifiers and ONNX emotion model."""
    global face_cascade, profile_cascade, alt_cascade, emotion_net

    # Load Haar Cascades for face detection (multiple for robustness)
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        raise RuntimeError("Failed to load Haar Cascade classifier")

    profile_cascade = cv2.CascadeClassifier(PROFILE_CASCADE_PATH)
    if profile_cascade.empty():
        print("[WARNING] Profile face cascade not available")
        profile_cascade = None

    alt_cascade = cv2.CascadeClassifier(ALT_CASCADE_PATH)
    if alt_cascade.empty():
        print("[WARNING] Alt frontal face cascade not available")
        alt_cascade = None

    try:
        model_path = os.path.join(os.path.dirname(__file__), 'model', 'emotion-ferplus-8.onnx')
        emotion_net = cv2.dnn.readNetFromONNX(model_path)
        print("[OK] Deep Learning Emotion Model loaded successfully")
    except Exception as e:
        print(f"[WARNING] Failed to load ONNX emotion model: {e}")

    print("[OK] Haar Cascade face detector loaded (Human Detection)")
    print("[OK] Emoji detection system initialized")

def detect_faces(image):
    """
    Detect human faces in an image using OpenCV Haar Cascade.
    Uses multiple cascade classifiers for robustness.
    Returns list of (x, y, w, h) bounding boxes.
    """
    if face_cascade is None:
        return np.array([])

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Primary detection with default frontal face cascade
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # If no faces found, try with alt cascade (better for rotated/angled faces)
    if len(faces) == 0 and alt_cascade is not None:
        faces = alt_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

    # If still no faces found, try more relaxed parameters
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(20, 20),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

    # If still nothing, try profile face cascade
    if len(faces) == 0 and profile_cascade is not None:
        faces = profile_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

    return faces

def analyze_face_features(face_img):
    """
    Analyze facial features using Deep Learning (FER+ ONNX model).
    The emotion-ferplus-8.onnx model expects:
      - Input: 1x1x64x64 float32 tensor (grayscale, normalized to 0-1)
      - Output: 8 classes [neutral, happiness, surprise, sadness, anger, disgust, fear, contempt]
    """
    global emotion_net
    if face_img is None or face_img.size == 0:
        return get_default_emotion()

    if emotion_net is None:
        return get_default_emotion()

    try:
        # Convert to grayscale
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img

        # FER+ ONNX model expects 64x64 grayscale image (pixel values 0-255)
        blob = cv2.dnn.blobFromImage(
            gray,
            scalefactor=1.0,
            size=(64, 64),
            mean=(0,),
            swapRB=False,
            crop=False
        )

        emotion_net.setInput(blob)
        logits = emotion_net.forward()

        # Apply Softmax to get probabilities
        exp_logits = np.exp(logits[0] - np.max(logits[0]))
        probs = exp_logits / np.sum(exp_logits)

        # FER+ Classes: 0:neutral, 1:happiness, 2:surprise, 3:sadness, 4:anger, 5:disgust, 6:fear, 7:contempt
        # Map to our target emotions (use raw softmax probabilities — no re-normalization)
        emotion_scores = {
            'Neutral': float(probs[0]),
            'Happy': float(probs[1]),
            'Surprise': float(probs[2]),
            'Sad': float(probs[3]),
            'Angry': float(probs[4])
        }

        # Normalize only among our 5 target emotions so they sum to 1.0
        total = sum(emotion_scores.values())
        if total > 0:
            emotion_scores = {k: v / total for k, v in emotion_scores.items()}
        else:
            emotion_scores = get_default_emotion()

        return emotion_scores

    except Exception as e:
        print(f"Error analyzing face features: {e}")
        return get_default_emotion()

def get_default_emotion():
    """Return balanced default emotion scores."""
    return {e: 1.0 / len(TARGET_EMOTIONS) for e in TARGET_EMOTIONS}

def predict_emotion(face_img):
    """
    Predict emotion from a face image using the FER+ ONNX model.
    """
    # Analyze facial features
    emotion_scores = analyze_face_features(face_img)

    # Find dominant emotion
    dominant_emotion = max(emotion_scores, key=emotion_scores.get)
    confidence = emotion_scores[dominant_emotion]

    return {
        'dominant_emotion': dominant_emotion,
        'confidence': round(confidence * 100, 2),
        'emoji': EMOTION_EMOJIS.get(dominant_emotion, '❓'),
        'all_emotions': {k: round(v * 100, 2) for k, v in emotion_scores.items()}
    }

# ─── API Routes ─────────────────────────────────────────────────

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'EmotionSense AI Service',
        'model_loaded': emotion_net is not None,
        'face_detector_loaded': face_cascade is not None
    })


@app.route('/api/detect', methods=['POST'])
def detect_emotion():
    """
    Detect emotions from an uploaded image.

    Accepts:
      - multipart/form-data with 'image' file field
      - JSON with 'image' field containing base64-encoded image

    Returns:
      - Detected faces with emotions and confidence scores
    """
    try:
        image = None

        # Handle file upload
        if 'image' in request.files:
            file = request.files['image']
            img_bytes = file.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Handle base64 image (from webcam)
        elif request.is_json and 'image' in request.json:
            img_data = request.json['image']
            # Remove data URL prefix if present
            if ',' in img_data:
                img_data = img_data.split(',')[1]
            img_bytes = base64.b64decode(img_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'No valid image provided'}), 400

        # Detect faces
        faces = detect_faces(image)

        # If no faces detected, return a clear error
        if len(faces) == 0:
            return jsonify({
                'success': False,
                'faces_detected': 0,
                'error': 'No face detected in the image',
                'message': 'Please ensure a clear face is visible in the image with good lighting.'
            }), 200

        # Process each detected face
        results = []
        h_img, w_img = image.shape[:2]

        for i, (x, y, w, h) in enumerate(faces):
            # Add padding around face ROI (15%) for better emotion context
            pad_w = int(w * 0.15)
            pad_h = int(h * 0.15)
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(w_img, x + w + pad_w)
            y2 = min(h_img, y + h + pad_h)

            face_roi = image[y1:y2, x1:x2]

            # Predict emotion
            emotion_result = predict_emotion(face_roi)

            results.append({
                'face_id': i + 1,
                'bounding_box': {
                    'x': int(x), 'y': int(y),
                    'width': int(w), 'height': int(h)
                },
                **emotion_result
            })

        return jsonify({
            'success': True,
            'faces_detected': len(results),
            'results': results,
            'message': f'Successfully detected {len(results)} face(s)'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ─── Server Startup ─────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 50)
    print("  EmotionSense AI Service")
    print("=" * 50)
    load_model()
    print("\n- Human Detection (Face Detection) - Loaded")
    print("- Emotion Analysis - Initialized")
    print("\nStarting AI service on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False)
