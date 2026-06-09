/**
 * EmotionSense Dashboard — Frontend Application Logic
 * =====================================================
 * Handles: Navigation, image upload, webcam, API calls,
 *          results display, history, human detection, emoji detection,
 *          and dashboard stats.
 */

const API_BASE = window.location.origin + '/api';

// ─── Emoji Map ──────────────────────────────────────────────────
const EMOJI_MAP = { 
    Happy: '😊', Sad: '😢', Angry: '😠', Neutral: '😐', Surprise: '😲',
    Fear: '😨', Disgust: '🤢'
};
const COLOR_MAP = {
    Happy: '#fbbf24', Sad: '#60a5fa', Angry: '#f87171',
    Neutral: '#94a3b8', Surprise: '#a78bfa',
    Fear: '#f97316', Disgust: '#84cc16'
};

// ─── State ──────────────────────────────────────────────────────
let currentMode = 'upload';
let selectedFile = null;
let capturedFrame = null;
let webcamStream = null;
let lastDetectionResult = null;

// ─── Navigation ─────────────────────────────────────────────────
document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        const section = link.dataset.section;
        document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
        link.classList.add('active');
        document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
        document.getElementById('section-' + section).classList.add('active');
        if (section === 'dashboard') loadDashboard();
        if (section === 'history') loadHistory();
    });
});

// ─── Mode Switching ─────────────────────────────────────────────
function switchMode(mode) {
    currentMode = mode;
    document.getElementById('tabUpload').classList.toggle('active', mode === 'upload');
    document.getElementById('tabWebcam').classList.toggle('active', mode === 'webcam');
    document.getElementById('modeUpload').classList.toggle('active', mode === 'upload');
    document.getElementById('modeWebcam').classList.toggle('active', mode === 'webcam');
    selectedFile = null;
    capturedFrame = null;
    document.getElementById('btnDetect').disabled = true;
    if (mode === 'upload') stopWebcam();
}

// ─── File Upload ────────────────────────────────────────────────
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');

uploadZone.addEventListener('click', () => fileInput.click());
uploadZone.addEventListener('dragover', (e) => { e.preventDefault(); uploadZone.classList.add('dragover'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => { if (fileInput.files.length > 0) handleFile(fileInput.files[0]); });

function handleFile(file) {
    if (!file.type.startsWith('image/')) { showToast('Please select an image file', 'error'); return; }
    if (file.size > 10 * 1024 * 1024) { showToast('File size must be under 10MB', 'error'); return; }
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('previewImage').src = e.target.result;
        document.getElementById('previewContainer').style.display = 'block';
        uploadZone.style.display = 'none';
        document.getElementById('btnDetect').disabled = false;
    };
    reader.readAsDataURL(file);
}

function clearPreview() {
    selectedFile = null;
    document.getElementById('previewContainer').style.display = 'none';
    uploadZone.style.display = 'block';
    document.getElementById('btnDetect').disabled = true;
    fileInput.value = '';
}

// ─── Webcam ─────────────────────────────────────────────────────
async function startWebcam() {
    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: 640, height: 480 } });
        const video = document.getElementById('webcamVideo');
        video.srcObject = webcamStream;
        document.getElementById('webcamOverlay').classList.add('hidden');
        document.getElementById('btnStartCam').style.display = 'none';
        document.getElementById('btnStopCam').style.display = 'inline-flex';
        document.getElementById('btnCapture').style.display = 'inline-flex';
    } catch (err) {
        showToast('Camera access denied: ' + err.message, 'error');
    }
}

function stopWebcam() {
    if (webcamStream) {
        webcamStream.getTracks().forEach(t => t.stop());
        webcamStream = null;
    }
    document.getElementById('webcamVideo').srcObject = null;
    document.getElementById('webcamOverlay').classList.remove('hidden');
    document.getElementById('btnStartCam').style.display = 'inline-flex';
    document.getElementById('btnStopCam').style.display = 'none';
    document.getElementById('btnCapture').style.display = 'none';
    capturedFrame = null;
    document.getElementById('btnDetect').disabled = true;
}

function captureFrame() {
    const video = document.getElementById('webcamVideo');
    const canvas = document.getElementById('webcamCanvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    capturedFrame = canvas.toDataURL('image/jpeg', 0.85);
    document.getElementById('btnDetect').disabled = false;
    showToast('Frame captured! Click Analyze.', 'info');
}

// ─── Emotion Detection ──────────────────────────────────────────
async function detectEmotion() {
    const btn = document.getElementById('btnDetect');
    btn.disabled = true;
    showPanel('loading');

    try {
        let response;
        if (currentMode === 'upload' && selectedFile) {
            const formData = new FormData();
            formData.append('image', selectedFile);
            response = await fetch(API_BASE + '/detect', { method: 'POST', body: formData });
        } else if (currentMode === 'webcam' && capturedFrame) {
            response = await fetch(API_BASE + '/detect/webcam', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: capturedFrame })
            });
        } else {
            showToast('No image to analyze', 'error');
            showPanel('empty');
            btn.disabled = false;
            return;
        }

        const data = await response.json();
        if (data.success && data.results && data.results.length > 0) {
            displayResults(data);
            showToast(`Detected: ${data.results[0].dominant_emotion} ${data.results[0].emoji}`, 'success');
            // Update Dashboard and History immediately
            loadDashboard();
            loadHistory();
        } else {
            showPanel('empty');
            showToast(data.error || data.message || 'No face detected', 'error');
        }
    } catch (err) {
        showPanel('empty');
        showToast('Detection failed: ' + err.message, 'error');
    }
    btn.disabled = false;
}

// ─── Display Results ────────────────────────────────────────────
function displayResults(data) {
    lastDetectionResult = data;
    const r = data.results[0];
    
    // Main emotion display
    document.getElementById('resultEmoji').textContent = r.emoji;
    document.getElementById('resultEmotion').textContent = r.dominant_emotion;
    
    // Human Detection Information
    const humanDetected = data.faces_detected > 0;
    const faceCountElement = document.getElementById('resultFaces');
    faceCountElement.textContent = humanDetected ? 
        `👤 Humans Detected: ${data.faces_detected}` : 
        '⚠️ No humans detected (mock result)';
    
    if (humanDetected) {
        faceCountElement.style.color = '#10b981';
        faceCountElement.style.fontWeight = 'bold';
    } else {
        faceCountElement.style.color = '#f59e0b';
        faceCountElement.style.fontWeight = 'normal';
    }
    
    document.getElementById('resultTime').textContent = new Date().toLocaleTimeString();

    // Confidence ring
    const pct = r.confidence;
    const circ = 326.73;
    document.getElementById('ringFill').style.strokeDashoffset = circ - (circ * pct / 100);
    document.getElementById('ringText').textContent = pct.toFixed(1) + '%';

    // Color the ring based on emotion
    document.getElementById('ringFill').style.stroke = COLOR_MAP[r.dominant_emotion] || '#8b5cf6';

    // All emotion bars with emojis
    const barsDiv = document.getElementById('resultBars');
    barsDiv.innerHTML = '<div class="emotions-title">Emotion Analysis</div>';
    const emotions = r.all_emotions;
    for (const [emotion, score] of Object.entries(emotions)) {
        barsDiv.innerHTML += `
            <div class="result-bar-row">
                <span class="result-bar-label">${EMOJI_MAP[emotion] || ''} ${emotion}</span>
                <div class="result-bar-track">
                    <div class="result-bar-fill" style="width:${score}%; background:${COLOR_MAP[emotion]}"></div>
                </div>
                <span class="result-bar-val">${score.toFixed(1)}%</span>
            </div>`;
    }
    
    // Add bounding box info if available
    if (r.bounding_box) {
        const bbox = r.bounding_box;
        barsDiv.innerHTML += `
            <div class="bbox-info">
                <div class="bbox-title">Face Region</div>
                <div class="bbox-coords">
                    Position: (${bbox.x}, ${bbox.y}) | Size: ${bbox.width}×${bbox.height}px
                </div>
            </div>`;
    }
    
    showPanel('content');
}

function showPanel(which) {
    document.getElementById('resultsEmpty').style.display = which === 'empty' ? 'block' : 'none';
    document.getElementById('resultsContent').style.display = which === 'content' ? 'block' : 'none';
    document.getElementById('resultsLoading').style.display = which === 'loading' ? 'block' : 'none';
}

// ─── Dashboard Stats ────────────────────────────────────────────
async function loadDashboard() {
    try {
        const res = await fetch(API_BASE + '/stats');
        const data = await res.json();
        if (!data.success) return;
        const s = data.stats;

        document.getElementById('statTotal').textContent = s.totalDetections;
        document.getElementById('statConfidence').textContent = s.averageConfidence.toFixed(1) + '%';
        document.getElementById('statTopEmotion').textContent = s.mostCommonEmotion;
        document.getElementById('statTopEmoji').textContent = EMOJI_MAP[s.mostCommonEmotion] || '😐';

        // Update emotion bars
        const total = Object.values(s.emotionDistribution).reduce((a, b) => a + b, 0) || 1;
        for (const [emotion, count] of Object.entries(s.emotionDistribution)) {
            const row = document.querySelector(`.emotion-bar-row[data-emotion="${emotion}"]`);
            if (row) {
                const pct = (count / total * 100).toFixed(1);
                row.querySelector('.bar-fill').style.width = pct + '%';
                row.querySelector('.bar-value').textContent = count;
            }
        }

        // Recent activity
        const list = document.getElementById('recentList');
        if (s.recentDetections.length === 0) {
            list.innerHTML = '<div class="empty-state-small">No detections yet. Start detecting!</div>';
        } else {
            list.innerHTML = s.recentDetections.map(d => `
                <div class="recent-item">
                    <span class="recent-emoji">${d.emoji || EMOJI_MAP[d.emotion]}</span>
                    <div class="recent-info">
                        <div class="recent-emotion">${d.emotion}</div>
                        <div class="recent-time">${new Date(d.createdAt).toLocaleString()} · ${d.inputMethod}</div>
                    </div>
                    <span class="recent-conf">${d.confidence.toFixed(1)}%</span>
                </div>`).join('');
        }
    } catch (err) {
        console.error('Dashboard load error:', err);
    }
}

// ─── History ────────────────────────────────────────────────────
async function loadHistory() {
    try {
        const res = await fetch(API_BASE + '/history?limit=50');
        const data = await res.json();
        const tbody = document.getElementById('historyBody');
        if (!data.success || data.detections.length === 0) {
            tbody.innerHTML = '<tr class="empty-row"><td colspan="6"><div class="empty-state-small">No detection history yet</div></td></tr>';
            return;
        }
        tbody.innerHTML = data.detections.map((d, i) => `
            <tr>
                <td>${i + 1}</td>
                <td><span class="emotion-badge ${d.emotion}">${EMOJI_MAP[d.emotion] || ''} ${d.emotion}</span></td>
                <td>${d.confidence.toFixed(1)}%</td>
                <td><span class="method-badge">${d.inputMethod === 'webcam' ? '📷' : '📤'} ${d.inputMethod}</span></td>
                <td>${d.facesDetected}</td>
                <td>${new Date(d.createdAt).toLocaleString()}</td>
            </tr>`).join('');
    } catch (err) {
        console.error('History load error:', err);
    }
}

async function clearHistory() {
    if (!confirm('Clear all detection history?')) return;
    try {
        await fetch(API_BASE + '/history', { method: 'DELETE' });
        showToast('History cleared', 'success');
        loadHistory();
        loadDashboard();
    } catch (err) {
        showToast('Failed to clear history', 'error');
    }
}

// ─── Health Check ───────────────────────────────────────────────
async function checkHealth() {
    try {
        const res = await fetch(API_BASE + '/health');
        const data = await res.json();
        const dot = document.getElementById('statusDot');
        const txt = document.getElementById('statusText');
        const allGood = data.database === 'connected' && data.aiService === 'healthy';
        dot.className = 'status-dot ' + (allGood ? 'connected' : 'disconnected');
        txt.textContent = allGood ? 'All Systems Online' :
            data.aiService !== 'healthy' ? 'AI Service Offline' : 'DB Offline';
    } catch {
        document.getElementById('statusDot').className = 'status-dot disconnected';
        document.getElementById('statusText').textContent = 'Backend Offline';
    }
}

// ─── Toast Notifications ────────────────────────────────────────
function showToast(msg, type = 'info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = msg;
    container.appendChild(toast);
    setTimeout(() => { toast.style.opacity = '0'; setTimeout(() => toast.remove(), 300); }, 3500);
}

// ─── Init ───────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    loadDashboard();
    checkHealth();
    setInterval(checkHealth, 15000);
});
