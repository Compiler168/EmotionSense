/**
 * EmotionSense Backend Server
 * ============================
 * Main entry point for the Node.js/Express backend.
 * 
 * Responsibilities:
 *   - MongoDB connection via Mongoose
 *   - Express middleware setup (CORS, JSON, Morgan)
 *   - Route mounting for detection API
 *   - Static file serving for frontend
 *   - Health check and AI service proxy
 */

require('dotenv').config();
const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const morgan = require('morgan');
const path = require('path');
const axios = require('axios');

const detectionRoutes = require('./routes/detection');

// ─── Configuration ──────────────────────────────────────────────
const app = express();
const PORT = process.env.PORT || 3000;
const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://127.0.0.1:27017/emotionsense';
const AI_SERVICE_URL = process.env.AI_SERVICE_URL || 'http://127.0.0.1:5000';

// ─── Middleware ─────────────────────────────────────────────────
app.use(cors());
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));
app.use(morgan('dev'));

// Serve frontend static files
app.use(express.static(path.join(__dirname, '..', 'frontend')));

// ─── API Routes ─────────────────────────────────────────────────
app.use('/api', detectionRoutes);

// ─── Health Check ───────────────────────────────────────────────
app.get('/api/health', async (req, res) => {
    let aiStatus = 'disconnected';
    try {
        const aiHealth = await axios.get(`${AI_SERVICE_URL}/api/health`, { timeout: 3000 });
        aiStatus = aiHealth.data.status || 'connected';
    } catch {
        aiStatus = 'disconnected';
    }

    res.json({
        status: 'healthy',
        service: 'EmotionSense Backend',
        database: mongoose.connection.readyState === 1 ? 'connected' : 'disconnected',
        aiService: aiStatus,
        timestamp: new Date().toISOString()
    });
});

// ─── Serve Frontend (SPA fallback) ─────────────────────────────
app.use((req, res) => {
    res.sendFile(path.join(__dirname, '..', 'frontend', 'index.html'));
});

// ─── MongoDB Connection & Server Start ──────────────────────────
async function startServer() {
    console.log('═'.repeat(50));
    console.log('  EmotionSense Backend Server');
    console.log('═'.repeat(50));

    try {
        await mongoose.connect(MONGODB_URI);
        console.log(`✓ MongoDB connected: ${MONGODB_URI}`);
    } catch (err) {
        console.error('✗ MongoDB connection failed:', err.message);
        console.log('  → Server will start without database (history disabled)');
    }

    app.listen(PORT, () => {
        console.log(`\n🚀 Backend server running on http://localhost:${PORT}`);
        console.log(`📊 Dashboard:  http://localhost:${PORT}`);
        console.log(`🔌 API:        http://localhost:${PORT}/api`);
        console.log(`🧠 AI Service: ${AI_SERVICE_URL}`);
        console.log('');
    });
}

startServer();
