/**
 * EmotionSense - Detection Routes
 * =================================
 * API routes for emotion detection and history management.
 * 
 * Routes:
 *   POST /api/detect         - Detect emotion from uploaded image
 *   POST /api/detect/webcam  - Detect emotion from webcam base64 image
 *   GET  /api/history        - Get detection history
 *   DELETE /api/history      - Clear detection history
 *   GET  /api/stats          - Get emotion detection statistics
 */

const express = require('express');
const router = express.Router();
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');
const Detection = require('../models/Detection');

// ─── In-Memory Fallback ─────────────────────────────────────────
// If MongoDB is unavailable (e.g., connection blocked or not running),
// we store detection history in memory so the dashboard continues to work.
let fallbackHistory = [];

// Configure multer for in-memory file storage
const upload = multer({
    storage: multer.memoryStorage(),
    limits: { fileSize: 10 * 1024 * 1024 }, // 10MB limit
    fileFilter: (req, file, cb) => {
        const allowed = ['image/jpeg', 'image/png', 'image/webp', 'image/gif'];
        if (allowed.includes(file.mimetype)) {
            cb(null, true);
        } else {
            cb(new Error('Only image files (JPEG, PNG, WebP, GIF) are allowed'));
        }
    }
});

const AI_SERVICE_URL = process.env.AI_SERVICE_URL || 'http://127.0.0.1:5000';

// ─── POST /api/detect ──────────────────────────────────────────
// Detect emotion from an uploaded image file
router.post('/detect', upload.single('image'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No image file provided' });
        }

        // Forward image to Python AI service
        const formData = new FormData();
        formData.append('image', req.file.buffer, {
            filename: req.file.originalname,
            contentType: req.file.mimetype
        });

        const aiResponse = await axios.post(
            `${AI_SERVICE_URL}/api/detect`,
            formData,
            {
                headers: formData.getHeaders(),
                timeout: 30000 // 30s timeout
            }
        );

        const result = aiResponse.data;

        // Try to save to database if detection was successful (non-blocking)
        if (result.success && result.results && result.results.length > 0) {
            const primary = result.results[0];

            // Create thumbnail from uploaded image (base64)
            const thumbnail = `data:${req.file.mimetype};base64,${req.file.buffer.toString('base64')}`;

            try {
                const mongoose = require('mongoose');
                if (mongoose.connection.readyState === 1) {
                    const detection = new Detection({
                        emotion: primary.dominant_emotion,
                        confidence: primary.confidence,
                        emoji: primary.emoji,
                        allEmotions: primary.all_emotions,
                        facesDetected: result.faces_detected,
                        inputMethod: 'upload',
                        thumbnail: thumbnail.length < 500000 ? thumbnail : null
                    });

                    // Save synchronously to ensure the record is available for history
                    await detection.save();
                    result.detection_id = detection._id;
                } else {
                    // Fallback to in-memory array
                    const detection = {
                        _id: 'local_' + Date.now(),
                        emotion: primary.dominant_emotion,
                        confidence: primary.confidence,
                        emoji: primary.emoji,
                        allEmotions: primary.all_emotions,
                        facesDetected: result.faces_detected,
                        inputMethod: 'upload',
                        createdAt: new Date(),
                        thumbnail: thumbnail.length < 500000 ? thumbnail : null
                    };
                    fallbackHistory.unshift(detection);
                    if (fallbackHistory.length > 200) fallbackHistory.pop(); // Keep last 200
                    result.detection_id = detection._id;
                }
            } catch (dbErr) {
                console.error('Database error (non-blocking):', dbErr.message);
                // Continue - detection still succeeded, just can't save to DB
            }
        }

        res.json(result);

    } catch (error) {
        console.error('Detection error:', error.message);

        if (error.code === 'ECONNREFUSED') {
            return res.status(503).json({
                success: false,
                error: 'AI service is not running. Please start the Python AI service.'
            });
        }

        res.status(500).json({
            success: false,
            error: error.message || 'Detection failed'
        });
    }
});


// ─── POST /api/detect/webcam ───────────────────────────────────
// Detect emotion from a webcam base64 image
router.post('/detect/webcam', async (req, res) => {
    try {
        const { image } = req.body;

        if (!image) {
            return res.status(400).json({ error: 'No image data provided' });
        }

        // Forward base64 image to Python AI service
        const aiResponse = await axios.post(
            `${AI_SERVICE_URL}/api/detect`,
            { image },
            { timeout: 30000 }
        );

        const result = aiResponse.data;

        // Try to save to database if detection was successful (non-blocking)
        if (result.success && result.results && result.results.length > 0) {
            const primary = result.results[0];

            try {
                const mongoose = require('mongoose');
                if (mongoose.connection.readyState === 1) {
                    const detection = new Detection({
                        emotion: primary.dominant_emotion,
                        confidence: primary.confidence,
                        emoji: primary.emoji,
                        allEmotions: primary.all_emotions,
                        facesDetected: result.faces_detected,
                        inputMethod: 'webcam',
                        thumbnail: image.length < 500000 ? image : null
                    });

                    // Save synchronously to ensure the record is available for history
                    await detection.save();
                    result.detection_id = detection._id;
                } else {
                    // Fallback to in-memory array
                    const detection = {
                        _id: 'local_' + Date.now(),
                        emotion: primary.dominant_emotion,
                        confidence: primary.confidence,
                        emoji: primary.emoji,
                        allEmotions: primary.all_emotions,
                        facesDetected: result.faces_detected,
                        inputMethod: 'webcam',
                        createdAt: new Date(),
                        thumbnail: image.length < 500000 ? image : null
                    };
                    fallbackHistory.unshift(detection);
                    if (fallbackHistory.length > 200) fallbackHistory.pop(); // Keep last 200
                    result.detection_id = detection._id;
                }
            } catch (dbErr) {
                console.error('Database error (non-blocking):', dbErr.message);
                // Continue - detection still succeeded, just can't save to DB
            }
        }

        res.json(result);

    } catch (error) {
        console.error('Webcam detection error:', error.message);

        if (error.code === 'ECONNREFUSED') {
            return res.status(503).json({
                success: false,
                error: 'AI service is not running. Please start the Python AI service.'
            });
        }

        res.status(500).json({
            success: false,
            error: error.message || 'Detection failed'
        });
    }
});


// ─── GET /api/history ──────────────────────────────────────────
// Get detection history (newest first)
router.get('/history', async (req, res) => {
    try {
        const limit = parseInt(req.query.limit) || 50;
        const page = parseInt(req.query.page) || 1;
        const skip = (page - 1) * limit;

        const mongoose = require('mongoose');
        if (mongoose.connection.readyState !== 1) {
            // Serve from in-memory fallback
            const paginatedFallback = fallbackHistory.slice(skip, skip + limit);
            return res.json({
                success: true,
                detections: paginatedFallback,
                pagination: { 
                    total: fallbackHistory.length, 
                    page, 
                    limit, 
                    pages: Math.ceil(fallbackHistory.length / limit) 
                }
            });
        }

        const [detections, total] = await Promise.all([
            Detection.find()
                .sort({ createdAt: -1 })
                .skip(skip)
                .limit(limit)
                .select('-thumbnail') // Exclude thumbnails for performance
                .lean(),
            Detection.countDocuments()
        ]);

        res.json({
            success: true,
            detections,
            pagination: {
                total,
                page,
                limit,
                pages: Math.ceil(total / limit)
            }
        });

    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});


// ─── DELETE /api/history ───────────────────────────────────────
// Clear all detection history
router.delete('/history', async (req, res) => {
    try {
        const mongoose = require('mongoose');
        if (mongoose.connection.readyState !== 1) {
            const count = fallbackHistory.length;
            fallbackHistory = [];
            return res.json({
                success: true,
                message: `Cleared ${count} local detection records`
            });
        }

        const result = await Detection.deleteMany({});
        res.json({
            success: true,
            message: `Cleared ${result.deletedCount} detection records`
        });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});


// ─── GET /api/stats ────────────────────────────────────────────
// Get emotion detection statistics
router.get('/stats', async (req, res) => {
    try {
        // Check if database is connected (readyState 1 = connected)
        const mongoose = require('mongoose');
        if (mongoose.connection.readyState !== 1) {
            // Compute stats from in-memory fallback
            const totalDetections = fallbackHistory.length;
            const emotionDistribution = { 'Happy': 0, 'Sad': 0, 'Angry': 0, 'Neutral': 0, 'Surprise': 0 };
            let avgConfSum = 0;
            let mostCommon = 'None';
            let maxCount = 0;

            fallbackHistory.forEach(d => {
                if (emotionDistribution[d.emotion] !== undefined) {
                    emotionDistribution[d.emotion]++;
                }
                avgConfSum += d.confidence;
            });

            Object.entries(emotionDistribution).forEach(([emo, count]) => {
                if (count > maxCount) {
                    maxCount = count;
                    mostCommon = emo;
                }
            });

            return res.json({
                success: true,
                stats: {
                    totalDetections,
                    mostCommonEmotion: mostCommon,
                    averageConfidence: totalDetections > 0 ? Math.round((avgConfSum / totalDetections) * 100) / 100 : 0,
                    emotionDistribution,
                    recentDetections: fallbackHistory.slice(0, 10)
                }
            });
        }

        const [
            totalDetections,
            emotionCounts,
            recentDetections,
            avgConfidence
        ] = await Promise.all([
            Detection.countDocuments(),
            Detection.aggregate([
                { $group: { _id: '$emotion', count: { $sum: 1 } } },
                { $sort: { count: -1 } }
            ]),
            Detection.find()
                .sort({ createdAt: -1 })
                .limit(10)
                .select('emotion confidence emoji createdAt inputMethod')
                .lean(),
            Detection.aggregate([
                { $group: { _id: null, avgConf: { $avg: '$confidence' } } }
            ])
        ]);

        // Build emotion distribution
        const emotionDistribution = {};
        const emotions = ['Happy', 'Sad', 'Angry', 'Neutral', 'Surprise'];
        emotions.forEach(e => { emotionDistribution[e] = 0; });
        emotionCounts.forEach(item => {
            if (emotions.includes(item._id)) {
                emotionDistribution[item._id] = item.count;
            }
        });

        // Find most common emotion
        const mostCommon = emotionCounts.length > 0 ? emotionCounts[0]._id : 'None';

        res.json({
            success: true,
            stats: {
                totalDetections,
                mostCommonEmotion: mostCommon,
                averageConfidence: avgConfidence.length > 0
                    ? Math.round(avgConfidence[0].avgConf * 100) / 100
                    : 0,
                emotionDistribution,
                recentDetections
            }
        });

    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});


module.exports = router;
