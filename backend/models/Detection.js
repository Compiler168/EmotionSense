/**
 * EmotionSense - Detection Schema
 * ================================
 * Mongoose schema for storing emotion detection history.
 * Each document represents one detection session with:
 *   - Detected emotion and confidence
 *   - All emotion scores
 *   - Timestamp for history tracking
 *   - Input method (upload/webcam)
 */

const mongoose = require('mongoose');

const detectionSchema = new mongoose.Schema({
    // Primary detected emotion
    emotion: {
        type: String,
        required: true,
        enum: ['Happy', 'Sad', 'Angry', 'Neutral', 'Surprise']
    },

    // Confidence percentage (0-100)
    confidence: {
        type: Number,
        required: true,
        min: 0,
        max: 100
    },

    // Emoji representation
    emoji: {
        type: String,
        default: '😐'
    },

    // All emotion scores from the model
    allEmotions: {
        Happy: { type: Number, default: 0 },
        Sad: { type: Number, default: 0 },
        Angry: { type: Number, default: 0 },
        Neutral: { type: Number, default: 0 },
        Surprise: { type: Number, default: 0 }
    },

    // Number of faces detected
    facesDetected: {
        type: Number,
        default: 1
    },

    // Input method
    inputMethod: {
        type: String,
        enum: ['upload', 'webcam'],
        default: 'upload'
    },

    // Thumbnail of the analyzed image (base64, compressed)
    thumbnail: {
        type: String,
        default: null
    },

    // Timestamp
    createdAt: {
        type: Date,
        default: Date.now
    }
});

// Index for efficient history queries (newest first)
detectionSchema.index({ createdAt: -1 });

module.exports = mongoose.model('Detection', detectionSchema);
