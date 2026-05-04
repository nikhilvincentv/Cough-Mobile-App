import OpenAI from 'openai';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import FormData from 'form-data';
import axios from 'axios';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ML Service URL (Python Flask service)
const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:5000';

let openai = null;

// Initialize OpenAI client safely
if (process.env.OPENAI_API_KEY) {
  openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY
  });
} else {
  console.warn("⚠️ OPENAI_API_KEY is missing. OpenAI routes will not work.");
}

// Analyze audio using trained PyTorch models via Python ML service
export const analyzeCough = async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "No audio file uploaded" });
  }

  try {
    const audioType = req.body.audio_type || 'cough-heavy';
    // Use ML_SERVICE_URL from env (resolved at runtime)
    const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:5000';

    console.log(`Analyzing audio: ${req.file.originalname}, type: ${audioType}, size: ${req.file.size} bytes, target: ${ML_SERVICE_URL}`);

    // Try to use Python ML service with your trained models
    try {
      const formData = new FormData();
      formData.append('audio', req.file.buffer, {
        filename: req.file.originalname,
        contentType: req.file.mimetype
      });
      formData.append('audio_type', audioType);

      const mlResponse = await axios.post(`${ML_SERVICE_URL}/analyze`, formData, {
        headers: formData.getHeaders(),
        timeout: 30000
      });

      const result = mlResponse.data;

      // Return predictions with quality assessment
      return res.json({
        predictions: result.predictions,
        probs: result.predictions, // Alias for mobile app compatibility
        audio_features: result.audio_features, // Pass features through
        quality: result.quality,
        audio_type: result.audio_type,
        model_used: result.model_used,
        message: result.quality.should_retake
          ? `⚠️ ${result.quality.message} - Please retake the recording`
          : `✅ ${result.quality.message}`,
        userId: req.auth?.sub || 'unknown'
      });

    } catch (mlError) {
      console.error("ML Service error:", mlError.message);
      console.log("Falling back to demo mode");

      // Fallback to demo predictions (matching new 5-class system)
      const demoProbs = [
        ['Healthy', 0.65],
        ['COVID-19', 0.10],
        ['Bronchitis', 0.10],
        ['Asthma / COPD', 0.08],
        ['Common Cold', 0.07]
      ];

      return res.json({
        probs: demoProbs,
        predictions: demoProbs,
        quality: {
          rating: 'unknown',
          message: 'ML service unavailable - using demo data',
          should_retake: false
        },
        audio_type: audioType,
        model_used: 'demo',
        message: "Demo mode - ML service not available. Start Python service to use trained models.",
        userId: req.auth?.sub || 'unknown'
      });
    }

  } catch (err) {
    console.error("Error analyzing audio:", err);
    return res.status(500).json({
      error: "Failed to analyze audio",
      details: err.message
    });
  }
};
