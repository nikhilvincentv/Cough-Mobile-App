import express from 'express';
import cors from 'cors';
import multer from 'multer';

const app = express();
app.use(cors());
app.use(express.json({ limit: '15mb' }));

// Health check
app.get('/health', (_, res) => res.json({ ok: true }));

// Multer for audio uploads
const upload = multer({ storage: multer.memoryStorage() });

// Simple cough analysis endpoint that returns demo data
app.post('/predict', upload.single('audio'), (req, res) => {
  console.log('Received audio file:', req.file ? 'Yes' : 'No');
  
  // Simulate processing time
  setTimeout(() => {
    // Return demo analysis results
    const demoResults = {
      probs: [
        ['Healthy', 0.65],
        ['Asthma', 0.20],
        ['Bronchitis', 0.10],
        ['Pneumonia', 0.05]
      ],
      message: 'Analysis complete (demo mode)'
    };
    
    res.json(demoResults);
  }, 2000); // 2 second delay to simulate processing
});

const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`🚀 Backend server running on http://localhost:${port}`);
  console.log(`📊 Health check: http://localhost:${port}/health`);
  console.log(`🎤 Cough analysis: http://localhost:${port}/predict`);
});
