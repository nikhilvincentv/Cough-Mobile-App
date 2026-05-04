# Cough AI - Respiratory Disease Detection

This project uses deep learning to detect respiratory diseases from audio recordings of coughs. It is currently a proof-of-concept with a balanced accuracy of 64.4% across all classes.

## What it does

The system analyzes cough audio to detect three possible states:
* Healthy (59.9% accuracy)
* COVID-19 (64.2% accuracy)
* Bronchitis (67.1% accuracy)

Overall, the model averages 64.4% accuracy across these three categories.

## Quick start

You will need three terminals to run the different parts of the system.

Terminal 1: ML Service
```bash
cd backend/ml_service
source ../../.venv/bin/activate
python app.py
```

Terminal 2: Backend API
```bash
cd backend
npm start
```

Terminal 3: Mobile App
```bash
cd mobile
npx expo start
```

For more detailed setup instructions, check out RUN_SYSTEM.md.

## Features

* Audio recording: Capture high-quality cough samples directly from the app.
* ML analysis: Process audio files using our trained model to find patterns.
* Instant results: Receive probability predictions in real time.
* Secure authentication: User accounts are managed securely using Clerk.
* Rate limiting: API endpoints are protected using Upstash Redis.
* Demo mode: You can test the frontend without having to set up the backend.

## System architecture

The project consists of a React Native (Expo) mobile app that communicates over HTTPS/JWT with an Express.js backend. The backend handles authentication via Clerk, rate limiting via Redis, and passes audio data to our ML models and OpenAI for analysis.

## Documentation

* RUN_SYSTEM.md: Instructions for getting the whole system up and running.
* ML_TRAINING_COMPLETE_JOURNEY.md: The full story of how we trained the model over 10 different iterations.
* FINAL_RESULTS.md: Detailed model performance metrics.
* COMPLETE_PROJECT_DOCUMENTATION.md: Full technical documentation.

## Configuration

By default, the app runs in demo mode, which requires no extra setup. If you want to run it in production mode, you need to set up some environment variables.

Backend (`backend/.env`):
```env
CLERK_SECRET_KEY=sk_test_...
UPSTASH_REDIS_REST_URL=https://...
UPSTASH_REDIS_REST_TOKEN=...
OPENAI_API_KEY=sk-proj-...
PORT=4000
```

Mobile (`mobile/app.json`):
```json
{
  "expo": {
    "extra": {
      "CLERK_PUBLISHABLE_KEY": "pk_test_...",
      "API_URL": "http://localhost:4000",
      "DEMO_MODE": "false"
    }
  }
}
```

## Tech stack

* Mobile: React Native, Expo, Clerk, Expo AV, Axios.
* Backend: Express.js, Clerk, Upstash Redis, OpenAI Whisper API, Multer.
* ML: YAMNet Transfer Learning (via Google AudioSet), TensorFlow/Keras, Coswara Dataset, CoughVID Dataset.

## API endpoints

GET /health
Returns the health status of the API.

POST /api/analyze
Analyzes a cough audio sample.
* Requires a Bearer token.
* Rate limited to 20 requests per minute per IP.
* Expects multipart/form-data with an "audio" field.
* Returns probability predictions.

## Testing

You can test the backend using curl:

```bash
# Health check
curl http://localhost:4000/health

# Test analyze endpoint (requires auth)
curl -X POST http://localhost:4000/api/analyze \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "audio=@sample.wav"
```

To test the mobile app, start Expo with `npx expo start`, then press `i` for the iOS simulator, `a` for Android, or scan the QR code with your physical device.

## Troubleshooting

If the backend won't start, try doing a fresh install of the node modules:
```bash
cd backend
rm -rf node_modules
npm install
npm run dev
```

If the mobile app shows a network error, make sure the backend is actually running by checking `http://localhost:4000/health`. Alternatively, you can enable demo mode by setting `DEMO_MODE: "true"` in your app.json.

If audio recording is failing, make sure you granted microphone permissions. It is highly recommended to use a physical device for this, as simulators can have issues with audio.

## Deployment

Backend: The Express app can be deployed to Railway, Render, or Vercel. Make sure to set up your environment variables on the platform.
Mobile: You can build the app using Expo Application Services (EAS):
```bash
cd mobile
eas build --platform ios
eas build --platform android
```

## Acknowledgments

We used the Coswara Dataset from IISc Bangalore for healthy, COVID, Asthma, and COPD samples. The Bronchitis samples came from the CoughVID Dataset on Kaggle. We also rely on Clerk for authentication and Upstash for Redis.
