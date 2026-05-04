# 🏥 Cough AI - Respiratory Disease Detection

AI-powered respiratory disease detection from cough sounds using deep learning.

**Model Accuracy:** 64.4% (Balanced across all classes)
**Status:** ✅ Production-ready proof-of-concept

## 🎯 What It Does

Detects **3 respiratory diseases** from cough audio:
- ✅ **Healthy** (59.9% accuracy)
- ✅ **COVID-19** (64.2% accuracy)
- ✅ **Bronchitis** (67.1% accuracy)

**Overall Accuracy: 64.4%** (Balanced across all classes)

## 🚀 Quick Start (3 Terminals)

```bash
# Terminal 1: ML Service
cd backend/ml_service
source ../../.venv/bin/activate
python app.py

# Terminal 2: Backend API
cd backend
npm start

# Terminal 3: Mobile App
cd mobile
npx expo start
```

**See [RUN_SYSTEM.md](./RUN_SYSTEM.md) for detailed instructions**

## 📱 Features

- **Audio Recording**: Record cough samples with high-quality audio
- **AI Analysis**: Analyze cough patterns using ML models
- **Real-time Results**: Get instant probability predictions
- **Secure Authentication**: Clerk-based user authentication
- **Rate Limiting**: Upstash Redis for API protection
- **Demo Mode**: Test without backend configuration

## 🏗️ Architecture

```
┌─────────────────┐
│   Mobile App    │
│  (React Native) │
│   + Expo        │
└────────┬────────┘
         │ HTTPS + JWT
         ▼
┌─────────────────┐
│  Backend API    │
│   (Express.js)  │
├─────────────────┤
│ • Clerk Auth    │
│ • Rate Limiting │
│ • OpenAI        │
│ • ML Models     │
└─────────────────┘
```

## 📚 Documentation

### 🎯 Essential Guides
- **[RUN_SYSTEM.md](./RUN_SYSTEM.md)** - ⭐ How to run the complete system
- **[ML_TRAINING_COMPLETE_JOURNEY.md](./ML_TRAINING_COMPLETE_JOURNEY.md)** - Complete ML training story (10 attempts)
- **[FINAL_RESULTS.md](./FINAL_RESULTS.md)** - Model performance and analysis
- **[COMPLETE_PROJECT_DOCUMENTATION.md](./COMPLETE_PROJECT_DOCUMENTATION.md)** - Technical documentation

## 🔧 Configuration

### Demo Mode (Default)
Works out of the box, no configuration needed.

### Production Mode
Set these environment variables:

**Backend** (`backend/.env`):
```env
CLERK_SECRET_KEY=sk_test_...
UPSTASH_REDIS_REST_URL=https://...
UPSTASH_REDIS_REST_TOKEN=...
OPENAI_API_KEY=sk-proj-...
PORT=4000
```

**Mobile** (`mobile/app.json`):
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

## 🛠️ Tech Stack

### Mobile
- React Native + Expo
- Clerk (Authentication)
- Expo AV (Audio Recording)
- Axios (HTTP Client)

### Backend
- Express.js
- Clerk Backend (JWT Verification)
- Upstash Redis (Rate Limiting)
- OpenAI (Whisper API)
- Multer (File Uploads)

### ML/AI
- YAMNet Transfer Learning (Google AudioSet)
- TensorFlow/Keras (64.4% accuracy)
- Coswara Dataset (Healthy, COVID)
- CoughVID Dataset (Bronchitis)

## 📊 API Endpoints

### `GET /health`
Health check endpoint

### `POST /api/analyze`
Analyze cough audio sample
- **Auth**: Bearer token required
- **Rate Limit**: 20 req/min per IP
- **Body**: `multipart/form-data` with `audio` field
- **Response**: Probability predictions

## 🔐 Security

- JWT token verification via Clerk
- Rate limiting via Upstash Redis
- Secure token storage in mobile app
- Environment variable protection

## 🎯 Recent Fixes

✅ Rate limiter lazy initialization  
✅ Proper JWT verification  
✅ API endpoint alignment  
✅ Demo mode implementation  
✅ Environment file cleanup  
✅ Enhanced error handling  
✅ Graceful service degradation  

See [CHANGES.md](./CHANGES.md) for details.

## 🧪 Testing

```bash
# Backend health check
curl http://localhost:4000/health

# Test analyze endpoint (requires auth)
curl -X POST http://localhost:4000/api/analyze \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "audio=@sample.wav"
```

## 📱 Mobile Testing

1. Start Expo: `npx expo start`
2. Press `i` for iOS simulator
3. Press `a` for Android emulator
4. Scan QR code for physical device

## 🐛 Troubleshooting

**Backend won't start?**
```bash
cd backend
rm -rf node_modules
npm install
npm run dev
```

**Mobile shows network error?**
- Enable demo mode: Set `DEMO_MODE: "true"` in `app.json`
- Or check backend is running: `curl http://localhost:4000/health`

**Audio recording fails?**
- Grant microphone permissions
- Use real device (simulator has limited audio support)

## 🚢 Deployment

### Backend
Deploy to Railway, Render, or Vercel:
```bash
cd backend
# Set environment variables in platform
# Deploy via Git or CLI
```

### Mobile
Build with Expo Application Services:
```bash
cd mobile
eas build --platform ios
eas build --platform android
```

## 📈 Project Status

- [x] Audio recording (React Native + Expo)
- [x] ML model training (10 iterations, 64.4% accuracy)
- [x] YAMNet transfer learning
- [x] Targeted data augmentation
- [x] Backend API (Express.js)
- [x] Authentication (Clerk)
- [x] Rate limiting (Upstash Redis)
- [x] Complete documentation
- [ ] Model deployment to production
- [ ] Analysis history
- [ ] Export reports

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- **Coswara Dataset** (IISc Bangalore) - Healthy, COVID, Asthma, COPD samples
- **CoughVID Dataset** (Kaggle) - Bronchitis samples
- Clerk for authentication
- Upstash for Redis hosting
- Expo team for mobile framework

## 📞 Support

- 📖 Read [RUN_SYSTEM.md](./RUN_SYSTEM.md) for setup
- 📚 Check [ML_TRAINING_COMPLETE_JOURNEY.md](./ML_TRAINING_COMPLETE_JOURNEY.md) for ML details
- 🐛 Open an issue for bugs
- 💬 Discussions for questions

---

**Built with ❤️ for healthcare innovation**

**Status**: ✅ Production-Ready Proof-of-Concept  
**Model**: YAMNet + Targeted Augmentation (64.4%)  
**Last Updated**: 2025-10-15
