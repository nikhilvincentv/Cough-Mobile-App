# CoughAI Mobile App

A React Native mobile app for AI-powered cough analysis using Expo and NativeWind for styling.

## Features

- 🎤 **Audio Recording**: Record cough samples with high-quality audio
- 🤖 **AI Analysis**: Analyze cough samples using machine learning
- 📊 **Results Visualization**: View analysis results with progress bars
- 🎨 **Modern UI**: Beautiful, responsive design with Tailwind CSS
- 📱 **Cross-Platform**: Works on iOS, Android, and Web

## Prerequisites

- Node.js (v16 or higher)
- npm or yarn
- Expo CLI (`npm install -g @expo/cli`)
- For mobile development: Expo Go app on your device

## Installation

1. Install dependencies:
```bash
npm install --legacy-peer-deps
```

2. Start the development server:
```bash
# For web
npm run web

# For mobile (requires Expo Go app)
npm start

# For specific platforms
npm run ios
npm run android
```

## Configuration

The app is configured to run in demo mode by default, which bypasses authentication. To enable authentication:

1. Update `app.json` with your Clerk publishable key
2. Set `DEMO_MODE = false` in `app/_layout.jsx`

## Backend Integration

The app expects a backend server running on `http://localhost:3000` with a `/predict` endpoint that accepts audio files. If the backend is not available, the app will show demo data.

## Project Structure

```
mobile/
├── app/
│   ├── (tabs)/
│   │   ├── index.jsx          # Home screen
│   │   ├── cough/
│   │   │   └── index.jsx      # Cough analysis screen
│   │   └── profile.jsx        # Profile screen
│   └── _layout.jsx            # Root layout
├── global.css                 # Tailwind CSS imports
├── tailwind.config.js         # Tailwind configuration
├── babel.config.js            # Babel configuration
└── metro.config.js            # Metro bundler configuration
```

## Troubleshooting

### Common Issues

1. **Metro bundler issues**: Clear cache with `npx expo start -c`
2. **NativeWind not working**: Ensure `global.css` is imported in `_layout.jsx`
3. **Audio recording issues**: Check microphone permissions on your device
4. **Backend connection**: Verify the backend server is running on the correct port

### Dependencies

- **Expo**: Cross-platform development framework
- **NativeWind**: Tailwind CSS for React Native
- **Expo AV**: Audio recording and playback
- **Axios**: HTTP client for API calls
- **Clerk**: Authentication (optional in demo mode)

## Development

The app uses:
- **Expo Router** for navigation
- **NativeWind** for styling (Tailwind CSS)
- **Expo AV** for audio recording
- **React Native** components

## License

This project is for demonstration purposes only. Please consult healthcare professionals for medical advice.
