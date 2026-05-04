import './load-env.js'; // MUST BE FIRST
import express from 'express';
import cors from 'cors';
import multer from 'multer';
import { ratelimiter } from './lib/ratelimit.js';
import { clerkAuth } from './lib/clerk-middleware.js';
import { analyzeCough } from './routes/analyze.js';
import { finalAnalysis } from './routes/final-analysis.js';

const app = express();
app.use(cors());

// Debug Middleware: Log all requests
app.use((req, res, next) => {
    console.log(`[REQUEST] ${req.method} ${req.url}`);
    console.log(`[REQUEST] Headers: Content-Type=${req.headers['content-type']}, Content-Length=${req.headers['content-length']}`);
    next();
});

app.use(express.json({ limit: '15mb' }));

// Debug Middleware: Log body after parsing
app.use((req, res, next) => {
    if (req.method === 'POST' && req.url.includes('final-analysis')) {
        console.log(`[DEBUG BODY] Keys: ${Object.keys(req.body)}`);
        // console.log(`[DEBUG BODY] Full: ${JSON.stringify(req.body).substring(0, 200)}...`);
    }
    next();
});

// Debug env variables (temporary, remove in production)
console.log("OPENAI_API_KEY:", process.env.OPENAI_API_KEY ? "Loaded" : "MISSING");
console.log("UPSTASH_REDIS_REST_URL:", process.env.UPSTASH_REDIS_REST_URL ? "Loaded" : "MISSING");
console.log("UPSTASH_REDIS_REST_TOKEN:", process.env.UPSTASH_REDIS_REST_TOKEN ? "Loaded" : "MISSING");
console.log("CLERK_SECRET_KEY:", process.env.CLERK_SECRET_KEY ? "Loaded" : "MISSING");

// Simple health check
app.get('/health', (_, res) => res.json({ ok: true }));

// Protect /api/* with Clerk + rate limit (Upstash)
app.use('/api', ratelimiter, clerkAuth);

// Multer for audio uploads
const upload = multer({ storage: multer.memoryStorage() });

import { getHistory } from './routes/history.js';
import { initDB } from './lib/db.js';

// ... (existing code) ...

// Initialize DB
initDB();

app.post('/api/analyze', upload.single('audio'), analyzeCough);
app.post('/api/final-analysis', finalAnalysis);
app.get('/api/history', getHistory);

const port = process.env.PORT || 4000;
app.listen(port, () => console.log(`API listening on ${port}`));
