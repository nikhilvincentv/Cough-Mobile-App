import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load .env from backend root (one level up from src)
const envPath = path.resolve(__dirname, '../.env');
console.log(`Loading environment config from: ${envPath}`);
const result = dotenv.config({ path: envPath });

if (result.error) {
    console.error("Error loading .env file:", result.error);
} else {
    // Debug log specific keys to verify loading
    console.log("Environment loaded. ML_SERVICE_URL:", process.env.ML_SERVICE_URL);
    console.log("Environment loaded. DATABASE_URL:", process.env.DATABASE_URL ? "Set" : "Unset");
}
