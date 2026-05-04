import { neon } from '@neondatabase/serverless';
import dotenv from 'dotenv';
dotenv.config();

const sql = neon(process.env.DATABASE_URL || process.env.NEON_DATABASE_URL);

// Initialize DB table if not exists (simple migration for MVP)
export async function initDB() {
    try {
        await sql`
      CREATE TABLE IF NOT EXISTS analyses (
        id SERIAL PRIMARY KEY,
        user_id TEXT NOT NULL,
        audio_type TEXT,
        results_json JSONB,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
      );
    `;
        console.log('✅ Database table "analyses" ready');
    } catch (error) {
        console.error('❌ Database initialization failed:', error);
    }
}

export { sql };
