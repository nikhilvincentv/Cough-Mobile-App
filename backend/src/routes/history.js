import { sql } from '../lib/db.js';

export const getHistory = async (req, res) => {
    try {
        const userId = req.auth.userId;

        // Fetch last 50 records for the user
        const history = await sql`
      SELECT * FROM analyses 
      WHERE user_id = ${userId} 
      ORDER BY created_at DESC 
      LIMIT 50
    `;

        return res.json({ history });
    } catch (error) {
        console.error('History fetch error:', error);
        return res.status(500).json({ error: 'Failed to fetch history' });
    }
};
