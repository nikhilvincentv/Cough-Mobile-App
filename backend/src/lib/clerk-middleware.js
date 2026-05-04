import { verifyToken } from '@clerk/backend';

export const clerkAuth = async (req, res, next) => {
    if (!process.env.CLERK_SECRET_KEY) {
        console.warn('⚠️ CLERK_SECRET_KEY missing. Skipping auth (Unsafe for production).');
        req.auth = { userId: 'demo_user' };
        return next();
    }

    // Get token from header
    const authHeader = req.headers.authorization;
    if (!authHeader?.startsWith('Bearer ')) {
        return res.status(401).json({ error: 'Missing bearer token' });
    }

    const token = authHeader.split(' ')[1];

    try {
        const verifiedToken = await verifyToken(token, {
            secretKey: process.env.CLERK_SECRET_KEY,
        });
        req.auth = { userId: verifiedToken.sub };
        next();
    } catch (error) {
        console.error('Clerk auth error:', error);
        return res.status(401).json({ error: 'Invalid token' });
    }
};
