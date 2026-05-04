import { Redis } from '@upstash/redis';
import { Ratelimit } from '@upstash/ratelimit';

// Create a new ratelimiter, that allows 10 requests per 10 seconds
export const ratelimiter = async (req, res, next) => {
    // If UPSTASH env vars are missing, skip rate limiting (dev mode safe)
    if (!process.env.UPSTASH_REDIS_REST_URL || !process.env.UPSTASH_REDIS_REST_TOKEN) {
        console.warn('⚠️ Upstash Redis credentials missing. Rate limiting disabled.');
        return next();
    }

    try {
        const redis = new Redis({
            url: process.env.UPSTASH_REDIS_REST_URL,
            token: process.env.UPSTASH_REDIS_REST_TOKEN,
        });

        const limiter = new Ratelimit({
            redis: redis,
            limiter: Ratelimit.slidingWindow(20, '1 m'),
            analytics: true,
            prefix: '@upstash/ratelimit',
        });

        // Use user ID if authenticated, or IP address if not
        const identifier = req.auth?.userId || req.ip || 'anonymous';

        const { success } = await limiter.limit(identifier);

        if (!success) {
            return res.status(429).json({
                error: 'Too Many Requests',
                message: 'You have exceeded the rate limit. Please try again later.'
            });
        }

        next();
    } catch (error) {
        console.error('Rate limit error:', error);
        // Fail open if Redis is down
        next();
    }
};
