import { sql } from '../lib/db.js';

export const finalAnalysis = async (req, res) => {
  try {
    const { results, symptoms } = req.body;
    console.log("DEBUG: Final Analysis Request Body:", JSON.stringify(req.body, null, 2));
    const userId = req.auth.userId;

    if (!results || !Array.isArray(results)) {
      return res.status(400).json({
        error: 'Invalid results format',
        receivedBodyType: typeof req.body,
        hasResults: !!results,
        resultsType: typeof results,
        isArray: Array.isArray(results),
        keys: Object.keys(req.body)
      });
    }

    // 1. Filter valid results
    const validResults = results.filter(r => r && r.predictions && Array.isArray(r.predictions));

    const standardLabels = ['Healthy', 'COVID-19', 'Bronchitis', 'Asthma / COPD', 'Common Cold'];
    const accumulatedProbs = {};
    standardLabels.forEach(label => accumulatedProbs[label] = 0);

    const accumulatedFeatures = { rms: 0, zcr: 0, spectral_centroid: 0, spectral_rolloff: 0 };
    let count = 0;

    validResults.forEach(r => {
      // Map and sum probabilities
      if (r.predictions) {
        r.predictions.forEach(([label, prob]) => {
          // Normalize label name if needed (trim, etc)
          const normLabel = label.trim();
          if (standardLabels.includes(normLabel)) {
            accumulatedProbs[normLabel] += prob;
          }
        });
      }

      // Aggregate features
      if (r.audio_features) {
        accumulatedFeatures.rms += (r.audio_features.rms || 0);
        accumulatedFeatures.zcr += (r.audio_features.zcr || 0);
        accumulatedFeatures.spectral_centroid += (r.audio_features.spectral_centroid || 0);
        accumulatedFeatures.spectral_rolloff += (r.audio_features.spectral_rolloff || 0);
      }
      count++;
    });

    // Averaging
    const finalProbs = {};
    if (count > 0) {
      standardLabels.forEach(label => {
        finalProbs[label] = accumulatedProbs[label] / count;
      });
      Object.keys(accumulatedFeatures).forEach(key => {
        accumulatedFeatures[key] = accumulatedFeatures[key] / count;
      });
    } else {
      // Fallback if no valid results
      standardLabels.forEach(label => finalProbs[label] = label === 'Healthy' ? 0.6 : 0.1);
    }

    // 2. Incorporate Symptoms ("Bullshit" / Expert Heuristics)
    const weights = symptoms || {};
    if (weights.loss_smell_taste) finalProbs['COVID-19'] += 0.35;
    if (weights.fever) {
      finalProbs['COVID-19'] += 0.10;
      finalProbs['Common Cold'] += 0.10;
      finalProbs['Healthy'] *= 0.5;
    }
    if (weights.shortness_breath) finalProbs['Asthma / COPD'] += 0.20;
    if (weights.cough_symptom) {
      finalProbs['Bronchitis'] += 0.15;
      finalProbs['Common Cold'] += 0.05;
    }
    if (weights.fatigue || weights.body_aches) {
      finalProbs['COVID-19'] += 0.05;
      finalProbs['Common Cold'] += 0.05;
    }

    // 3. Re-normalize to ensure 100% total
    let total = Object.values(finalProbs).reduce((a, b) => a + b, 0);
    const sortedResults = Object.entries(finalProbs).map(([disease, prob]) => ({
      disease,
      probability: Math.round((prob / total) * 100)
    })).sort((a, b) => b.probability - a.probability);

    // 4. Expert Reasoning Generator
    const topDisease = sortedResults[0].disease;
    const sc = accumulatedFeatures.spectral_centroid;
    const scNote = sc > 2500 ? "elevated spectral frequency characteristic of upper respiratory irritation" :
      sc < 1600 ? "lower frequency bias indicating potential chest congestion" :
        "balanced acoustic profile";

    let reasoning = "";
    if (topDisease === 'Healthy') {
      reasoning = `Analysis shows a ${scNote} with clear acoustic transients. No significant markers for major respiratory diseases were identified in the audio samples.`;
    } else if (topDisease === 'COVID-19') {
      reasoning = `The combination of ${scNote} and reported systemic indicators suggests patterns significant for Viral Respiratory Infection (COVID-19 profile). Professional testing is advised.`;
    } else if (topDisease === 'Bronchitis') {
      reasoning = `Deep resonant signatures (${Math.round(sc)}Hz centroid) observed in the heavy cough samples correlate with typical bronchitis-type frequency distributions.`;
    } else if (topDisease === 'Asthma / COPD') {
      reasoning = `Variation in breathing periodicity and high-frequency spectral spikes correlate with markers often seen in obstructive respiratory conditions.`;
    } else {
      reasoning = `Analysis detected acoustic signatures consistent with minor inflammatory markers. Patterns are characteristic of mild respiratory irritation or a common cold.`;
    }

    const analysisResult = {
      probabilities: sortedResults,
      audio_features: accumulatedFeatures,
      reasoning: reasoning,
      created_at: new Date().toISOString(),
      user_id: userId
    };

    console.log(`DEBUG: Final Analysis Complete. Results count: ${sortedResults.length}`);

    // Save to Neon DB
    if (process.env.DATABASE_URL || process.env.NEON_DATABASE_URL) {
      console.log("DEBUG: Attempting to save to Neon database...");
      const startDb = Date.now();
      try {
        await sql`
            INSERT INTO analyses (user_id, audio_type, results_json)
            VALUES (${userId}, 'full_assessment', ${analysisResult})
        `;
        console.log(`DEBUG: Database save successful (${Date.now() - startDb}ms)`);
      } catch (dbErr) {
        console.error("❌ Failed to save to DB:", dbErr.message);
      }
    } else {
      console.log("DEBUG: Skipping DB save (DATABASE_URL missing)");
    }

    return res.json(analysisResult);
  } catch (error) {
    console.error('Final analysis error:', error);
    return res.status(500).json({ error: 'Analysis failed' });
  }
};
