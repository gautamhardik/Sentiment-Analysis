const express = require('express');
const { predict, getHistory, getModels, getHealth, getMetrics } = require('../controllers/predict.controller');
const { validate, predictSchema } = require('../middleware/validation');
const { apiLimiter } = require('../middleware/rateLimiter');

const router = express.Router();

router.get('/health', getHealth);
router.get('/models', getModels);
router.get('/metrics', getMetrics);
router.get('/history/:sessionId', getHistory);

// Apply rate limiting and validation to the predict route
router.post('/predict', apiLimiter, validate(predictSchema), predict);

module.exports = router;
