document.addEventListener('DOMContentLoaded', () => {
    const analyzeBtn = document.getElementById('analyzeBtn');
    const reviewInput = document.getElementById('reviewInput');
    const errorMessage = document.getElementById('errorMessage');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const resultsSection = document.getElementById('resultsSection');
    
    const API_URL = '/api/v1/predict';

    // Session Management
    function getSessionId() {
        let sessionId = localStorage.getItem('screening_room_session');
        if (!sessionId) {
            sessionId = crypto.randomUUID();
            localStorage.setItem('screening_room_session', sessionId);
        }
        return sessionId;
    }
    const currentSessionId = getSessionId();

    analyzeBtn.addEventListener('click', async () => {
        const text = reviewInput.value.trim();
        
        if (!text) {
            errorMessage.textContent = 'Please enter a review to analyze.';
            return;
        }
        
        if (text.length < 10) {
            errorMessage.textContent = 'Review is too short. Please provide more context.';
            return;
        }

        errorMessage.textContent = '';
        
        try {
            // Show loading state
            loadingOverlay.classList.remove('hidden');
            resultsSection.classList.add('hidden');
            analyzeBtn.disabled = true;

            const response = await fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: text,
                    models: ['lr', 'lstm', 'bert'],
                    sessionId: currentSessionId
                })
            });

            if (!response.ok) {
                throw new Error(`Server returned ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            updateUI(data);
            
            // Hide loading, show results
            loadingOverlay.classList.add('hidden');
            resultsSection.classList.remove('hidden');
            
            // Scroll to results smoothly
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

        } catch (error) {
            console.error('Prediction error:', error);
            errorMessage.textContent = 'Failed to analyze review. Ensure the ML service is running.';
            loadingOverlay.classList.add('hidden');
        } finally {
            analyzeBtn.disabled = false;
        }
    });

    function deduplicateKeywords(words) {
        if (!words) return [];
        const sorted = [...words].sort((a, b) => b.length - a.length);
        const result = [];
        for (const w of sorted) {
            const lowerW = w.toLowerCase();
            if (!result.some(r => r.toLowerCase().includes(lowerW))) {
                result.push(w);
            }
        }
        return result.slice(0, 3);
    }

    function updateUI(data) {
        // Update Verdict Banner
        const stamp = document.getElementById('verdictStamp');
        const verdictLabel = data.overall.label.toUpperCase();
        stamp.textContent = verdictLabel;
        stamp.setAttribute('data-verdict', verdictLabel);
        stamp.className = `stamp ${data.overall.label.toLowerCase() === 'positive' ? 'approved' : 'panned'}`;
        
        document.getElementById('verdictAgreement').textContent = data.overall.agreement;
        document.getElementById('timingMs').textContent = data.timing.total_ms.toFixed(0);

        // Update LR Critic
        if (data.models.lr) {
            updateCriticCard('criticLr', data.models.lr);
            
            // Update keywords specific to LR
            const posList = document.querySelector('#criticLr .pos-keywords ul');
            const negList = document.querySelector('#criticLr .neg-keywords ul');
            
            posList.innerHTML = '';
            negList.innerHTML = '';
            
            if (data.models.lr.top_positive_words) {
                const words = deduplicateKeywords(data.models.lr.top_positive_words);
                words.forEach(word => {
                    const li = document.createElement('li');
                    li.textContent = word;
                    posList.appendChild(li);
                });
            }
            
            if (data.models.lr.top_negative_words) {
                const words = deduplicateKeywords(data.models.lr.top_negative_words);
                words.forEach(word => {
                    const li = document.createElement('li');
                    li.textContent = word;
                    negList.appendChild(li);
                });
            }
        }

        // Update LSTM Critic
        if (data.models.lstm) {
            updateCriticCard('criticLstm', data.models.lstm);
        }

        // Update BERT Critic
        if (data.models.bert) {
            updateCriticCard('criticBert', data.models.bert);
        }
    }

    function updateCriticCard(cardId, modelData) {
        const card = document.getElementById(cardId);
        if (!card) return;

        const chip = card.querySelector('.prediction-chip');
        chip.textContent = modelData.label;
        chip.className = `prediction-chip ${modelData.label.toLowerCase()}`;

        const confValue = card.querySelector('.conf-value');
        const confBar = card.querySelector('.confidence-bar-fill');
        
        const confPercent = (modelData.confidence * 100).toFixed(1) + '%';
        if (confValue) confValue.textContent = confPercent;
        if (confBar) confBar.style.width = confPercent;

        const reason = card.querySelector('.reasoning');
        reason.textContent = modelData.reasoning;

        const latency = card.querySelector('.critic-footer span');
        latency.textContent = `${modelData.latency_ms.toFixed(1)}ms`;
    }
});
