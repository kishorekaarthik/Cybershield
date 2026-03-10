/**
 * CyberShield - Twitter/X Content Script
 * Injects real-time toxicity analysis into Twitter's compose box.
 * Acid / Paper / Ink theme.
 */

(function () {
    'use strict';

    let debounceTimer = null;
    let overlayInjected = new WeakSet();
    let lastAnalyzedText = '';
    let currentScore = 0;
    let warningShown = false;

    // =========================================
    // 1. OBSERVE THE DOM FOR COMPOSE BOX
    // =========================================
    function init() {
        const observer = new MutationObserver(() => {
            findAndInjectOverlay();
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true
        });

        findAndInjectOverlay();
        console.log('CyberShield: Extension loaded');
    }

    // =========================================
    // 2. FIND TWITTER'S COMPOSE BOX
    // =========================================
    function findAndInjectOverlay() {
        const selectors = [
            '[data-testid="tweetTextarea_0"]',
            '[data-testid="tweetTextarea_1"]',
            '[role="textbox"][data-testid]',
            '.DraftEditor-root',
            '[contenteditable="true"][role="textbox"]'
        ];

        const processed = new Set();

        for (const selector of selectors) {
            const editors = document.querySelectorAll(selector);
            editors.forEach(editor => {
                if (processed.has(editor)) return;
                processed.add(editor);

                const container = findComposeContainer(editor);
                if (!container) return;
                if (overlayInjected.has(container)) return;
                if (container.querySelector('.cybershield-overlay')) return;

                injectOverlay(container, editor);
                overlayInjected.add(container);
                interceptTweetButton(container, editor);
            });
        }
    }

    function findComposeContainer(editor) {
        let el = editor;
        for (let i = 0; i < 15; i++) {
            if (!el.parentElement) break;
            el = el.parentElement;
            if (el.querySelector('[data-testid="tweetButton"]') ||
                el.querySelector('[data-testid="tweetButtonInline"]') ||
                el.querySelector('[role="group"]')) {
                return el;
            }
        }
        return editor.closest('[data-testid]') || editor.parentElement?.parentElement;
    }

    // =========================================
    // 3. INJECT THE OVERLAY
    // =========================================
    function injectOverlay(container, editor) {
        const overlay = document.createElement('div');
        overlay.className = 'cybershield-overlay';
        overlay.innerHTML = `
            <!-- Decorative Markers -->
            <div class="cs-marker cs-m-tl"></div>
            <div class="cs-marker cs-m-br"></div>

            <!-- Left: Shield Content -->
            <div class="cs-shield-content">
                <div class="cs-shield-header">
                    <div class="cs-brand-mark">
                        <div class="cs-star-icon"><span class="cs-star-ring"></span></div>
                        CYBERSHIELD
                    </div>
                    <div class="cs-status-indicator" id="csStatusInd">
                        ID: ${String(Math.floor(Math.random() * 999)).padStart(3, '0')} // ACTIVE<br>
                        REAL-TIME SCAN
                    </div>
                </div>

                <div class="cs-analysis-grid">
                    <div class="cs-score-block">
                        <div class="cs-score-glass" id="csScoreGlass">
                            <span class="cs-score-value" id="csScoreValue">0%</span>
                            <span class="cs-score-label" id="csScoreLabel">Toxicity<br>Level</span>
                        </div>
                    </div>
                    <div class="cs-feedback-text" id="csFeedback">
                        Awaiting input. Begin typing to activate real-time content analysis.
                    </div>
                </div>

                <div class="cs-logic-lines">
                    <div class="cs-line-curve"></div>
                    <div class="cs-line-dot"></div>
                </div>
            </div>

            <!-- Right: Actions Panel -->
            <div class="cs-actions-panel">
                <button class="cs-action-circle" id="csRewriteBtn" title="Auto-rewrite">
                    <div class="cs-action-icon"></div>
                </button>
                <div class="cs-rewrite-label">Auto<br>Fix</div>
            </div>

            <!-- Detail Panel (expandable) -->
            <div class="cs-detail-panel" id="csDetails"></div>

            <!-- Rewrite Panel -->
            <div class="cs-rewrite-panel" id="csRewritePanel">
                <div class="cs-rewrite-header">SUGGESTED REVISION</div>
                <div class="cs-rewrite-body" id="csRewriteText"></div>
                <div class="cs-rewrite-actions">
                    <button class="cs-pill-btn primary" id="csUseBtn">APPLY</button>
                    <button class="cs-pill-btn secondary" id="csDismissBtn">DISMISS</button>
                </div>
            </div>
        `;

        container.appendChild(overlay);

        // Get references
        const scoreValue = overlay.querySelector('#csScoreValue');
        const scoreLabel = overlay.querySelector('#csScoreLabel');
        const scoreGlass = overlay.querySelector('#csScoreGlass');
        const feedbackText = overlay.querySelector('#csFeedback');
        const statusInd = overlay.querySelector('#csStatusInd');
        const detailPanel = overlay.querySelector('#csDetails');
        const rewriteBtn = overlay.querySelector('#csRewriteBtn');
        const rewritePanel = overlay.querySelector('#csRewritePanel');
        const rewriteText = overlay.querySelector('#csRewriteText');
        const useBtn = overlay.querySelector('#csUseBtn');
        const dismissBtn = overlay.querySelector('#csDismissBtn');

        // Toggle details on header click
        const header = overlay.querySelector('.cs-shield-header');
        header.style.cursor = 'pointer';
        header.addEventListener('click', () => {
            detailPanel.classList.toggle('visible');
        });

        // Dismiss rewrite
        dismissBtn.addEventListener('click', () => {
            rewritePanel.classList.remove('visible');
        });

        // Listen for input
        const handleInput = () => {
            const text = getEditorText(editor);
            if (text === lastAnalyzedText) return;
            lastAnalyzedText = text;

            clearTimeout(debounceTimer);
            if (!text.trim()) {
                resetOverlay(overlay, scoreValue, scoreLabel, feedbackText, detailPanel, rewritePanel);
                return;
            }
            debounceTimer = setTimeout(() => {
                analyzeText(text, overlay, scoreValue, scoreLabel, feedbackText, detailPanel);
            }, 400);
        };

        editor.addEventListener('input', handleInput);
        const inputObserver = new MutationObserver(handleInput);
        inputObserver.observe(editor, { childList: true, subtree: true, characterData: true });

        // Rewrite button
        rewriteBtn.addEventListener('click', () => {
            const text = getEditorText(editor);
            if (!text.trim()) return;

            rewriteBtn.disabled = true;

            if (!chrome.runtime?.sendMessage) {
                console.warn('CyberShield: Extension context invalid — please refresh the page.');
                rewriteBtn.disabled = false;
                return;
            }

            chrome.runtime.sendMessage({ type: 'REWRITE', text }, (response) => {
                if (chrome.runtime.lastError) {
                    console.warn('CyberShield: Connection error -', chrome.runtime.lastError.message);
                    rewriteBtn.disabled = false;
                    return;
                }
                if (response && response.success) {
                    const data = response.data;
                    rewriteText.textContent = data.rewritten;
                    rewritePanel.classList.add('visible');

                    useBtn.onclick = () => {
                        setEditorText(editor, data.rewritten);
                        rewritePanel.classList.remove('visible');
                        // Re-inject overlay after React re-render removes it
                        setTimeout(() => {
                            if (!container.querySelector('.cybershield-overlay')) {
                                container.appendChild(overlay);
                            }
                        }, 150);
                    };
                } else {
                    console.error('CyberShield rewrite failed:', response?.error);
                }
                rewriteBtn.disabled = false;
            });
        });

        console.log('CyberShield: Overlay injected into compose box');
    }

    // =========================================
    // 4. ANALYZE TEXT VIA API
    // =========================================
    function analyzeText(text, overlay, scoreValue, scoreLabel, feedbackText, detailPanel) {
        if (!chrome.runtime?.sendMessage) {
            console.warn('CyberShield: Extension context invalid — please refresh the page.');
            return;
        }

        chrome.runtime.sendMessage({ type: 'ANALYZE', text }, (response) => {
            if (chrome.runtime.lastError) {
                console.warn('CyberShield: Connection error -', chrome.runtime.lastError.message);
                return;
            }
            if (response && response.success) {
                updateOverlay(response.data, overlay, scoreValue, scoreLabel, feedbackText, detailPanel);
            } else {
                console.warn('CyberShield: Analysis returned no data');
            }
        });
    }

    function updateOverlay(data, overlay, scoreValue, scoreLabel, feedbackText, detailPanel) {
        const score = data.score;
        currentScore = score;
        const isHigh = score >= 60;
        const isMedium = score >= 30;

        // Score
        scoreValue.textContent = score + '%';

        // Label
        if (isHigh) {
            scoreLabel.innerHTML = 'Toxicity<br>Detected';
        } else if (isMedium) {
            scoreLabel.innerHTML = 'Moderate<br>Risk';
        } else {
            scoreLabel.innerHTML = 'Toxicity<br>Level';
        }

        // Threat level class (controls pulse animation)
        overlay.classList.remove('threat-high');
        if (isHigh) overlay.classList.add('threat-high');

        // Feedback text
        const label = data.label || 'Not Bullying';
        if (isHigh) {
            feedbackText.textContent = 'Content may violate platform safety guidelines.';
        } else if (isMedium) {
            feedbackText.textContent = 'Moderate risk detected (' + label + '). Consider revising phrasing for clarity and tone.';
        } else {
            feedbackText.textContent = 'Content appears safe. No harmful patterns or hostile language detected.';
        }

        // Explanations
        if (data.explanation && data.explanation.length > 0) {
            detailPanel.innerHTML = data.explanation.map(e =>
                `<div class="cs-detail-item">${escapeHtml(e)}</div>`
            ).join('');
        }
    }

    function resetOverlay(overlay, scoreValue, scoreLabel, feedbackText, detailPanel, rewritePanel) {
        scoreValue.textContent = '0%';
        scoreLabel.innerHTML = 'Toxicity<br>Level';
        feedbackText.textContent = 'Awaiting input. Begin typing to activate real-time content analysis.';
        overlay.classList.remove('threat-high');
        detailPanel.classList.remove('visible');
        detailPanel.innerHTML = '';
        rewritePanel.classList.remove('visible');
        lastAnalyzedText = '';
        currentScore = 0;
    }

    // =========================================
    // 5b. TWEET BUTTON FRICTION WARNING
    // =========================================
    function interceptTweetButton(container, editor) {
        const checkBtn = () => {
            const btn = container.querySelector('[data-testid="tweetButton"]') ||
                container.querySelector('[data-testid="tweetButtonInline"]');
            if (!btn || btn.dataset.csIntercepted) return;
            btn.dataset.csIntercepted = 'true';

            btn.addEventListener('click', (e) => {
                if (warningShown) return; // User already chose "Post Anyway"
                if (currentScore < 60) return; // Safe — let it through

                // Block the tweet
                e.preventDefault();
                e.stopImmediatePropagation();
                showWarningModal(container, editor, btn);
            }, true); // Capture phase to intercept before Twitter
        };

        checkBtn();
        // Re-check periodically since Twitter re-renders buttons
        const btnObserver = new MutationObserver(checkBtn);
        btnObserver.observe(container, { childList: true, subtree: true });
    }

    function showWarningModal(container, editor, tweetBtn) {
        // Remove existing modal if any
        const existing = container.querySelector('.cs-warning-modal');
        if (existing) existing.remove();

        const modal = document.createElement('div');
        modal.className = 'cs-warning-modal';
        modal.innerHTML = `
            <div class="cs-warning-backdrop"></div>
            <div class="cs-warning-card">
                <div class="cs-warning-icon">⚠️</div>
                <div class="cs-warning-title">HOSTILE CONTENT DETECTED</div>
                <div class="cs-warning-score">Toxicity Score: ${currentScore}%</div>
                <div class="cs-warning-message">
                    CyberShield detected potentially harmful content.
                    Would you like to revise before posting?
                </div>
                <div class="cs-warning-actions">
                    <button class="cs-warning-btn revise">✏️ REVISE</button>
                    <button class="cs-warning-btn post-anyway">POST ANYWAY</button>
                </div>
                <div class="cs-warning-footer">This is a recommendation, not a block.</div>
            </div>
        `;

        document.body.appendChild(modal);

        // Revise — close modal, focus editor
        modal.querySelector('.cs-warning-btn.revise').addEventListener('click', () => {
            modal.remove();
            editor.focus();
        });

        // Post Anyway — close modal, re-click tweet button
        modal.querySelector('.cs-warning-btn.post-anyway').addEventListener('click', () => {
            modal.remove();
            warningShown = true;
            tweetBtn.click();
            // Reset after a tick so future tweets get checked again
            setTimeout(() => { warningShown = false; }, 500);
        });

        // Backdrop click = revise
        modal.querySelector('.cs-warning-backdrop').addEventListener('click', () => {
            modal.remove();
            editor.focus();
        });
    }

    // =========================================
    // 5. TWITTER EDITOR HELPERS
    // =========================================
    function getEditorText(editor) {
        // Clone and strip any CyberShield overlay elements to get pure tweet text
        const clone = editor.cloneNode(true);
        clone.querySelectorAll('.cybershield-overlay, .cs-warning-modal').forEach(el => el.remove());
        return (clone.textContent || clone.innerText || '').trim();
    }

    function setEditorText(editor, text) {
        editor.focus();
        const selection = window.getSelection();
        const range = document.createRange();
        range.selectNodeContents(editor);
        selection.removeAllRanges();
        selection.addRange(range);
        document.execCommand('insertText', false, text);
        editor.dispatchEvent(new Event('input', { bubbles: true }));
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // =========================================
    // 6. START
    // =========================================
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
