/**
 * CyberShield - Twitter/X Content Script
 * Injects real-time toxicity analysis into Twitter's compose box.
 * 
 * Detects Twitter's compose textarea (contentEditable div) and overlays
 * a toxicity meter that updates as the user types.
 */

(function () {
    'use strict';

    let debounceTimer = null;
    let overlayInjected = new WeakSet();
    let lastAnalyzedText = '';

    // =========================================
    // 1. OBSERVE THE DOM FOR COMPOSE BOX
    // =========================================
    function init() {
        // Watch for compose box appearing (Twitter loads dynamically)
        const observer = new MutationObserver(() => {
            findAndInjectOverlay();
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true
        });

        // Initial check
        findAndInjectOverlay();
        console.log('🛡️ CyberShield: Extension loaded');
    }

    // =========================================
    // 2. FIND TWITTER'S COMPOSE BOX
    // =========================================
    function findAndInjectOverlay() {
        // Twitter/X uses contentEditable divs for tweet composition
        // The main selectors for the compose area:
        const selectors = [
            '[data-testid="tweetTextarea_0"]',           // Main compose
            '[data-testid="tweetTextarea_1"]',           // Reply compose
            '[role="textbox"][data-testid]',              // Generic textbox
            '.DraftEditor-root',                          // Draft.js editor
            '[contenteditable="true"][role="textbox"]'    // Fallback
        ];

        for (const selector of selectors) {
            const editors = document.querySelectorAll(selector);
            editors.forEach(editor => {
                // Walk up to find the compose container
                const container = findComposeContainer(editor);
                if (container && !overlayInjected.has(container)) {
                    injectOverlay(container, editor);
                    overlayInjected.add(container);
                }
            });
        }
    }

    function findComposeContainer(editor) {
        // Walk up the DOM to find a suitable parent to attach the overlay to
        let el = editor;
        for (let i = 0; i < 15; i++) {
            if (!el.parentElement) break;
            el = el.parentElement;
            // Look for the toolbox row (media buttons, emoji, etc)
            if (el.querySelector('[data-testid="tweetButton"]') ||
                el.querySelector('[data-testid="tweetButtonInline"]') ||
                el.querySelector('[role="group"]')) {
                return el;
            }
        }
        // Fallback: just use a reasonable parent
        return editor.closest('[data-testid]') || editor.parentElement?.parentElement;
    }

    // =========================================
    // 3. INJECT THE OVERLAY
    // =========================================
    function injectOverlay(container, editor) {
        // Create overlay element
        const overlay = document.createElement('div');
        overlay.className = 'cybershield-overlay';
        overlay.innerHTML = `
            <div class="cs-header">
                <span class="cs-brand">
                    <span class="cs-dot safe"></span>
                    CYBERSHIELD
                </span>
                <span>
                    <span class="cs-score safe" id="csScore">0%</span>
                    <span class="cs-zone safe" id="csZone">SAFE</span>
                </span>
            </div>
            <div class="cs-meter-bg">
                <div class="cs-meter-fill safe" id="csMeter"></div>
            </div>
            <div class="cs-warning" id="csWarning">⚠️ This tweet may violate community guidelines</div>
            <span class="cs-toggle" id="csToggle">show details</span>
            <div class="cs-explanations" id="csExplanations"></div>
            <button class="cs-rewrite-btn" id="csRewriteBtn">✨ Rewrite with Empathy</button>
            <div class="cs-rewrite-box" id="csRewriteBox">
                <div class="cs-rewrite-label">✅ Suggested Rewrite</div>
                <div class="cs-rewrite-text" id="csRewriteText"></div>
                <span class="cs-rewrite-score" id="csRewriteScore"></span>
                <button class="cs-use-btn" id="csUseBtn">Use This</button>
            </div>
        `;

        // Insert the overlay into the container
        container.appendChild(overlay);

        // Get references
        const scoreEl = overlay.querySelector('#csScore');
        const zoneEl = overlay.querySelector('#csZone');
        const meterEl = overlay.querySelector('#csMeter');
        const dotEl = overlay.querySelector('.cs-dot');
        const warningEl = overlay.querySelector('#csWarning');
        const toggleEl = overlay.querySelector('#csToggle');
        const expEl = overlay.querySelector('#csExplanations');
        const rewriteBtn = overlay.querySelector('#csRewriteBtn');
        const rewriteBox = overlay.querySelector('#csRewriteBox');
        const rewriteText = overlay.querySelector('#csRewriteText');
        const rewriteScore = overlay.querySelector('#csRewriteScore');
        const useBtn = overlay.querySelector('#csUseBtn');

        let showDetails = false;

        // Toggle details
        toggleEl.addEventListener('click', () => {
            showDetails = !showDetails;
            expEl.classList.toggle('visible', showDetails);
            toggleEl.textContent = showDetails ? 'hide details' : 'show details';
        });

        // Listen for input on the editor
        const handleInput = () => {
            const text = getEditorText(editor);
            if (text === lastAnalyzedText) return;
            lastAnalyzedText = text;

            clearTimeout(debounceTimer);
            if (!text.trim()) {
                resetOverlay(overlay, scoreEl, zoneEl, meterEl, dotEl, warningEl, rewriteBtn, rewriteBox, expEl);
                return;
            }
            debounceTimer = setTimeout(() => {
                analyzeText(text, overlay, scoreEl, zoneEl, meterEl, dotEl, warningEl, rewriteBtn, expEl);
            }, 400);
        };

        editor.addEventListener('input', handleInput);
        // Also observe for programmatic changes
        const inputObserver = new MutationObserver(handleInput);
        inputObserver.observe(editor, { childList: true, subtree: true, characterData: true });

        // Rewrite button
        rewriteBtn.addEventListener('click', () => {
            const text = getEditorText(editor);
            if (!text.trim()) return;

            rewriteBtn.textContent = '⏳ Rewriting...';
            rewriteBtn.disabled = true;

            chrome.runtime.sendMessage({ type: 'REWRITE', text }, (response) => {
                if (response && response.success) {
                    const data = response.data;
                    rewriteText.textContent = data.rewritten;
                    rewriteScore.textContent = `New score: ${data.new_score}%`;
                    rewriteBox.classList.add('visible');

                    useBtn.onclick = () => {
                        setEditorText(editor, data.rewritten);
                        rewriteBox.classList.remove('visible');
                    };
                } else {
                    console.error('CyberShield rewrite failed:', response?.error);
                }
                rewriteBtn.textContent = '✨ Rewrite with Empathy';
                rewriteBtn.disabled = false;
            });
        });

        console.log('🛡️ CyberShield: Overlay injected into compose box');
    }

    // =========================================
    // 4. ANALYZE TEXT VIA API
    // =========================================
    function analyzeText(text, overlay, scoreEl, zoneEl, meterEl, dotEl, warningEl, rewriteBtn, expEl) {
        chrome.runtime.sendMessage({ type: 'ANALYZE', text }, (response) => {
            if (response && response.success) {
                updateOverlay(response.data, overlay, scoreEl, zoneEl, meterEl, dotEl, warningEl, rewriteBtn, expEl);
            } else {
                console.error('CyberShield analysis failed:', response?.error);
            }
        });
    }

    function updateOverlay(data, overlay, scoreEl, zoneEl, meterEl, dotEl, warningEl, rewriteBtn, expEl) {
        const score = data.score;
        const zone = (data.zone || 'SAFE').toLowerCase();

        // Score
        scoreEl.textContent = score + '%';
        scoreEl.className = 'cs-score ' + zone;

        // Zone badge
        zoneEl.textContent = data.zone || 'SAFE';
        zoneEl.className = 'cs-zone ' + zone;

        // Meter
        meterEl.style.width = Math.max(score, 2) + '%';
        meterEl.className = 'cs-meter-fill ' + zone;

        // Dot
        dotEl.className = 'cs-dot ' + zone;

        // Overlay background
        overlay.className = 'cybershield-overlay';
        if (score >= 60) overlay.classList.add('danger');
        else if (score >= 40) overlay.classList.add('warning');

        // Warning
        warningEl.classList.toggle('visible', data.is_bullying);

        // Rewrite button
        rewriteBtn.classList.toggle('visible', score >= 30);

        // Explanations
        if (data.explanation && data.explanation.length > 0) {
            expEl.innerHTML = data.explanation.map(e =>
                `<div class="cs-exp-item">${escapeHtml(e)}</div>`
            ).join('');
        }
    }

    function resetOverlay(overlay, scoreEl, zoneEl, meterEl, dotEl, warningEl, rewriteBtn, rewriteBox, expEl) {
        scoreEl.textContent = '0%';
        scoreEl.className = 'cs-score safe';
        zoneEl.textContent = 'SAFE';
        zoneEl.className = 'cs-zone safe';
        meterEl.style.width = '0%';
        meterEl.className = 'cs-meter-fill safe';
        dotEl.className = 'cs-dot safe';
        overlay.className = 'cybershield-overlay';
        warningEl.classList.remove('visible');
        rewriteBtn.classList.remove('visible');
        rewriteBox.classList.remove('visible');
        expEl.classList.remove('visible');
        expEl.innerHTML = '';
        lastAnalyzedText = '';
    }

    // =========================================
    // 5. TWITTER EDITOR HELPERS
    // =========================================
    function getEditorText(editor) {
        // Twitter uses a Draft.js-like editor with nested spans
        return editor.textContent || editor.innerText || '';
    }

    function setEditorText(editor, text) {
        // For contentEditable, we need to simulate user input
        editor.focus();

        // Select all existing content
        const selection = window.getSelection();
        const range = document.createRange();
        range.selectNodeContents(editor);
        selection.removeAllRanges();
        selection.addRange(range);

        // Insert new text via execCommand to trigger React's onChange
        document.execCommand('insertText', false, text);

        // Dispatch input event
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
