/**
 * CyberShield - Background Service Worker
 * Routes API calls from content script to the Flask server.
 * Required because Manifest V3 content scripts can be blocked
 * from making cross-origin requests directly.
 */

const API_BASE = 'http://localhost:5000';

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === 'ANALYZE') {
        fetch(`${API_BASE}/analyze-live`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: message.text })
        })
            .then(r => r.json())
            .then(data => sendResponse({ success: true, data }))
            .catch(err => sendResponse({ success: false, error: err.message }));
        return true; // Keep message channel open for async response
    }

    if (message.type === 'REWRITE') {
        fetch(`${API_BASE}/rewrite-empathy`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: message.text })
        })
            .then(r => r.json())
            .then(data => sendResponse({ success: true, data }))
            .catch(err => sendResponse({ success: false, error: err.message }));
        return true;
    }
});
