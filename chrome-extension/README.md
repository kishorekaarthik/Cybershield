# CyberShield Chrome Extension — Safe Compose

Real-time toxicity analysis injected directly into Twitter/X's compose box.

## How to Install

1. Open Chrome and go to `chrome://extensions/`
2. Enable **Developer Mode** (toggle in top right)
3. Click **"Load unpacked"**
4. Select the `chrome-extension` folder from this project
5. The CyberShield icon will appear in your Chrome toolbar

## How to Use

1. Make sure your Flask server is running: `python app.py`
2. Go to [twitter.com](https://twitter.com) or [x.com](https://x.com)
3. Click on any compose box (new tweet, reply, etc.)
4. Start typing — a **toxicity meter** will appear below the compose box
5. If toxicity exceeds 30%, a **"✨ Rewrite with Empathy"** button appears
6. Click it to get a civil rewrite suggestion
7. Click **"Use This"** to replace your text with the rewrite

## Requirements

- Flask server running on `localhost:5000`
- `flask-cors` installed (`pip install flask-cors`)
