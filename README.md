# CyberShield: Real-Time Toxicity Detection Platform

CyberShield is a comprehensive content moderation system designed to detect and prevent online harassment in real-time. It integrates a sophisticated cross-modal transformer fusion engine with a browser-based intervention tool, enabling proactive moderation directly within social media interfaces.

## Project Overview

This repository contains the complete source code for the CyberShield platform, which consists of three main components:
1. **Core Detection Engine**: A Flask-based backend API that scores text with a pretrained transformer toxicity classifier (`unitary/toxic-bert` by default), combined with a rule engine for explanations, threat/self-harm patterns and anti-evasion. A BERT + ResNet fusion architecture for MMHS150K training is included in `model_engine.py` but has no trained checkpoint yet and is not used for scoring.
2. **Safe Compose Extension**: A Chrome extension that injects a real-time toxicity analysis overlay into Twitter/X.

## Features

### 1. Safe Compose (Chrome Extension)
The flagship feature of CyberShield. It provides immediate feedback to users as they type on social media platforms.
- **Real-Time Analysis**: As the user types in the compose box, the text is sent to the local inference server for analysis.
- **Visual Toxicity Meter**: A dynamic bar indicates the toxicity level (Safe, Uncertain, Moderate, High).
- **Intervention System**: High-toxicity content triggers a warning and temporarily disables the "Post" button (simulated).
- **Generative Rewrite**: Users can click "Rewrite with Empathy" to have an LLM-based agent suggest a non-toxic version of their message while preserving the original intent.

### 2. Cyberbullying Detection Engine
The backend logic that powers the analysis.
- **Multi-Modal Analysis**: Capable of processing both text and image data (architecture supports future image integration).
- **Granular Classification**: Identifies specific types of toxicity including Racism, Sexism, Physical Threats, and Personal Attacks.
- **Anti-Evasion**: Detects "leetspeak" and adversarial text modifications (e.g., using numbers for letters).

## Installation Guide

### Prerequisites
- Python 3.8 or higher
- Google Chrome or a Chromium-based browser (Brave, Edge, etc.)
- Git

### Part 1: Backend Setup
The backend must be running for the extension and dashboard to function.

1. **Clone the Repository**
   ```bash
   git clone https://github.com/YourUsername/CyberShield.git
   cd Cybershield
   ```

2. **Create a Virtual Environment**
   It is recommended to use a virtual environment to manage dependencies.
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # Mac/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the Server**
   ```bash
   python app.py
   ```
   You should see output indicating the server is running on `http://localhost:5000`.

   On first start the toxicity classifier (about 400 MB) is downloaded from Hugging Face and cached. The startup log prints the active scoring engine, for example `Scoring engine: classifier:unitary/toxic-bert`. If the model cannot be loaded, the server falls back to `rules-only` keyword/pattern scoring and reports that in the `engine` field of every response.

   To use a different classifier (a Hugging Face model id or a local directory), set `CYBERSHIELD_TEXT_MODEL` before starting the server. Set `CYBERSHIELD_MODEL_ACTIVATION=softmax` for single-label models; multi-label models such as toxic-bert use `sigmoid` (the default).

### Part 2: Extension Setup
1. Open Google Chrome and navigate to `chrome://extensions/`.
2. Toggle **Developer mode** in the top right corner.
3. Click the **Load unpacked** button.
4. Select the `chrome-extension` folder located inside the cloned repository.
5. The "CyberShield - Safe Compose" extension should now appear in your list.

## Usage Instructions

### Using Safe Compose (Twitter/X)
1. Ensure the Flask backend is running (`python app.py`).
2. Navigate to [twitter.com](https://twitter.com) or [x.com](https://x.com).
3. Click on the "Post" button to open the compose box.
4. Start typing. You will see the CyberShield overlay appear at the bottom of the compose window.
   - **Green Bar**: Your text is safe.
   - **Yellow/Orange Bar**: Your text contains potential toxicity.
   - **Red Bar**: High toxicity detected.
5. If High Toxicity is detected:
   - Click the **Rewrite with Empathy** button.
   - Wait for the suggestion.
   - Click **Use This** to replace your text with the sanitized version.

### Using the Dashboards
1. Open your browser and go to `http://localhost:5000`.
2. **Live Demo**: Navigate to `/safe-compose` (or click "Safe Compose" in the nav) to test the detection engine without the extension.

## API Reference

The backend exposes the following REST endpoints:

- **POST /analyze-live**
  - **Input**: JSON `{ "text": "user input string" }`
  - **Output**: JSON containing `score` (0-100), `is_bullying`, `zone` (SAFE/UNCERTAIN/MODERATE/HIGH), `categories` (per-label classifier probabilities: toxic, severe_toxic, obscene, threat, insult, identity_hate), `explanation`, `triggers` (words to highlight) and `engine` (which scorer produced the result).
  - **Description**: Real-time analysis endpoint used by the Chrome Extension.

- **POST /rewrite-empathy**
  - **Input**: JSON `{ "text": "toxic string" }`
  - **Output**: JSON `{ "rewritten": "sanitized string", "new_score": 15 }`
  - **Description**: Generates a fast, non-toxic rewrite of the input text.

## How Scoring Works

1. The rule engine (`rule_based_analysis`) finds keyword and emoji triggers and high-precision patterns (direct threats, suicide encouragement). It also produces de-obfuscated variants of the text (leetspeak, repeated characters, spaced-out letters).
2. The classifier scores the original text and each variant; the highest probability per label is kept, so `1d10t` is scored as `idiot`.
3. The severity score is the highest toxic-label probability, raised to at least 0.90 for suicide encouragement and 0.80 for direct threats. Text scoring 0.60 or above is flagged.

## Running Tests

```bash
pip install pytest
python -m pytest tests
```

The tests use a stub classifier, so they do not download a model.

## Project Structure

- `app.py`: Main Flask application entry point.
- `model_engine.py`: Rule engine, `CyberShieldDetector` (inference used by the server) and the experimental `CyberBullyModel` fusion architecture and training loop.
- `tests/`: Unit tests for the detector.
- `chrome-extension/`: Source code for the browser extension.
  - `manifest.json`: Extension configuration (Manifest V3).
  - `content.js`: Script that injects the overlay into Twitter.
  - `background.js`: Service worker for API communication.
- `templates/`: HTML files for the web dashboard (index, features, etc.).
- `static/`: CSS and assets for the web dashboard.
- `requirements.txt`: Python dependency list.

## Troubleshooting

**Extension says "Connection Error"**
- Ensure `app.py` is running.
- Check if `http://localhost:5000/` loads in your browser.
- Verify that `flask-cors` is installed (`pip install flask-cors`).

**Extension not showing up on Twitter**
- Refresh the Twitter page.
- Ensure you are on the "Compose" or "Reply" screen.
- Check `chrome://extensions` to make sure the extension is enabled and has no errors.

## License
This project is licensed under the MIT License.
