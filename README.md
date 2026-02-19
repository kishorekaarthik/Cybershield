# CyberShield: Advanced Toxicity Detection and Prevention System

CyberShield is a real-time content moderation platform designed to detect and prevent online harassment. It combines a cross-modal transformer fusion engine trained on the MMHS150K dataset with direct integration into social media platforms to identify toxicity before it is posted.

## Core Features

### 1. Kid Shield
A safety monitoring system for youth protection. It analyzes interaction patterns to detect cyberbullying, grooming attempts, and hostile behavior without requiring full surveillance.
- **Safety Digest**: Generates periodic summaries of online risk exposure.
- **Risk Assessment**: Classifies interactions into Safe, Uncertain, Moderate, or High risk tiers.
- **Pattern Recognition**: Detects repeated hostility and targeted harassment.

### 2. Safe Compose (Chrome Extension)
A proactive intervention tool for Twitter/X.
- **Real-Time Analysis**: Scans text as the user types in the compose box.
- **Toxicity Meter**: Visualizes the toxicity level of the current draft.
- **Intervention**: Flags harmful content and suggests rewrites using empathetic language models.
- **Cross-Origin Integration**: Connects securely to the local Flask backend for inference.

## Technical Architecture

The system utilizes a hybrid architecture:
- **Backend**: Flask-based Python server hosting the inference engine.
- **Model**: Cross-modal transformer fusion network (BERT + ResNet variants) fine-tuned on hate speech datasets.
- **Frontend**: Retro-futuristic dashboard interface for system monitoring and configuration.
- **Extension**: Manifest V3 Chrome Extension with background service worker for secure API communication.

## Installation

### Prerequisites
- Python 3.8+
- Google Chrome or Chromium-based browser
- Node.js (optional, for advanced frontend development)

### Backend Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/kishorekaarthik/Cybershield.git
   cd Cybershield
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the server:
   ```bash
   python app.py
   ```
   The server will run on http://localhost:5000.

### Chrome Extension Setup
1. Ensure the Flask server is running.
2. Open Chrome and navigate to `chrome://extensions/`.
3. Enable "Developer mode" in the top right.
4. Click "Load unpacked".
5. Select the `chrome-extension` directory from this repository.
6. Open Twitter/X and start composing a tweet to see the overlay.

## Privacy & Ethics
CyberShield processes data locally or through a controlled inference server. It focuses on user-side intervention rather than platform-level censorship, empowering users to make better communication choices.

## License
MIT License
