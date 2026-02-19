# CyberShield: Real-Time Toxicity Detection & Prevention

CyberShield is an advanced content moderation system that proactively detects and prevents online harassment before it happens. At its core is a **Chrome Extension** that integrates directly into social media platforms (Twitter/X) to analyze text in real-time, coupled with a powerful **AI Detection Engine**.

## 🚀 Core Features

### 1. Safe Compose (Chrome Extension)
The primary interface for CyberShield. It injects a real-time toxicity analysis overlay directly into the Twitter/X compose box.
- **Live Toxicity Meter**: As you type, the extension scores your text for toxicity, aggression, and hate speech.
- **Instant Feedback**: Visual indicators (Green/Yellow/Red) let you know if your message is safe or harmful.
- **Smart Rewrites**: Uses LLMs to suggest empathetic, non-toxic alternatives for your draft.
- **Privacy-First**: Analysis happens securely via your local inference server; no data is stored by the extension.

### 2. Cyberbullying Detection Engine
The brain behind the system. A sophisticated AI model designed to understand context and nuance.
- **Cross-Modal Fusion**: Combines BERT-based text analysis with metadata processing for higher accuracy.
- **MMHS150K Trained**: Fine-tuned on massive hate speech datasets to recognize subtle forms of abuse.
- **Leetspeak Detection**: Identifies evasion attempts (e.g., "h4te", "@$$hole") that standard filters miss.
- **Risk Classification**: Categorizes content into granular risk tiers (Personal Attack, Racism, Sexism, etc.).

### 3. Kid Shield (Additional Feature)
A supplemental safety tool for monitoring youth activity.
- **Safety Digest**: Provides periodic summaries of online interactions.
- **Risk Assessment**: Flags potential grooming or bullying patterns without invasive surveillance.

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Google Chrome or Chromium-based browser

### Backend Setup (The Detection Engine)
1. **Clone the repository**:
   ```bash
   git clone https://github.com/kishorekaarthik/Cybershield.git
   cd Cybershield
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Start the API server**:
   ```bash
   python app.py
   ```
   *The server runs on `http://localhost:5000` to handle analysis requests.*

### Chrome Extension Setup
1. Open Chrome and go to `chrome://extensions/`.
2. Enable **Developer mode** (top right toggle).
3. Click **Load unpacked**.
4. Select the `chrome-extension` folder from this project.
5. Go to Twitter/X and start composing!

## 🏗️ Architecture
- **Frontend**: Chrome Extension (JS/HTML/CSS) for direct user interaction.
- **Backend**: Flask API serving the PyTorch/Transformer model.
- **Model**: Custom Late-Fusion Architecture (BERT + ResNet + Metadata).

## License
MIT License
