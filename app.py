from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from collections import Counter
import random
import re
import torch

# -------------------------------------------------
# IMPORT YOUR MODEL CODE (UNCHANGED)
# -------------------------------------------------
from model_engine import CyberBullyModel, fetch_user_tweets

# -------------------------------------------------
# FLASK APP
# -------------------------------------------------
app = Flask(__name__)
CORS(app)  # Enable CORS for Chrome extension

# -------------------------------------------------
# LOAD MODEL ONCE (CRITICAL)
# -------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Loading CyberShield model on {DEVICE}")

model = CyberBullyModel(device=DEVICE)
model.eval()

# -------------------------------------------------
# DECISION ZONES (UNCERTAINTY-AWARE)
# -------------------------------------------------
def risk_zone(score: float) -> str:
    """
    score: severity score in range [0,1]
    """
    if score < 0.4:
        return "SAFE"
    elif score < 0.6:
        return "UNCERTAIN"
    elif score < 0.8:
        return "MODERATE"
    else:
        return "HIGH"

# -------------------------------------------------
# POLICY LAYER (NON-ENFORCING)
# -------------------------------------------------
POLICY_MAP = {
    "SAFE": {
        "recommendation": "No Action",
        "reach_reduction": "0%",
        "account_status": "Normal"
    },
    "UNCERTAIN": {
        "recommendation": "Monitor",
        "reach_reduction": "0%",
        "account_status": "Observed"
    },
    "MODERATE": {
        "recommendation": "Visibility Filtering",
        "reach_reduction": "50–70%",
        "account_status": "Limited"
    },
    "HIGH": {
        "recommendation": "Warning",
        "reach_reduction": "80–90%",
        "account_status": "Restricted"
    }
}

# -------------------------------------------------
# ROUTES — CONTENT PAGES
# -------------------------------------------------
@app.route("/")
def landing():
    return render_template("index.html")

@app.route("/features")
def features():
    return render_template("features.html")

@app.route("/architecture")
def architecture():
    return render_template("architecture.html")

@app.route("/ethics")
def ethics():
    return render_template("ethics.html")

@app.route("/try")
def try_page():
    return render_template("try.html")


@app.route("/safe-compose")
def safe_compose():
    return render_template("safe_compose.html")

# -------------------------------------------------
# ANALYSIS ENDPOINT
# -------------------------------------------------
@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    username = data.get("username")

    if not username:
        return jsonify({"error": "Username required"}), 400

    # Fetch tweets (real or dummy)
    tweets = fetch_user_tweets(username, max_results=100)

    results = []
    scores = []

    for t in tweets:
        res = model.predict_and_explain(t["text"])
        scores.append(res["severity_score"])

        results.append({
            "text": t["text"],
            "is_bullying": res["is_bullying"],
            "score": round(res["severity_score"] * 100, 1),
            "explanation": res["explanation"]
        })

    # Account-level aggregation
    avg_score = sum(scores) / len(scores) if scores else 0.0
    avg_score_pct = int(avg_score * 100)

    # Decision zone + policy
    zone = risk_zone(avg_score)
    policy = POLICY_MAP[zone]

    return jsonify({
        "username": username,
        "risk_score": avg_score_pct,
        "risk_zone": zone,
        "policy": policy,
        "tweets": results
    })

# -------------------------------------------------
# BULLY RING DETECTION ENDPOINT
# -------------------------------------------------
@app.route("/analyze-ring", methods=["POST"])
def analyze_ring():
    data = request.get_json()
    target = data.get("username")

    if not target:
        return jsonify({"error": "Username required"}), 400

    # Fetch target's tweets to simulate accounts interacting with them
    tweets = fetch_user_tweets(target, max_results=100)

    if not tweets:
        return jsonify({"error": "Could not fetch tweets for this user"}), 404

    # Simulate multiple source accounts targeting this user
    fake_accounts = ["user_" + str(i) for i in range(1, min(len(tweets), 6) + 1)]
    random.shuffle(fake_accounts)

    account_results = {}
    all_triggers = []
    all_emojis = []

    for i, t in enumerate(tweets):
        res = model.predict_and_explain(t["text"])
        source = fake_accounts[i % len(fake_accounts)]

        # Extract trigger words from explanation
        triggers = []
        emojis = []
        for exp in res["explanation"]:
            if "Keyword Trigger" in exp:
                word = exp.split("'")[1] if "'" in exp else ""
                if word:
                    triggers.append(word)
                    all_triggers.append(word)
            if "Emoji Trigger" in exp:
                emoji = exp.split("'")[1] if "'" in exp else ""
                if emoji:
                    emojis.append(emoji)
                    all_emojis.append(emoji)

        if source not in account_results:
            account_results[source] = {
                "tweets": [],
                "scores": [],
                "triggers": [],
                "emojis": []
            }

        account_results[source]["tweets"].append({
            "text": t["text"],
            "score": round(res["severity_score"] * 100, 1),
            "is_bullying": res["is_bullying"],
            "explanation": res["explanation"]
        })
        account_results[source]["scores"].append(res["severity_score"])
        account_results[source]["triggers"].extend(triggers)
        account_results[source]["emojis"].extend(emojis)

    # Build per-account summaries
    accounts = []
    hostile_count = 0
    for acct, info in account_results.items():
        avg = sum(info["scores"]) / len(info["scores"]) if info["scores"] else 0
        is_hostile = avg >= 0.4
        if is_hostile:
            hostile_count += 1
        accounts.append({
            "username": acct,
            "avg_score": round(avg * 100, 1),
            "zone": risk_zone(avg),
            "tweet_count": len(info["tweets"]),
            "triggers": list(set(info["triggers"])),
            "emojis": list(set(info["emojis"])),
            "is_hostile": is_hostile,
            "tweets": info["tweets"]
        })

    # Coordination detection
    trigger_counts = Counter(all_triggers)
    shared_triggers = [w for w, c in trigger_counts.items() if c >= 2]
    emoji_counts = Counter(all_emojis)
    shared_emojis = [e for e, c in emoji_counts.items() if c >= 2]

    coordination_score = min(100, int(
        (len(shared_triggers) * 15) +
        (len(shared_emojis) * 10) +
        (hostile_count * 12)
    ))

    if coordination_score >= 60:
        verdict = "HIGH COORDINATION"
    elif coordination_score >= 30:
        verdict = "MODERATE COORDINATION"
    else:
        verdict = "LOW COORDINATION"

    return jsonify({
        "target": target,
        "accounts": sorted(accounts, key=lambda x: x["avg_score"], reverse=True),
        "coordination_score": coordination_score,
        "verdict": verdict,
        "shared_triggers": shared_triggers,
        "shared_emojis": shared_emojis,
        "hostile_accounts": hostile_count,
        "total_accounts": len(accounts)
    })

# -------------------------------------------------
# SAFE COMPOSE — LIVE ANALYSIS ENDPOINT
# -------------------------------------------------
@app.route("/analyze-live", methods=["POST"])
def analyze_live():
    data = request.get_json()
    text = data.get("text", "")

    if not text.strip():
        return jsonify({"score": 0, "is_bullying": False, "explanation": [], "triggers": []})

    res = model.predict_and_explain(text)

    # Extract trigger words for highlighting
    triggers = []
    for exp in res["explanation"]:
        if "Keyword Trigger" in exp and "'" in exp:
            word = exp.split("'")[1]
            if word:
                triggers.append(word.lower())
        if "Emoji Trigger" in exp and "'" in exp:
            emoji = exp.split("'")[1]
            if emoji:
                triggers.append(emoji)

    return jsonify({
        "score": round(res["severity_score"] * 100, 1),
        "is_bullying": res["is_bullying"],
        "explanation": res["explanation"],
        "triggers": list(set(triggers)),
        "zone": risk_zone(res["severity_score"])
    })

# -------------------------------------------------
# SAFE COMPOSE — EMPATHY REWRITE ENDPOINT
# -------------------------------------------------
REWRITE_MAP = {
    # Profanity → civil alternatives
    "fuck": "dislike",
    "fucked": "messed up",
    "fucker": "person",
    "fuckers": "people",
    "fucking": "very",
    "fuck you": "I strongly disagree with you",
    "fuck off": "please go away",
    "shit": "stuff",
    "shitty": "poor",
    "bullshit": "nonsense",
    "dick": "jerk",
    "dickhead": "unkind person",
    "cock": "person",
    "pussy": "coward",
    "idiot": "person I disagree with",
    "idiots": "people I disagree with",
    "stupid": "misguided",
    "dumb": "uninformed",
    "moron": "person",
    "morons": "people",
    "loser": "person",
    "losers": "people",
    "trash": "this content",
    "garbage": "this approach",
    "pathetic": "disappointing",
    "disgusting": "concerning",
    "ugly": "unappealing",
    "fat": "different",
    "retard": "person",
    "retarded": "misguided",
    "creep": "person",
    "freak": "individual",
    "weirdo": "unique person",
    "clown": "person",
    "fool": "person",
    "fools": "people",
    "jerk": "unkind person",
    "scum": "person",
    "worthless": "undervalued",
    "useless": "unhelpful",
    "horrible": "concerning",
    "terrible": "unfortunate",
    "awful": "disappointing",
    "suck": "fall short",
    "sucks": "falls short",
    "hate": "strongly dislike",
    "hating": "strongly disagreeing with",
    "shut up": "please reconsider",
    "die": "stop",
    "kill yourself": "please take a break",
    "kys": "please step back",
    "go to hell": "I strongly disagree",
    "ass": "person",
    "asshole": "unkind person",
    "bastard": "person",
    "bitch": "person",
    "damn": "darn",
    "crap": "stuff",
    "hell": "heck",
    "piss off": "please leave",
    "screw you": "I disagree with you",
    "f**k": "dislike",
    "stfu": "please reconsider",
    "wtf": "I'm confused",
    "lmao": "",
    "🤡": "",
    "🖕": "",
    "💀": "",
    "🤬": "",
    "😡": "",
    "👿": "",
}

@app.route("/rewrite-empathy", methods=["POST"])
def rewrite_empathy():
    data = request.get_json()
    text = data.get("text", "")

    if not text.strip():
        return jsonify({"original": text, "rewritten": text, "changes": 0})

    rewritten = text
    changes = 0

    # Sort by length (longest first) to avoid partial replacements
    sorted_map = sorted(REWRITE_MAP.items(), key=lambda x: len(x[0]), reverse=True)

    for bad, good in sorted_map:
        pattern = re.compile(re.escape(bad), re.IGNORECASE)
        if pattern.search(rewritten):
            rewritten = pattern.sub(good, rewritten)
            changes += 1

    # Clean up extra spaces
    rewritten = re.sub(r'\s+', ' ', rewritten).strip()

    # Capitalize first letter
    if rewritten:
        rewritten = rewritten[0].upper() + rewritten[1:]

    # Run analysis on rewritten text
    res_new = model.predict_and_explain(rewritten)

    return jsonify({
        "original": text,
        "rewritten": rewritten,
        "changes": changes,
        "new_score": round(res_new["severity_score"] * 100, 1),
        "new_is_bullying": res_new["is_bullying"]
    })

# -------------------------------------------------
# RUN SERVER
# -------------------------------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
