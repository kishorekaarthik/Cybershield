"""Detector tests. They use a stub classifier, so no model download is needed."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_engine import CyberShieldDetector, rule_based_analysis  # noqa: E402

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


class StubClassifier:
    """Scores 0.95 'toxic'/'insult' when a known insult appears, else 0.02."""

    def __init__(self):
        self.seen = []

    def predict(self, texts, batch_size=32):
        self.seen.extend(texts)
        out = []
        for t in texts:
            hit = any(w in t.lower().split() for w in ("idiot", "asshole", "moron"))
            p = 0.95 if hit else 0.02
            out.append({"toxic": p, "severe_toxic": 0.01, "obscene": 0.01,
                        "threat": 0.01, "insult": p, "identity_hate": 0.01})
        return out


@pytest.fixture
def detector():
    return CyberShieldDetector(model_name="stub", classifier=StubClassifier())


def test_benign_text_is_safe(detector):
    res = detector.predict_and_explain("Hello class, have a great day!")
    assert not res["is_bullying"]
    assert res["severity_score"] < 0.4
    assert res["engine"] == "classifier:stub"


def test_classifier_score_drives_result(detector):
    res = detector.predict_and_explain("you are an idiot")
    assert res["is_bullying"]
    assert res["severity_score"] == pytest.approx(0.95)
    assert res["categories"]["insult"] == pytest.approx(0.95)
    assert any(e.startswith("Model: insult") for e in res["explanation"])


def test_mild_keyword_alone_is_not_flagged(detector):
    # The old rules gave any keyword hit a score >= 0.74
    res = detector.predict_and_explain("damn, that was a hard exam")
    assert not res["is_bullying"]


@pytest.mark.parametrize("text", ["Just do it!", "Let's end it here and finish it tomorrow"])
def test_common_phrases_not_flagged_as_suicide(detector, text):
    res = detector.predict_and_explain(text)
    assert not any("Suicide" in e for e in res["explanation"])
    assert not res["is_bullying"]


def test_suicide_pattern_sets_floor(detector):
    res = detector.predict_and_explain("nobody would miss you, go die")
    assert res["is_bullying"]
    assert res["severity_score"] >= 0.9


def test_threat_pattern_sets_floor(detector):
    res = detector.predict_and_explain("watch your back")
    assert res["is_bullying"]
    assert res["severity_score"] >= 0.8


def test_obfuscated_text_is_scored_by_classifier(detector):
    res = detector.predict_and_explain("you are an 1d10t")
    assert "you are an idiot" in detector.classifier.seen
    assert res["is_bullying"]


def test_batch_matches_single(detector):
    texts = ["have a nice day", "you are an idiot", "a s s h o l e"]
    batch = detector.predict_batch(texts)
    single = [detector.predict_and_explain(t) for t in texts]
    assert [r["severity_score"] for r in batch] == [r["severity_score"] for r in single]


def test_rules_only_fallback_when_model_missing():
    det = CyberShieldDetector(model_name="/nonexistent/model/path")
    assert det.engine == "rules-only"
    res = det.predict_and_explain("you are an idiot")
    assert res["is_bullying"]
    assert res["categories"] == {}


def test_rule_analysis_returns_variants():
    r = rule_based_analysis("you are an 1d10t")
    assert "you are an idiot" in r["variants"]
