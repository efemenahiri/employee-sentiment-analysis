"""Rule-based sentiment labeller (Positive / Negative / Neutral).

Why rule-based?  The task needs reproducible, offline labelling of an *unlabelled*
corpus.  No pretrained model or internet was available when this was built, so a
VADER-style lexicon scorer was implemented from scratch with a hand-curated,
business-email-oriented word list.  The approach:

1. Every word in the lexicon has a valence from -4 (very negative) to +4 (very positive).
2. A negator ("not", "never", "n't" ...) in the 3 words before a term flips and
   dampens its valence (x -0.74).
3. A booster ("very", "extremely") before a term raises its magnitude, a dampener
   ("slightly") lowers it.  ALL-CAPS words and "!" add a little emphasis.
4. Valences are summed and squashed to a compound score in [-1, 1]:
   compound = s / sqrt(s^2 + 15)   (the same normalisation VADER uses).
5. Subject-line words count too (the subject is prepended to the body).
6. Label: compound >= POS_THRESHOLD -> Positive, <= NEG_THRESHOLD -> Negative,
   otherwise Neutral.  The thresholds are deliberately wider than VADER's +-0.05
   because routine business e-mail is full of polite words ("thanks", "please")
   that should not make a scheduling note "Positive".

Limitations (also stated in the README): no sarcasm handling, bag-of-words context
only, and the lexicon is a judgement call -- see the spot-check in the notebook.
"""
import math
import re

POS_THRESHOLD = 0.25
NEG_THRESHOLD = -0.25

POSITIVE = {
    # strong
    "excellent": 3.2, "outstanding": 3.2, "fantastic": 3.2, "wonderful": 3.0, "amazing": 3.0,
    "awesome": 3.0, "terrific": 3.0, "superb": 3.0, "perfect": 2.8, "thrilled": 3.0,
    "love": 2.8, "loved": 2.8, "delighted": 2.8, "congratulations": 3.0, "congrats": 3.0,
    "congratulate": 2.8, "brilliant": 2.8, "great": 2.5, "best": 2.5, "happy": 2.5,
    "excited": 2.4, "exciting": 2.4, "success": 2.4, "successful": 2.4, "successfully": 2.3,
    "pleased": 2.3, "glad": 2.2, "enjoy": 2.2, "enjoyed": 2.2, "impressive": 2.4,
    "impressed": 2.4, "appreciate": 2.2, "appreciated": 2.2, "appreciation": 2.2,
    "grateful": 2.4, "win": 2.2, "won": 2.0, "winner": 2.3, "winning": 2.2,
    "victory": 2.3, "celebrate": 2.3, "celebration": 2.3, "proud": 2.3, "beautiful": 2.4,
    # moderate
    "good": 1.9, "nice": 1.8, "well": 0.5, "better": 1.6, "improve": 1.6, "improved": 1.6,
    "improvement": 1.6, "improving": 1.6, "benefit": 1.0, "benefits": 0.8, "beneficial": 1.7,
    "positive": 1.8, "like": 0.2, "likes": 0.4, "fun": 2.1, "funny": 1.8, "lucky": 1.8,
    "fortunate": 1.8, "opportunity": 1.0, "opportunities": 1.0, "growth": 0.9, "gain": 0.9,
    "gains": 0.9, "profit": 0.9, "profitable": 1.2, "profits": 0.9, "thanks": 0.4,
    "thank": 0.4, "thankful": 1.6, "welcome": 1.0, "welcomed": 1.0, "pleasure": 2.0,
    "pleasant": 1.8, "agree": 0.7, "agreed": 0.6, "approve": 0.7, "approved": 0.7,
    "approval": 0.6, "support": 0.5, "supportive": 1.8, "helpful": 1.9, "help": 0.4,
    "helped": 0.8, "helping": 0.6, "useful": 1.5, "valuable": 1.8, "strong": 1.0,
    "stronger": 1.1, "solid": 1.3, "smooth": 1.4, "easy": 0.9, "easier": 0.9,
    "fine": 0.5, "ok": 0.5, "okay": 0.5, "cool": 1.6, "sweet": 1.5, "hope": 0.6,
    "hopeful": 1.6, "hopefully": 0.5, "confident": 1.6, "confidence": 1.5,
    "committed": 0.8, "dedicated": 1.4, "recommend": 0.7, "recommended": 0.7,
    "reward": 1.5, "rewarded": 1.6, "promoted": 2.0, "promotion": 1.8, "congrat": 2.5,
    "bonus": 1.6, "raise": 1.0, "kudos": 2.5, "cheers": 1.8, "cheer": 1.6,
    "yes": 0.3, "sure": 0.3, "definitely": 0.5, "absolutely": 0.6,
    "interesting": 0.9, "interested": 0.5,
    "accomplish": 1.8, "accomplished": 2.0, "achievement": 2.0,
    "achieve": 1.6, "achieved": 1.8, "progress": 1.3, "resolved": 1.0, "solved": 1.0,
    "solution": 0.4, "fixed": 0.6, "safe": 0.8, "secure": 0.6, "stable": 0.7,
    "lol": 1.8, "haha": 2.0, "hehe": 1.8, "smile": 1.8,
    "wish": 0.5, "wishes": 0.6, "enjoyable": 2.0, "rocks": 1.8,
    "friend": 1.0, "friends": 1.0, "friendly": 1.6, "party": 0.8, "birthday": 1.2,
    "holiday": 0.3, "holidays": 0.3, "vacation": 0.6, "relax": 1.3, "relaxing": 1.4,
}

NEGATIVE = {
    # strong
    "terrible": -3.2, "horrible": -3.2, "awful": -3.2, "disaster": -3.2, "disastrous": -3.2,
    "hate": -3.0, "hated": -3.0, "furious": -3.2, "outraged": -3.2, "outrageous": -3.0,
    "disgusting": -3.2, "unacceptable": -2.8, "worst": -3.0, "pathetic": -3.0,
    "ridiculous": -2.6, "incompetent": -3.0, "useless": -2.8, "worthless": -3.0,
    "fraud": -3.0, "fraudulent": -3.0, "lawsuit": -2.4, "sue": -2.4, "sued": -2.6,
    "litigation": -1.8, "bankruptcy": -2.4, "bankrupt": -2.6, "collapse": -2.6,
    "scandal": -2.8, "investigation": -1.4, "subpoena": -2.0, "violation": -2.2,
    "violate": -2.2, "violated": -2.2, "angry": -2.7, "anger": -2.6, "upset": -2.3,
    "mad": -2.2, "frustrated": -2.4, "frustrating": -2.5, "frustration": -2.4,
    "annoyed": -2.2, "annoying": -2.3, "disappointed": -2.4, "disappointing": -2.4,
    "disappointment": -2.4, "sad": -2.2, "unhappy": -2.4, "miserable": -2.8,
    "devastated": -3.0, "shocked": -1.8, "shocking": -2.0, "stupid": -2.6, "idiot": -3.0,
    "crap": -2.4, "damn": -2.0, "hell": -1.5, "sucks": -2.6, "suck": -2.4,
    # moderate
    "bad": -2.2, "wrong": -2.0, "worse": -2.2, "poor": -1.9, "problem": -1.6,
    "problems": -1.7, "issue": -0.8, "issues": -1.0, "trouble": -1.8, "troubles": -1.8,
    "concern": -1.0, "concerned": -1.4, "concerns": -1.2, "worry": -1.6, "worried": -1.7,
    "worries": -1.6, "fear": -1.9, "afraid": -1.8, "risky": -1.2, "danger": -2.0, "dangerous": -2.2, "threat": -2.0, "threaten": -2.0,
    "fail": -2.0, "failed": -2.1, "failure": -2.3, "failing": -2.0, "fails": -2.0,
    "mistake": -1.8, "mistakes": -1.8, "error": -1.5, "errors": -1.5, "wrongly": -1.7,
    "loss": -1.2, "losses": -1.3, "lose": -1.8, "lost": -1.7, "losing": -1.8,
    "cancel": -1.3, "cancelled": -1.4, "canceled": -1.4, "cancellation": -1.4,
    "delay": -1.4, "delayed": -1.5, "delays": -1.4, "late": -0.9, "behind": -0.8,
    "unfortunately": -1.6, "unfortunate": -1.8, "sorry": -1.2, "apologize": -1.0,
    "apologies": -1.0, "regret": -1.8, "regrettably": -1.8, "complain": -1.9,
    "complaint": -1.9, "complaints": -1.9, "complaining": -1.9, "refuse": -1.7,
    "refused": -1.8, "reject": -1.7, "rejected": -1.9, "denied": -1.6, "deny": -1.5,
    "difficult": -1.3, "difficulty": -1.4, "hard": -0.7, "tough": -1.1, "struggle": -1.6,
    "struggling": -1.6, "stress": -1.6, "stressed": -1.8, "stressful": -1.9,
    "pressure": -0.9, "overwhelmed": -1.9, "exhausted": -1.6, "tired": -1.0,
    "sick": -1.5, "ill": -1.4, "pain": -1.8, "painful": -1.9, "hurt": -1.9,
    "broken": -1.7, "crash": -1.9, "crashed": -1.9, "bug": -1.0,
    "default": -1.3, "defaulted": -2.0, "debt": -1.2, "deficit": -1.5, "decline": -1.2,
    "declined": -1.2, "decrease": -0.9, "drop": -0.8, "dropped": -0.9, "down": -0.2,
    "weak": -1.4, "weaker": -1.5, "weakness": -1.4, "negative": -1.6, "against": -0.5,
    "conflict": -1.6, "dispute": -1.6, "disputes": -1.6, "disagree": -1.4,
    "disagreed": -1.4, "argue": -1.5, "argument": -1.4, "blame": -1.8, "fault": -1.4,
    "fired": -2.4, "fire": -0.9, "layoff": -2.0, "layoffs": -2.0, "terminate": -1.8,
    "terminated": -2.0, "termination": -1.6, "resign": -1.4, "resigned": -1.5,
    "resignation": -1.5, "quit": -1.3, 
    "unable": -1.4, "cannot": -0.8, "can't": -0.8, "won't": -0.7, "impossible": -1.8,
    "unclear": -1.0, "confusing": -1.5, "confused": -1.4, "confusion": -1.4,
    "mess": -1.9, "messy": -1.7, "chaos": -2.2, "chaotic": -2.1, "bother": -1.3,
    "bothered": -1.4, "nasty": -2.3, "rude": -2.2, "lie": -2.0, "lied": -2.2,
    "lies": -2.0, "lying": -2.2, "cheat": -2.2, "cheated": -2.3, "dishonest": -2.4,
    "unfair": -2.1, "illegal": -2.2, "illegally": -2.2, "criminal": -2.4,
    "guilty": -2.0, "penalty": -1.5, "penalties": -1.5, "fined": -1.8,
    "damage": -1.9, "damages": -1.7, "damaged": -1.9, "dislike": -1.7,
    "boring": -1.6, "bored": -1.5, "dull": -1.3, "slow": -1.1, "slowly": -0.6,
    "waste": -1.7, "wasted": -1.8, "wasting": -1.8, "overdue": -1.5, "missed": -1.2,
    "miss": -0.7, "missing": -0.9, "shortage": -1.4, "crisis": -2.3,
    "emergency": -1.4, "urgent": -0.4, "ugh": -1.9, "sigh": -1.1, "unfortunately,": -1.6,
}

# NOTE (domain adjustment, made after reading the messages behind the first flight-risk
# results): in this corpus "risk" is part of department/report names ("Risk Management",
# "RAC risk reporting") and "outage" is a routine IT / power-plant notice, so neither carries
# sentiment and both are left out of the lexicon.  "like" is mostly a preposition (tiny weight).
# Boilerplate (legal footers, "Best regards") is removed in preprocessing.py.

NEGATORS = {"not", "no", "never", "none", "nobody", "nothing", "neither", "nor", "cannot",
            "without", "hardly", "barely", "isn't", "wasn't", "aren't", "weren't", "don't",
            "doesn't", "didn't", "won't", "wouldn't", "shouldn't", "couldn't", "can't",
            "haven't", "hasn't", "hadn't", "ain't"}
BOOSTERS = {"very": 0.3, "really": 0.3, "extremely": 0.4, "absolutely": 0.35, "totally": 0.3,
            "completely": 0.3, "so": 0.2, "highly": 0.3, "incredibly": 0.4, "truly": 0.3,
            "especially": 0.25, "particularly": 0.2, "most": 0.2, "super": 0.3, "quite": 0.15}
DAMPENERS = {"slightly": -0.3, "somewhat": -0.25, "kinda": -0.25,
             "little": -0.2, "barely": -0.3, "marginally": -0.3}

NEGATION_FACTOR = -0.74
_TOKEN = re.compile(r"[A-Za-z][A-Za-z']*")


def _tokenise(text: str):
    return _TOKEN.findall(text)


def score_text(text: str) -> float:
    """Return a compound sentiment score in [-1, 1] for one cleaned message."""
    if not isinstance(text, str) or not text.strip():
        return 0.0
    tokens = _tokenise(text)
    lowered = [t.lower() for t in tokens]
    total = 0.0
    for i, word in enumerate(lowered):
        val = POSITIVE.get(word)
        if val is None:
            val = NEGATIVE.get(word)
        if val is None:
            continue
        # emphasis: ALL-CAPS word (len>2) among mixed-case text
        if tokens[i].isupper() and len(tokens[i]) > 2:
            val += 0.3 if val > 0 else -0.3
        # boosters / dampeners in the two preceding words
        for back in (1, 2):
            if i - back >= 0:
                prev = lowered[i - back]
                if prev in BOOSTERS:
                    val += BOOSTERS[prev] * (1 if val > 0 else -1) * (1.0 if back == 1 else 0.7)
                elif prev in DAMPENERS:
                    val += DAMPENERS[prev] * (1 if val > 0 else -1)
        # negation within three preceding words
        if any(lowered[j] in NEGATORS for j in range(max(0, i - 3), i)):
            val *= NEGATION_FACTOR
        total += val
    if total != 0:                                   # exclamation marks only amplify existing sentiment
        total += min(text.count("!"), 4) * 0.2 * (1 if total > 0 else -1)
    return total / math.sqrt(total * total + 15.0)


def label_from_score(score: float, pos: float = POS_THRESHOLD, neg: float = NEG_THRESHOLD) -> str:
    if score >= pos:
        return "Positive"
    if score <= neg:
        return "Negative"
    return "Neutral"


def label_text(text: str) -> str:
    return label_from_score(score_text(text))


def label_dataframe(df, text_col: str = "text"):
    """Add `sentiment_score` and `sentiment` columns to a DataFrame (returns a copy)."""
    out = df.copy()
    out["sentiment_score"] = out[text_col].map(score_text)
    out["sentiment"] = out["sentiment_score"].map(label_from_score)
    return out


def explain_text(text: str):
    """Debug helper: list the lexicon words that contributed to the score."""
    out = []
    for w in _tokenise(text):
        lw = w.lower()
        v = POSITIVE.get(lw, NEGATIVE.get(lw))
        if v is not None:
            out.append((lw, v))
    return out


# --------------------------------------------------------------------------- TextBlob (optional)
TEXTBLOB_THRESHOLD = 0.05      # common convention: polarity >= +0.05 Positive, <= -0.05 Negative


def textblob_polarity(text: str) -> float:
    """TextBlob polarity in [-1, 1].  TextBlob is an optional dependency (`pip install textblob`);
    it is imported lazily so the rest of the project works without it."""
    from textblob import TextBlob
    if not isinstance(text, str) or not text.strip():
        return 0.0
    return float(TextBlob(text).sentiment.polarity)


def label_dataframe_textblob(df, text_col: str = "text", threshold: float = TEXTBLOB_THRESHOLD):
    """Add `textblob_polarity` and `textblob_label` columns (returns a copy)."""
    out = df.copy()
    out["textblob_polarity"] = out[text_col].map(textblob_polarity)
    out["textblob_label"] = out["textblob_polarity"].map(lambda v: label_from_score(v, threshold, -threshold))
    return out
