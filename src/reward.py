# src/reward.py
# ---------------------------------------------------------------------------
# PART 2 — THE REWARD / ACCURACY SIGNAL.
#
# This is the RULER. It must be stable and free, so it is pure Python (no LLM):
# an LLM ruler couldn't reliably detect its own improvement, and it would cost a
# call per measurement. Two numbers are produced per section:
#
#   1. edit_reward  = 1 - normalised_edit_distance(draft, doctor's edit)
#                     Less editing by the doctor => higher reward. This is the
#                     "needs fewer corrections over time" signal from the task.
#
#   2. fact_score   = fraction of source FACTS still present in the draft.
#                     This is the ANTI-GAMING guard. Edit distance alone can be
#                     lowered by becoming vaguer or shorter; fact_score punishes
#                     exactly that. A draft that drops a drug, dose, date or
#                     diagnosis loses fact_score no matter how few edits it took.
#
# The combined reward gates on safety: dropping a fact caps the reward hard, so the
# learning loop can never improve its edit score by getting the medicine wrong.
# ---------------------------------------------------------------------------

import re
from difflib import SequenceMatcher

# A dropped fact costs more than any plausible formatting saving, so the optimiser
# can never trade a fact away for a lower edit distance.
FACT_PENALTY = 1.0


def _norm(text: str) -> str:
    """Lowercase and collapse whitespace so formatting noise doesn't dominate."""
    return re.sub(r"\s+", " ", text.strip().lower())


def edit_distance_ratio(a: str, b: str) -> float:
    """Normalised edit distance in [0,1]: 0 = identical, 1 = completely different.

    SequenceMatcher.ratio() gives similarity; 1 - ratio is the normalised distance.
    Operates on normalised text so pure-whitespace changes aren't counted as edits.
    """
    a, b = _norm(a), _norm(b)
    if not a and not b:
        return 0.0
    return 1.0 - SequenceMatcher(None, a, b).ratio()


# --- Fact extraction: the concrete clinical tokens that must survive editing. ---

_NUM_RE  = re.compile(r"\b\d+(?:\.\d+)?\b")                 # doses, labs, ages
_DATE_RE = re.compile(r"\b\d{1,4}[/.\-]\d{1,2}[/.\-]\d{2,4}\b")
# ALL-CAPS words of length >=3 are almost always drug names / abbreviations in
# these records (TAB AUGMENTIN, OFLOX, DKA, UTI). Cheap, high-recall fact tokens.
_CAPS_RE = re.compile(r"\b[A-Z][A-Z]{2,}\b")


def extract_facts(text: str) -> set:
    """Pull the concrete facts out of a piece of text: numbers, dates, drug/Dx tokens.

    Deterministic and explainable. Not a full medical NER — it is a high-recall
    proxy whose only job is: 'did the draft keep the hard facts that were in the
    source?'. Markers like MISSING/CONFLICT are stop-words, not facts.
    """
    if not text or _is_marker(text):
        return set()
    facts = set()
    facts |= set(_DATE_RE.findall(text))
    facts |= set(_NUM_RE.findall(text))
    facts |= {w for w in _CAPS_RE.findall(text) if w not in _STOP}
    return facts


_STOP = {"NOT", "FOUND", "MISSING", "CONFLICT", "DRAFT", "FOR", "ONLY",
         "AND", "THE", "PAGE", "REVIEW", "CLINICIAN", "REQUIRED"}
_MARKERS = ("⚠", "MISSING", "CONFLICT", "NOT FOUND", "CANNOT RECONCILE",
            "NO CHECK POSSIBLE")


def _is_marker(text: str) -> bool:
    up = text.upper()
    return any(m.upper() in up for m in _MARKERS)


def fact_key(fact: str):
    """Canonical key for a fact, making it FORMAT-insensitive for dates: a whole-date
    token (12/05/2026) maps to its sorted digit groups, so a date REFORMAT
    (-> 2026-05-12) is not counted as a lost fact — while a dropped number still is.
    Used by fact_score here and the formatter's fact fence, so both agree."""
    if _DATE_RE.fullmatch(fact):
        return tuple(sorted(re.split(r"[/.\-]", fact)))
    return fact


def fact_score(source_facts: set, draft_text: str) -> float:
    """Fraction of source facts that still appear in the draft. 1.0 if none to keep.

    Compared via fact_key, so reformatting a date does not register as a lost fact
    (the digits are unchanged); dropping a dose/lab/drug still does."""
    if not source_facts:
        return 1.0
    draft_keys = {fact_key(f) for f in extract_facts(draft_text)}
    kept = sum(1 for f in source_facts if fact_key(f) in draft_keys)
    return kept / len(source_facts)


def section_reward(draft_text: str, edited_text: str, source_facts: set) -> dict:
    """Score one section. Returns the edit reward, the fact score, and a combined,
    safety-gated reward in [0,1].

    combined = edit_reward, minus a penalty proportional to how many facts were lost.
    A draft that keeps every fact is scored purely on how little the doctor edited it;
    a draft that drops facts is dragged down regardless of edit distance.
    """
    dist = edit_distance_ratio(draft_text, edited_text)
    edit_reward = 1.0 - dist
    fact = fact_score(source_facts, draft_text)
    combined = max(0.0, edit_reward - FACT_PENALTY * (1.0 - fact))
    return {
        "edit_distance": round(dist, 4),
        "edit_reward": round(edit_reward, 4),
        "fact_score": round(fact, 4),
        "reward": round(combined, 4),
    }


def draft_reward(draft: dict, edited: dict, source_facts: dict) -> dict:
    """Score a whole draft section-by-section and aggregate.

    `source_facts` maps section_name -> set of facts that section must preserve.
    Returns per-section scores plus the means used for the improvement curve.
    """
    per_section = {}
    for name, draft_text in draft.items():
        per_section[name] = section_reward(
            draft_text, edited.get(name, draft_text), source_facts.get(name, set())
        )
    n = max(1, len(per_section))
    return {
        "per_section": per_section,
        "mean_edit_distance": round(sum(s["edit_distance"] for s in per_section.values()) / n, 4),
        "mean_reward": round(sum(s["reward"] for s in per_section.values()) / n, 4),
        "mean_fact_score": round(sum(s["fact_score"] for s in per_section.values()) / n, 4),
    }
