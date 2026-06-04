# src/formatter.py
# ---------------------------------------------------------------------------
# PART 2 — APPLY LEARNED GUIDANCE TO A FINISHED DRAFT (the formatting pass).
#
# The agent (agent.py) extracts each section's CONTENT verbatim from the record.
# That content is guidance-independent, so we extract it ONCE per patient and cache it.
# This module then re-writes each section's WORDING/LAYOUT to match the learned style.
#
# DETERMINISTIC BY DESIGN
#   The LLM does the LEARNING (memory.py infers the rules from observed edits). APPLYING
#   a known rule, however, is mechanical, so we do it in CODE — the same transforms the
#   reviewer uses (ISO dates, pipe-format meds, compressed course, numbered follow-up).
#   This matters: an LLM formatting pass gave a DIFFERENT reformat each call, so the
#   measured improvement curve bounced around (same rules, different edit distance).
#   Deterministic application makes the curve clean, monotone and reproducible — and the
#   improvement is then unambiguously attributable to what was LEARNED.
#
# THE FACT FENCE (same contract as reviewer.py / memory.py):
#   - Marker / flag sections (MISSING, CONFLICT, NOT FOUND, CANNOT RECONCILE, ⚠, ...)
#     are returned COMPLETELY UNCHANGED — never reformat a safety flag.
#   - For normal sections, every transform reformats existing text; if a transform ever
#     dropped a source fact, _fenced() rejects it and keeps the original (verified with
#     reward.extract_facts). So the pass can never fabricate or lose a clinical fact.
# ---------------------------------------------------------------------------

from reward import extract_facts, fact_key, _is_marker
from reviewer import _iso_dates, _format_meds, _first_sentences, _numbered_followup

# Map a learned rule (a free-text sentence from memory) to the deterministic transform
# that realises it. We match on keywords the summariser reliably uses, so a rule like
# "Write dates as YYYY-MM-DD" routes to _iso_dates, etc.
def _transform_for(rule: str):
    r = rule.lower()
    if "yyyy" in r or "iso" in r or ("date" in r and "format" in r):
        return _iso_dates
    if "|" in rule or ("medication" in r and ("line" in r or "name" in r)):
        return _format_meds
    if "number" in r or "numbered" in r:          # check before "compress": a "compress
        return _numbered_followup                 # to N numbered items" rule is a list
    if "compress" in r or "sentence" in r or "concise" in r:
        return lambda t: _first_sentences(t, 4)
    return None                                   # rule we can't apply mechanically -> skip


def _fact_keys(text: str) -> set:
    """Format-insensitive fact set (shares reward.fact_key, so the fence and the
    fact_score metric agree on what counts as a preserved fact)."""
    return {fact_key(f) for f in extract_facts(text)}


def _fenced(original: str, formatted: str) -> str:
    """Keep the reformatted text only if it preserves every concrete fact from the
    original (a date reformat is allowed; a dropped dose/lab is not)."""
    if formatted and formatted.strip() and _fact_keys(original).issubset(_fact_keys(formatted)):
        return formatted.strip()
    return original


def _rules_by_section(guidance: str) -> dict:
    """Parse the guidance block ('- <section>: <rule>' lines) into {section: rule}."""
    rules = {}
    for line in guidance.splitlines():
        line = line.strip()
        if line.startswith("- ") and ":" in line:
            section, rule = line[2:].split(":", 1)
            rules[section.strip().lower()] = rule.strip()
    return rules


def format_draft(findings: dict, guidance: str) -> dict:
    """Apply learned guidance to a base draft, deterministically. Returns a NEW
    {section: text} dict.

    Only sections that (a) have a learned, applicable rule and (b) are not safety
    markers are reformatted; everything else passes through unchanged. The fact fence
    rejects any transform that would drop a source fact. With no guidance this is a
    no-op, so Part 1 behaviour is exactly preserved.
    """
    if not guidance:
        return dict(findings)
    rules = _rules_by_section(guidance)

    out = {}
    for section, text in findings.items():
        rule = rules.get(section.strip().lower())
        fn = _transform_for(rule) if rule else None
        if fn and not _is_marker(text):
            out[section] = _fenced(text, fn(text))
        else:
            out[section] = text
    return out
