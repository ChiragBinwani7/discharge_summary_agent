# src/reviewer.py
# ---------------------------------------------------------------------------
# PART 2 — THE SIMULATED REVIEWER ("the doctor").
#
# In production a clinician edits the agent's draft before signing it. We have no
# real clinicians, so this module stands in for one: it takes a draft section and
# returns an edited version, applying a FIXED, HIDDEN editing policy.
#
#   - FIXED   : the policy is implemented as DETERMINISTIC CODE (not an LLM call).
#               This is the strongest possible "consistent policy": the same draft
#               always yields the same edit, every run. So any measured improvement is
#               attributable to the agent LEARNING the policy — never to reviewer noise
#               or model variance. (An LLM reviewer, even at temperature 0, gave
#               run-to-run variance that made the curve unstable; code removes it.)
#   - HIDDEN  : the agent never imports this module and never sees these rules. It only
#               ever sees that its drafts came back edited. The learning loop must
#               INFER the policy from (draft, edited) pairs (memory.py uses an LLM for
#               that inference — that part is still a learning problem).
#
# The policy makes FORMATTING edits only. It is fact-preserving by construction: every
# transform reorders/relabels/normalises text without inventing or dropping a clinical
# value, and any MISSING/CONFLICT/flag marker is returned COMPLETELY UNCHANGED. That
# fence is what lets the agent safely learn the policy without ever learning to
# fabricate (see reward.py for the paired numeric guard).
# ---------------------------------------------------------------------------

import re

# --- The hidden house style, as documentation. The agent must never see this. ---
# The functions below implement exactly these rules, deterministically.
FROZEN_POLICY = """HOUSE STYLE (applied identically every time):
1. Dates: rewrite every date as ISO YYYY-MM-DD.
2. Medications: one per line as  NAME | STRENGTH | FREQUENCY | DURATION  (use — for a
   field the text does not give).
3. Hospital course: compress to at most 4 sentences.
4. Follow-up: rewrite as a numbered list and append "Return immediately if symptoms
   worsen." as the final item.
5. Everything else: tidy whitespace.
NEVER add/remove/change a clinical fact. Return any MISSING/CONFLICT/flag marker
COMPLETELY UNCHANGED.
"""

# Sections whose content is a safety flag / gap marker, not free text. The reviewer
# returns these verbatim — a reviewer never tidies or fills a flagged gap.
_MARKERS = ("⚠", "MISSING", "CONFLICT", "NOT FOUND", "CANNOT RECONCILE",
            "NO CHECK POSSIBLE", "INTERACTIONS FOUND", "ERROR")


def _is_marker(text: str) -> bool:
    up = text.upper()
    return any(m.upper() in up for m in _MARKERS)


# --- Rule 1: dates -> ISO YYYY-MM-DD ----------------------------------------
_DATE = re.compile(r"\b(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2,4})\b")


def _iso_dates(text: str) -> str:
    def repl(m):
        d, mo, y = m.group(1), m.group(2), m.group(3)
        if len(y) == 2:
            y = "20" + y
        return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"
    return _DATE.sub(repl, text)


# --- Rule 2: medications -> NAME | STRENGTH | FREQUENCY | DURATION -----------
# The agent writes med lines a few ways, e.g.
#   "Amlodipine | Dose: 5 mg | Route: PO | Frequency: Once daily"
#   "- Co-amoxiclav (625 mg, PO, three times daily for 5 days)"
# The reviewer's house style drops the field LABELS and the route, keeping
# NAME | STRENGTH | FREQUENCY | DURATION. This is a pure reformat of what's there.
_DOSE   = re.compile(r"(\d+(?:\.\d+)?\s?(?:mg|mcg|g|ml|units?))", re.I)
_FREQ   = re.compile(r"(once daily|twice daily|three times daily|at night|"
                     r"\d-\d-\d|bd|od|tds|qds|sos|[1-4] times daily)", re.I)
_DURed  = re.compile(r"(?:for\s+)?(\d+\s?(?:day|days|week|weeks))", re.I)


def _med_line(line: str) -> str:
    raw = line.strip().lstrip("-*0123456789. ").strip()
    if not raw:
        return line
    # name = text up to the first "|", "(", or dose number
    name = re.split(r"[|(]|\bDose\b|\d", raw, 1)[0].strip(" :-")
    strength = (_DOSE.search(raw).group(1) if _DOSE.search(raw) else "—")
    freq = (_FREQ.search(raw).group(1) if _FREQ.search(raw) else "—")
    dur = (_DURed.search(raw).group(1) if _DURed.search(raw) else "—")
    return f"{name} | {strength} | {freq} | {dur}"


def _format_meds(text: str) -> str:
    lines = [l for l in text.splitlines() if l.strip()]
    return "\n".join(_med_line(l) for l in lines)


# --- Rule 3: hospital course -> at most 4 sentences -------------------------
def _first_sentences(text: str, n: int = 4) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(p.strip() for p in parts[:n] if p.strip())


# --- Rule 4: follow-up -> numbered list + safety-net line -------------------
_SAFETY_NET = "Return immediately if symptoms worsen."


def _numbered_followup(text: str) -> str:
    items = [re.sub(r"^[-*0-9.)\s]+", "", l).strip()
             for l in text.splitlines() if l.strip()]
    items = [i for i in items if i]
    if _SAFETY_NET.lower() not in " ".join(items).lower():
        items.append(_SAFETY_NET)
    return "\n".join(f"{i}. {t}" for i, t in enumerate(items, 1))


def edit_section(section_name: str, draft_text: str) -> str:
    """Apply the hidden house style to ONE section, deterministically.

    Marker/gap sections are returned unchanged. Every transform reformats existing
    text; none invents or drops a clinical fact.
    """
    if not draft_text or not draft_text.strip() or _is_marker(draft_text):
        return draft_text

    name = section_name.lower()
    text = draft_text

    if "medication" in name and "reconciliation" not in name:
        text = _format_meds(text)            # discharge / admission medications
    if "hospital course" in name:
        text = _first_sentences(text, 4)
    if "follow-up" in name or "follow up" in name:
        text = _numbered_followup(text)

    text = _iso_dates(text)                  # dates everywhere
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text or draft_text


def review_draft(draft: dict) -> dict:
    """Apply the hidden policy to a whole draft. Deterministic and instant (no LLM).

    Input/Output: {section_name: text}. Pair (draft, edited) -> reward.py / memory.py.
    """
    return {name: edit_section(name, text) for name, text in draft.items()}
