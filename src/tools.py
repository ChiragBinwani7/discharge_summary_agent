# src/tools.py
# ---------------------------------------------------------------------------
# The things the agent can DO. Each tool is a plain function that returns a short
# text "observation" the agent reads before deciding its next move.
#
# Two rules every tool follows:
#   1. It never invents a fact. The prompts forbid guessing.
#   2. It never crashes. Any failure is caught and returned as an "ERROR: ..."
#      string, so the agent can react to it instead of the program dying.
# ---------------------------------------------------------------------------

import json

from llm import ask, ask_json, extract_json

def _short(err) -> str:
    """First line of an error, capped — keeps the trace readable when a tool fails."""
    return str(err).splitlines()[0][:120]


# The same extraction RULES used by extract_field, shared so single- and batch-mode
# extraction obey one identical contract (verbatim only, NOT FOUND, CONFLICT).
_EXTRACT_RULES = """RULES FOR EVERY FIELD:
- Copy only information explicitly written in the record. Never guess or assume.
- If a field genuinely is not written anywhere, set it to exactly: NOT FOUND
- Words like "Not known" or "Nil" written in the record are real documented answers,
  not NOT FOUND.

FIELD LABELS MAY VARY:
- principal diagnosis: may be "final diagnosis", "primary/provisional diagnosis", or
  just "diagnosis". The SAME value under different labels (provisional vs final, both
  "pneumonia") is NOT a conflict — report it once.
- hospital course: may be "course in hospital" / "clinical course" / spread across notes.
- discharge condition: may be "condition/status at discharge".
- admission medications: admission drug orders, medication charts, ER/ICU records.

CONFLICT vs LIST:
- A CONFLICT means the record states TWO OR MORE DIFFERENT values for the SAME single
  field. Only these single-answer fields can conflict: principal diagnosis, patient
  demographics, discharge condition. Set the value to a string that STARTS WITH the
  word CONFLICT and then lists each differing value with its page. Never pick one,
  never reconcile them, never prefer the "final" one — surface ALL of them.
  Example: if one note calls the principal diagnosis "community-acquired pneumonia"
  and another note calls it "aspiration pneumonia", that IS a conflict; output:
  "CONFLICT: community-acquired pneumonia (p.X) vs aspiration pneumonia (p.Y)".
- "admission and discharge dates" holds BOTH dates together — that is NOT a conflict.
  Write it as: "Admission: <date>; Discharge: <date>" (mark either NOT FOUND if absent).
- List-type fields (secondary diagnoses, procedures, follow-up instructions, pending
  results, medications) are NOT conflicts: COMBINE the documented items from every page
  into ONE newline-separated plain-text list (no JSON arrays, no Python list syntax)."""


def _too_big_or_throttled(err) -> bool:
    t = str(err)
    return "413" in t or "too large" in t.lower() or "429" in t or "rate_limit" in t.lower()


def extract_all_fields(document: str, fields: list) -> dict:
    """Extract EVERY required field in ONE call. Returns {field: value/NOT FOUND/CONFLICT}.

    Instead of one LLM call per field, the model fills a single JSON object. The call
    goes to Gemini, whose large request limit ingests a full scanned record (Groq's
    free-tier per-request cap rejects records this size); if Gemini is unavailable we
    fall back to the fast Groq model. Same verbatim / NOT FOUND / CONFLICT contract as
    extract_field. On total failure, returns ERROR per field so the agent and the safety
    net react rather than crash.
    """
    prompt = f"""You are reading a hospital patient record. Extract the fields below.

{_EXTRACT_RULES}

RECORD:
{document}

Each value MUST be a plain string (use newlines inside the string for lists; never a
JSON array). Return ONLY a JSON object with EXACTLY these keys (one per field):
{json.dumps(fields, indent=2)}"""
    try:
        out = extract_json(prompt)               # Gemini: big context, accurate
    except Exception as e_gemini:
        try:
            out = ask_json(prompt, smart=False)  # fallback: fast Groq model
        except Exception as e_groq:
            err = e_groq if _too_big_or_throttled(e_gemini) else e_gemini
            return {f: f"ERROR: extraction failed ({_short(err)})" for f in fields}
    if not isinstance(out, dict):
        return {f: "ERROR: extraction failed (non-JSON response)" for f in fields}
    return {f: _as_text(out.get(f, "NOT FOUND")) for f in fields}


def _as_text(value) -> str:
    """Coerce an extracted value to clean text. Flattens a stray JSON list into a
    newline list so the draft never shows Python/JSON syntax."""
    if isinstance(value, list):
        return "\n".join(f"- {str(v).strip()}" for v in value if str(v).strip()) or "NOT FOUND"
    return str(value).strip() or "NOT FOUND"


def extract_field(document: str, field: str) -> str:
    """Find ONE field in the record.

    Returns the value, or 'NOT FOUND', or a string starting with 'CONFLICT'
    when documents disagree. The model is told to copy only what is written.
    """
    prompt = f"""You are reading a hospital patient record.
Find this one field and nothing else: {field}

THE SAME FIELD MAY BE LABELLED DIFFERENTLY:
- principal diagnosis: may be written as "final diagnosis", "primary diagnosis",
  "provisional diagnosis", or just "diagnosis".
- hospital course: may be "course in hospital", "clinical course", or described
  across the doctor/nursing notes.
- discharge condition: may be "condition at discharge" or "status at discharge".
- admission medications: look in admission drug orders, medication charts, or
  ER/ICU treatment records.

RULES:
- Copy only information explicitly written in the record. Never guess or assume.
- If it genuinely is not written anywhere, reply with exactly: NOT FOUND
- Words like "Not known" or "Nil" written in the record are real documented
  answers, not NOT FOUND.

CONFLICT vs LIST — read carefully:
- Only single-answer fields can conflict: principal diagnosis, admission/discharge
  dates, patient demographics, discharge condition. Reply starting with the word
  CONFLICT (then list each value with its page) ONLY when the record states genuinely
  DIFFERENT values for one of these. Do not pick one.
- The SAME value written under different labels (e.g. provisional vs final diagnosis
  that both say "pneumonia") is NOT a conflict — report it once.
- List-type fields (secondary diagnoses, procedures, follow-up instructions,
  pending results, medications) are NOT conflicts: COMBINE the documented items
  from every page into a single list.

RECORD:
{document}

Answer for "{field}":"""
    try:
        return ask(prompt)
    except Exception as e:
        return f"ERROR: extraction failed ({_short(e)})"


def reconcile_medications(admission: str, discharge: str) -> str:
    """Compare admission vs discharge medications and surface every change.

    Handles the common real-world case where admission meds were never written
    down: instead of silently skipping, it says so clearly so the clinician knows
    reconciliation could not be done.
    """
    if not admission or "NOT FOUND" in admission.upper() or "MISSING" in admission.upper():
        return ("CANNOT RECONCILE: admission medications are not documented in the record. "
                "Every discharge medication must be checked by the clinician against the "
                "patient's actual pre-admission regimen.")
    if not discharge or "NOT FOUND" in discharge.upper() or "MISSING" in discharge.upper():
        return "CANNOT RECONCILE: discharge medications are not documented in the record."

    prompt = f"""Compare these two medication lists from one hospital stay.

ADMISSION MEDICATIONS:
{admission}

DISCHARGE MEDICATIONS:
{discharge}

Using only what is written, list:
1. CONTINUED  (in both)
2. STOPPED    (in admission, not in discharge)
3. ADDED      (in discharge, not in admission)
4. CHANGED    (dose or frequency different)

For every STOPPED, ADDED or CHANGED medication, look for a documented reason.
If there is no documented reason, append to that line: -> FLAG: no documented reason."""
    try:
        return ask(prompt)
    except Exception as e:
        return f"ERROR: reconciliation failed ({_short(e)})"


def check_drug_interactions(discharge_meds: str) -> str:
    """Check the discharge medication list for interactions.

    The reply begins with one of two fixed lines so the agent can tell, without
    guessing, whether it needs to escalate:
        INTERACTIONS FOUND:
        NO SIGNIFICANT INTERACTIONS
    """
    if not discharge_meds or "NOT FOUND" in discharge_meds.upper() or "MISSING" in discharge_meds.upper():
        return "NO CHECK POSSIBLE: no discharge medications to check."

    prompt = f"""You are a clinical pharmacist reviewing a discharge medication list:

{discharge_meds}

Identify only well-established, clinically significant drug-drug interactions.
Start your reply with EXACTLY one of these two lines:
INTERACTIONS FOUND:
NO SIGNIFICANT INTERACTIONS
If any are found, list each interacting pair and the risk on one line each. Be concise."""
    try:
        return ask(prompt)
    except Exception as e:
        return f"ERROR: drug check failed ({_short(e)})"
