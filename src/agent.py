# src/agent.py
# ---------------------------------------------------------------------------
# THE AGENT LOOP.
#
# This is not a fixed pipeline. On every step the model looks at what has been
# done and what is still missing, then CHOOSES the next action itself. It
# re-plans when a tool returns NOT FOUND, a CONFLICT, or an error.
#
#     plan -> act -> observe -> decide -> repeat        (hard-capped at MAX_STEPS)
#
# After the loop a small deterministic backstop runs, so that even if the model
# forgets, nothing missing / conflicting / interacting is ever left unflagged.
# ---------------------------------------------------------------------------

import json

from llm import ask_json
from tools import (
    extract_field, extract_all_fields, reconcile_medications, check_drug_interactions,
)

MAX_STEPS = 25          # hard cap: the agent can never run forever (requirement: control)

# The 12 sections the discharge summary must contain.
REQUIRED_SECTIONS = [
    "patient demographics (name, age, gender)",
    "admission and discharge dates",
    "principal diagnosis",
    "secondary diagnoses",
    "hospital course",
    "procedures",
    "admission medications",
    "discharge medications",
    "allergies",
    "follow-up instructions",
    "pending results",
    "discharge condition",
]

# The menu of actions the agent may pick from on each step.
TOOL_MENU = """\
- extract_all_fields()          : read ALL remaining sections from the record in one go
                                  (each -> value / NOT FOUND / CONFLICT). Use this FIRST.
- extract_field(field)          : re-read ONE field (use only to re-check a single section)
- flag_for_review(field, reason): mark something for the clinician (missing, conflict, safety)
- reconcile_medications()       : compare admission vs discharge meds (after both are extracted)
- check_drug_interactions()     : check discharge meds for interactions (after they are extracted)
- finish()                      : stop, once every section is handled"""


def _canonical(field: str) -> str:
    """Snap the model's field name to the exact section name we track, so progress
    is counted correctly even if the wording drifts a little."""
    f = field.strip().lower()
    for s in REQUIRED_SECTIONS:
        if f == s.lower() or f in s.lower() or s.lower() in f:
            return s
    return field


def _controller_prompt(state: dict, guidance: str = "") -> str:
    """Build the small prompt that asks the model for its next action.

    The full record is NOT sent here (only inside extract_field), so each decision
    step is cheap and based on PROGRESS, not raw text.

    `guidance` (Part 2) is learned, fact-safe formatting advice injected by the
    correction memory. It only ever changes HOW a value is written, never WHAT the
    value is — extraction still copies facts verbatim from the record.
    """
    handled = {k: (v[:80] + "...") if len(v) > 80 else v
               for k, v in state["findings"].items()}
    remaining = [s for s in REQUIRED_SECTIONS if s not in state["findings"]]

    guidance_block = (
        f"\nLEARNED FORMATTING GUIDANCE (how reviewers prefer sections written — "
        f"affects wording/format ONLY, never the facts):\n{guidance}\n"
        if guidance else ""
    )

    return f"""You are an agent writing a DRAFT discharge summary for a clinician to review.
Your most important rule: NEVER invent a clinical fact. If something is not in the
record it must be extracted as NOT FOUND and flagged — never guessed.

{guidance_block}
ACTIONS YOU CAN TAKE:
{TOOL_MENU}

REQUIRED SECTIONS (use these names EXACTLY for the "field" value):
{json.dumps(REQUIRED_SECTIONS, indent=2)}

SECTIONS ALREADY HANDLED:
{json.dumps(handled, indent=2) if handled else "(none yet)"}

SECTIONS STILL TO DO:
{json.dumps(remaining, indent=2) if remaining else "(none — every section is handled)"}

FLAGS RAISED SO FAR:
{json.dumps(state["flags"], indent=2) if state["flags"] else "(none yet)"}

RECENT STEPS:
{chr(10).join(state["history"][-6:]) if state["history"] else "(this is the first step)"}

Choose the SINGLE best next action. Guidance:
- If sections are still to do, call extract_all_fields ONCE to read them all together.
- If an extraction came back NOT FOUND, flag that field as missing.
- If an extraction came back CONFLICT, flag it so the clinician resolves it (never pick one).
- Once admission AND discharge medications are extracted, reconcile them.
- Once discharge medications are extracted, check drug interactions; if interactions are
  found, flag them.
- Only finish when nothing remains and every missing/conflicting field has been flagged.

Reply as JSON with these keys:
  "reasoning": one sentence on why this action now,
  "action": one of [extract_all_fields, extract_field, flag_for_review, reconcile_medications, check_drug_interactions, finish],
  "field": the section name (for extract_field / flag_for_review; otherwise ""),
  "reason": the flag reason (for flag_for_review; otherwise "")"""


def _next_action(state: dict, guidance: str = "") -> dict:
    """Ask the model what to do next. Raises if the answer is not a JSON object."""
    decision = ask_json(_controller_prompt(state, guidance))
    if not isinstance(decision, dict):
        raise ValueError("controller did not return a JSON object")
    return decision


def run_agent(document: str, patient_id: str, guidance: str = ""):
    """Run the agent on one patient. Returns (findings, trace, flags).

    `guidance` (Part 2): optional learned, fact-safe formatting advice from the
    correction memory. Empty by default, so Part 1 behaviour is unchanged.
    """
    state = {"findings": {}, "flags": [], "history": []}
    trace = []
    step = 0

    def record(reasoning, action, tool_input, result, decision):
        """Write one observable trace entry: reasoning -> action -> input -> result -> next."""
        nonlocal step
        step += 1
        entry = (
            f"\n{'-' * 54}\n"
            f"STEP {step}\n"
            f"  REASONING : {reasoning}\n"
            f"  ACTION    : {action}\n"
            f"  INPUT     : {tool_input}\n"
            f"  RESULT    : {str(result)[:350]}\n"
            f"  DECISION  : {decision}"
        )
        trace.append(entry)
        print(entry)

    print(f"\n=== Agent starting: {patient_id} (max {MAX_STEPS} steps) ===")

    while step < MAX_STEPS:
        # 1. PLAN: ask the model for its next action.
        try:
            decision = _next_action(state, guidance)
        except Exception as e:
            # If the model's choice can't be read, fall back to extracting the next
            # missing section, so we always make safe progress instead of crashing.
            remaining = [s for s in REQUIRED_SECTIONS if s not in state["findings"]]
            decision = {
                "reasoning": f"controller unreadable ({str(e).splitlines()[0][:80]}); defaulting to next section",
                "action": "extract_field" if remaining else "finish",
                "field": remaining[0] if remaining else "",
                "reason": "",
            }

        action = decision.get("action", "")
        field = decision.get("field", "")
        reason = decision.get("reason", "")
        reasoning = decision.get("reasoning", "")

        # 2. ACT + OBSERVE: run the chosen tool and store what came back.
        if action == "finish":
            record(reasoning, "finish", "", "agent declared all sections handled", "stop loop")
            break

        elif action == "extract_all_fields":
            remaining = [s for s in REQUIRED_SECTIONS if s not in state["findings"]]
            results = extract_all_fields(document, remaining or REQUIRED_SECTIONS)
            n_missing = n_conflict = 0
            for fld, val in results.items():
                fld = _canonical(fld)
                state["findings"][fld] = val
                up = val.upper()
                if "NOT FOUND" in up:
                    n_missing += 1
                elif up.startswith("CONFLICT"):
                    n_conflict += 1
            summary = (f"extracted {len(results)} sections "
                       f"({n_missing} NOT FOUND, {n_conflict} CONFLICT)")
            note = "now flag every NOT FOUND / CONFLICT, then reconcile + interaction-check"
            state["history"].append(f"extract_all_fields() -> {summary}")
            record(reasoning, "extract_all_fields", "all remaining sections", summary, note)

        elif action == "extract_field":
            field = _canonical(field)
            result = extract_field(document, field)
            state["findings"][field] = result
            up = result.upper()
            if "NOT FOUND" in up:
                note = "missing -> should flag for clinician"
            elif up.startswith("CONFLICT"):
                note = "conflict -> should flag, do NOT pick one"
            elif up.startswith("ERROR"):
                note = "tool error -> should flag for clinician"
            else:
                note = "value stored from record"
            state["history"].append(f"extract_field({field}) -> {result[:60]}")
            record(reasoning, "extract_field", field, result, note)

        elif action == "flag_for_review":
            field = _canonical(field)
            state["flags"].append({"field": field, "reason": reason})
            state["history"].append(f"flag_for_review({field})")
            record(reasoning, "flag_for_review", field, reason, "flag recorded for clinician")

        elif action == "reconcile_medications":
            result = reconcile_medications(
                state["findings"].get("admission medications", ""),
                state["findings"].get("discharge medications", ""),
            )
            state["findings"]["medication reconciliation"] = result
            note = ("could not reconcile -> consider flagging"
                    if "CANNOT RECONCILE" in result else "reconciliation stored")
            state["history"].append("reconcile_medications()")
            record(reasoning, "reconcile_medications", "admission vs discharge", result, note)

        elif action == "check_drug_interactions":
            result = check_drug_interactions(state["findings"].get("discharge medications", ""))
            state["findings"]["drug interaction check"] = result
            note = ("interactions present -> should flag"
                    if result.upper().startswith("INTERACTIONS FOUND") else "no action needed")
            state["history"].append("check_drug_interactions()")
            record(reasoning, "check_drug_interactions", "discharge meds", result, note)

        else:
            # Unknown action name: record it and let the loop re-plan next step.
            record(reasoning, action or "unknown", field, "unrecognised action; re-planning", "ignored")
            state["history"].append(f"ignored unknown action: {action}")

    else:
        # The while-loop ended because the step cap was hit, not via finish().
        print(f"  ! reached step cap ({MAX_STEPS}); stopping.")

    # --- Safety backstop: the two safety-critical checks must ALWAYS run. ---
    # The agent is prompted to call these itself (and the trace shows when it does),
    # but a forgotten reconciliation or interaction check must never silently leave
    # the draft unchecked. If the loop did not run one, run it now.
    if "medication reconciliation" not in state["findings"]:
        result = reconcile_medications(
            state["findings"].get("admission medications", ""),
            state["findings"].get("discharge medications", ""),
        )
        state["findings"]["medication reconciliation"] = result
        note = "could not reconcile -> will flag" if "CANNOT RECONCILE" in result else "reconciliation stored"
        record("safety backstop: reconciliation was not performed during the loop",
               "reconcile_medications", "admission vs discharge", result, note)

    if "drug interaction check" not in state["findings"]:
        result = check_drug_interactions(state["findings"].get("discharge medications", ""))
        state["findings"]["drug interaction check"] = result
        note = ("interactions present -> will flag"
                if result.upper().startswith("INTERACTIONS FOUND") else "no escalation needed")
        record("safety backstop: interaction check was not performed during the loop",
               "check_drug_interactions", "discharge meds", result, note)

    _apply_safety_net(state)

    print(f"=== Agent finished: {step} steps used, {len(state['flags'])} flags ===\n")
    return state["findings"], trace, state["flags"]


def _apply_safety_net(state: dict):
    """Deterministic backstop run after the loop.

    The agent is meant to flag missing/conflicting fields itself, but we never rely
    on that alone for safety. Here we guarantee that:
      - every required section exists (unprocessed ones become NOT FOUND),
      - every missing / conflicting / failed field is flagged,
      - any drug interactions found are flagged.
    """
    flagged = {f["field"] for f in state["flags"]}

    for section in REQUIRED_SECTIONS:
        value = state["findings"].get(section)
        if value is None:                                   # never processed before cap
            state["findings"][section] = value = "NOT FOUND"

        up = value.upper()
        if section not in flagged and (
            "NOT FOUND" in up or up.startswith("CONFLICT") or up.startswith("ERROR")
        ):
            reason = ("not documented in the record" if "NOT FOUND" in up
                      else "conflicting values across documents — clinician must resolve"
                      if up.startswith("CONFLICT")
                      else "extraction failed — clinician review required")
            state["flags"].append({"field": section, "reason": reason})
            flagged.add(section)

    # Medication reconciliation is a clinician action item when EITHER it could not
    # be done, OR it found an add/stop/change with no documented reason. The tool
    # marks those lines with "FLAG: no documented reason"; we must surface that to
    # the flag list, not leave it buried inside the reconciliation text.
    recon = state["findings"].get("medication reconciliation", "")
    recon_up = recon.upper()
    if "medication reconciliation" not in flagged and (
        "CANNOT RECONCILE" in recon_up or "NO DOCUMENTED REASON" in recon_up
    ):
        reason = (recon if "CANNOT RECONCILE" in recon_up
                  else "medication added/stopped/changed with no documented reason — "
                       "clinician must reconcile (see Medication Reconciliation section)")
        state["flags"].append({"field": "medication reconciliation", "reason": reason})

    # Drug interactions are safety-critical: always escalate if any were found.
    dic = state["findings"].get("drug interaction check", "")
    if dic.upper().startswith("INTERACTIONS FOUND") and "drug interactions" not in flagged:
        state["flags"].append({"field": "drug interactions", "reason": dic})
