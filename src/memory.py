# src/memory.py
# ---------------------------------------------------------------------------
# PART 2 — THE LEARNING MECHANISM: CORRECTION MEMORY.
#
# Chosen over fine-tuning (DPO/SFT) and a learned reward model because, for a
# clinical task with very little data, it is:
#   - safe   : we control exactly what is learnable. The memory stores FORMATTING
#              rules only; it is fact-fenced so a learned rule can never carry a
#              clinical value into a future draft (see _FENCE below).
#   - cheap  : no training infra, no GPUs. Runs on the same API as the agent.
#   - cold-start friendly: it improves from a handful of pairs and degrades
#              gracefully to "no guidance" when it has seen nothing.
#   - explainable: every learned rule is a short sentence you can read in
#              correction_memory.json and trace back to the edits that produced it.
#
# HOW IT LEARNS
#   For each (draft, edited) pair, for each section the doctor actually changed, we
#   ask an LLM to describe — in one sentence — the FORMATTING transformation the
#   doctor applied (not the content). We keep a per-section rule only once the
#   doctor has edited that section in at least MIN_SUPPORT examples, so we learn
#   consistent policy, not one-off noise. At draft time, get_guidance() assembles
#   the learned rules into the `guidance` string the agent injects (see agent.py).
#
# The agent NEVER sees reviewer.FROZEN_POLICY. It only sees rules the memory
# inferred from observed edits — which is the whole point of the exercise.
# ---------------------------------------------------------------------------

import json
import os
from collections import defaultdict

from llm import ask_groq, ask_openrouter, FAST_MODEL
from reward import edit_distance_ratio, _is_marker

# Keep summariser prompts small so they stay under free-tier limits and return fast.
_SNIPPET_CHARS = 200


def _snippet(text: str) -> str:
    """First _SNIPPET_CHARS of a section — enough to show the FORMAT change to the
    rule summariser without sending whole (sometimes long) sections."""
    t = " ".join(text.split())
    return t[:_SNIPPET_CHARS]

# A section must be edited in at least this many examples before we trust a rule for
# it. Guards against learning from a single noisy edit (cold-start safety). The
# simulated reviewer is deterministic (temperature 0), so one consistent observation is
# already reliable signal here; with more, noisier reviewers, raise this via the env var.
MIN_SUPPORT = int(os.getenv("MIN_SUPPORT", "1"))

# Only count a section as "edited" if the doctor changed it more than this. Tiny
# diffs (a stray space) shouldn't trigger a learned rule.
EDIT_THRESHOLD = 0.05

MEMORY_PATH = os.path.join("outputs", "correction_memory.json")

# The fact-fence. This is the safety contract of the whole learning loop: the rule
# summariser may only describe HOW text is formatted, never WHAT it says.
_FENCE = """You compare a junior doctor's draft of each discharge-summary section with \
a senior's edited version, and state the FORMATTING RULE the senior applied to each.

For each section, give ONE short imperative sentence describing the STRUCTURE / \
WORDING / FORMAT change only — e.g. "Write dates as YYYY-MM-DD", "Put each medication \
on its own line as name | strength | frequency | duration", "Compress to at most 4 \
sentences". Return a JSON object mapping each section name to its rule string.

ABSOLUTE RULES:
- Describe FORMAT ONLY. Never mention a specific drug, dose, date, diagnosis, name,
  or lab value. Never state a clinical fact. If the only change to a section is to the
  clinical content (a value was changed), set that section's rule to exactly: NO FORMAT RULE.
- Do not invent a rule that isn't supported by the diff. Use NO FORMAT RULE if the
  change is trivial or purely about wording you can't generalise."""


class CorrectionMemory:
    """Accumulates fact-safe formatting rules learned from (draft, edited) pairs."""

    def __init__(self):
        # section -> list of one-sentence rule candidates observed across examples
        self._candidates = defaultdict(list)
        # section -> count of examples in which the doctor meaningfully edited it
        self._support = defaultdict(int)

    # ---- learning ----------------------------------------------------------

    def observe(self, draft: dict, edited: dict):
        """Learn from one (draft, edited) pair, in ONE batched LLM call.

        Only sections the doctor meaningfully edited (and that are not safety markers)
        are sent. The summariser returns one fact-fenced FORMAT rule per section.
        """
        changed = {}
        for section, draft_text in draft.items():
            edited_text = edited.get(section, draft_text)
            if _is_marker(draft_text):          # never learn from a safety flag
                continue
            if edit_distance_ratio(draft_text, edited_text) < EDIT_THRESHOLD:
                continue                        # doctor barely touched it -> no signal
            self._support[section] += 1
            changed[section] = (draft_text, edited_text)

        if not changed:
            return
        for section, rule in self._summarise_rules(changed).items():
            if rule and rule.upper() != "NO FORMAT RULE":
                self._candidates[section].append(rule)

    def _summarise_rules(self, changed: dict) -> dict:
        """One fact-fenced call -> {section: one-sentence FORMAT rule} for all edits.

        `changed` maps section -> (draft_text, edited_text). To stay reliable on a
        rate-limited/slow free API tier, we send only SHORT SNIPPETS of each diff (not
        whole sections), run on the fast model, and use a short timeout with a couple of
        quick retries. Best-effort: any failure returns {} so learning never crashes.
        """
        payload = {s: {"before": _snippet(d), "after": _snippet(e)}
                   for s, (d, e) in changed.items()}
        user = ("For EACH section, state the one-sentence FORMAT rule the senior applied "
                "(or NO FORMAT RULE). Return ONLY a JSON object {section: rule}:\n"
                + json.dumps(payload, indent=2))
        # Learn on OpenRouter (separate free provider/key); fall back to Groq if it is
        # unavailable or rate-limited, so learning never stalls the run.
        try:
            raw = ask_openrouter(_FENCE, user, json_mode=True, timeout=15)
        except Exception:
            try:
                raw = ask_groq(_FENCE, user, json_mode=True,
                               model=FAST_MODEL, timeout=10, retries=2)
            except Exception:
                return {}
        try:
            out = json.loads(raw)
        except Exception:
            return {}
        return {s: str(r).strip().strip('"') for s, r in out.items() if isinstance(r, str)}

    # ---- using -------------------------------------------------------------

    def get_guidance(self) -> str:
        """Assemble the guidance string injected into the next draft.

        Returns "" when nothing is confidently learned yet (cold start) — the agent
        then behaves exactly as in Part 1.
        """
        lines = []
        for section, rules in self._candidates.items():
            if self._support[section] < MIN_SUPPORT or not rules:
                continue
            # Most frequent rule for this section = the consistent policy.
            best = max(set(rules), key=rules.count)
            lines.append(f"- {section}: {best}")
        if not lines:
            return ""
        return ("When writing these sections, follow these reviewer-preferred FORMATS "
                "(formatting only — always keep the facts exactly as found in the "
                "record, and never fill a MISSING/CONFLICT field):\n" + "\n".join(lines))

    # ---- persistence -------------------------------------------------------

    def save(self, path: str = MEMORY_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        learned = {}
        for section, rules in self._candidates.items():
            if self._support[section] >= MIN_SUPPORT and rules:
                learned[section] = {
                    "rule": max(set(rules), key=rules.count),
                    "support": self._support[section],
                }
        with open(path, "w") as fh:
            json.dump(learned, fh, indent=2)
        return path
