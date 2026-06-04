Discharge Summary Agent
=======================

An agentic AI system that reads a patient's messy source-note PDFs and produces a
structured discharge-summary draft for clinician review. It plans, uses tools,
recovers from missing/conflicting/failed data, and — above all — never invents a
clinical fact. The output is always a draft, never a finalised clinical document.

Part 2 adds a learning loop: a simulated reviewer ("doctor") edits the drafts, and a
fact-safe correction memory learns the reviewer's formatting preferences so future
drafts need fewer edits — without ever learning to fabricate.


Quick start
-----------

  # 1. Install (uv recommended; uses pyproject.toml + uv.lock)
  uv sync
  #    or, with plain pip:  pip install -r requirements.txt

  # 2. API keys (kept out of the repo — see .gitignore)
  cat > .env <<'EOF'
  GEMINI_API_KEY=your_gemini_key
  GROQ_API_KEY=your_groq_key      # only needed for Part 2
  EOF

  # 3. (test data) regenerate the synthetic patient_1 PDF if needed
  uv run python scripts/make_patient_1.py

  # 4. Part 1 — run the agent on every patient (or one at a time)
  uv run python main.py                # all patients
  uv run python main.py patient_2      # just one, by id
  #    -> outputs/<patient>/discharge_summary.md   (the draft)
  #    -> outputs/<patient>/trace.txt              (the step trace)
  #    -> outputs/<patient>/raw_text.txt           (cached PDF transcription)

  # 5. Part 2 — run the learning experiment (~12s with cached base drafts)
  uv run python main.py --part2     # or: uv run python src/learn.py
  #    -> outputs/part2/metrics.json
  #    -> outputs/part2/improvement_curve.png
  #    -> outputs/correction_memory.json

  # Or run everything (Part 1 for each patient, then Part 2) in one go:
  uv run python main.py --all

Models / keys (kept in .env, gitignored):
  - GEMINI_API_KEY  — reads the scanned PDFs (multimodal) and does the one batched
    field-extraction call per patient (large context, accurate).
  - GROQ_API_KEY    — the agent's cheap controller decisions (fast 8b model).
  - OPENROUTER_API_KEY (Part 2 only) — the rule-learning summariser; falls back to Groq
    if absent or rate-limited.
Any provider works — the call sites are all in src/llm.py. The project venv is Python
3.14; requires-python is set to >=3.10 for portability.


Repository layout
-----------------

  main.py                     Entry point: read PDF -> run agent -> write draft + trace.
  src/pdf_reader.py           PDF ingestion. Gemini multimodal transcribes scanned/
                              handwritten pages; result cached to raw_text.txt.
  src/agent.py                The agent loop + the deterministic safety backstop.
  src/tools.py                Tools: extract_all_fields (batched), extract_field,
                              reconcile_medications, check_drug_interactions.
  src/llm.py                  The single place that talks to the LLMs (Gemini + Groq +
                              OpenRouter), with pacing, timeouts and retry/fallback.
  src/output_builder.py       Renders discharge_summary.md and trace.txt.
  src/reviewer.py             Part 2 — simulated doctor: a fixed, hidden policy in CODE.
  src/reward.py               Part 2 — edit-distance reward + anti-gaming fact score.
  src/memory.py               Part 2 — correction memory; learns format rules from edits.
  src/formatter.py            Part 2 — applies learned rules to a draft (deterministic).
  src/learn.py                Part 2 — experiment driver; before/after curve.
  scripts/make_patient_1.py   Generates the synthetic patient_1 test PDF.


Part 1 — agent design
---------------------

The loop (not a hardcoded pipeline)

On every step the controller LLM is shown progress, not raw text: which sections are
handled, which remain, which flags exist, and the last few steps. It chooses one action
from a fixed menu, then observes the result and re-plans:

  plan -> act -> observe -> decide -> (repeat, hard-capped at MAX_STEPS = 25)

Actions: extract_all_fields, extract_field, flag_for_review, reconcile_medications,
check_drug_interactions, finish. The agent genuinely re-plans on what it reads — e.g. a
section that comes back CONFLICT leads it to flag_for_review next; it only calls
reconcile_medications once both med lists are present, and escalates a drug interaction
when one is found. extract_all_fields reads every section in one batched call (instead
of one call per field) — the agent still decides to call it, inspects the results, and
chooses what to flag/reconcile/escalate/finish. (See outputs/patient_*/trace.txt for the
reasoning -> action -> input -> result -> decision trail.)

Why a small from-scratch loop rather than a framework: clinical safety needs every
decision to be inspectable and every guardrail to be one we control. The loop is ~120
readable lines and each tool is a plain function, so every choice the agent makes can be
explained and audited.

The no-fabrication guardrail (the core requirement)

Enforced in two independent layers, so we never rely on the LLM alone:

  1. Extraction is verbatim-only. extract_field instructs the model to copy only what is
     written and reply exactly NOT FOUND if a field is absent. It distinguishes a genuine
     conflict (different values for a single-answer field -> reply begins CONFLICT, lists
     every value with its page, picks none) from list fields that should simply be
     merged. Documented answers like "Nil"/"Not known" are kept as real answers, not
     treated as missing.

  2. A deterministic safety backstop (_apply_safety_net, runs after the loop) guarantees
     that regardless of what the LLM did: every required section exists (unprocessed ->
     NOT FOUND), every NOT FOUND / CONFLICT / ERROR field is flagged, med reconciliation
     always runs, drug interactions always get checked, and any found interaction is
     escalated. A forgotten flag is impossible.

The renderer (output_builder.py) turns each marker into an unmissable clinician cue
(MISSING, CONFLICT, ...) and stamps every draft
"DRAFT — FOR CLINICIAN REVIEW ONLY."

How failures and conflicts are handled

  - Tool/read failures never crash. Every tool catches exceptions and returns an
    "ERROR: ..." string the agent can react to; the backstop then flags it for review.
  - Rate limits / transient errors are paced and retried with server-suggested backoff in
    src/llm.py (the free Gemini tier throttles aggressively). A throttled call waits and
    retries — it never silently behaves as if it succeeded.
  - Unreadable controller output falls back to "extract the next missing section", so the
    agent always makes safe forward progress instead of dying.
  - Conflicts are surfaced, never resolved. The agent lists each conflicting value with
    its page and flags the field; the clinician decides.
  - Medication reconciliation compares admission vs. discharge meds and labels each as
    continued / stopped / added / changed. A change with no documented reason is flagged
    rather than silently resolved. If admission meds were never recorded, it says so
    explicitly (CANNOT RECONCILE) instead of guessing a baseline.
  - Control: the loop is hard-capped (MAX_STEPS); it can never run forever.


Part 2 — learning from doctor edits
-----------------------------------

Goal: turn clinician edits into signal so future drafts need fewer corrections — without
degrading the Part 1 safety guarantees.

The loop:  agent draft  ->  simulated doctor edits it  ->  reward measures edit burden
                                  |                                      |
                                  +--> correction memory learns format --+
                                       rules, applied to the NEXT draft

1. Reward / accuracy signal (reward.py)
Pure-Python (stable, free, no LLM judging itself). Per section:
  - edit_reward = 1 - normalised_edit_distance(draft, doctor's edit) -> less editing,
    higher reward.
  - fact_score = fraction of source facts (numbers, doses, dates, drug/Dx tokens) still
    present in the draft -> the anti-gaming guard. Dates are compared format-insensitively
    (fact_key), so reformatting 12/05/2026 -> 2026-05-12 is NOT counted as a lost fact,
    but a dropped dose/lab still is.
  - reward = edit_reward - (1 - fact_score): dropping a fact caps the reward hard, so the
    loop can never "win" by getting vaguer or dropping the medicine.

2. Simulated reviewer (reviewer.py) — deterministic, hidden policy
A stand-in "doctor" applies a fixed house style: ISO dates, one-line-per-med layout
(name | strength | frequency | duration), <=4-sentence hospital course, numbered
follow-up. It is implemented as CODE, not an LLM call. Why: an LLM reviewer (even at
temperature 0) gave run-to-run variance that made the improvement curve bounce and
sometimes vanish; deterministic code is the strongest form of "a consistent, hidden
policy", so any measured change is attributable to LEARNING, not reviewer noise. The
agent never imports this module; the learner must INFER the policy from (draft, edited)
pairs. Marker/flag sections (MISSING, CONFLICT, ⚠, ...) are returned completely unchanged
— a reviewer never fills a flagged gap.

3. Learning mechanism (memory.py + formatter.py) — correction memory
Chosen over DPO/SFT or a learned reward model because, for a tiny-data clinical task, it
is safe (we control exactly what's learnable), cheap (no training infra), cold-start
friendly, and explainable (every rule is a readable sentence in correction_memory.json).
For each section the doctor changed, an LLM (OpenRouter, with a Groq fallback) is shown
SHORT before/after snippets and asked, behind a strict fact-fence, for the one-sentence
FORMAT rule — never a clinical fact. That inference is the genuine learning step.

Applying a learned rule, by contrast, is mechanical, so formatter.py does it in CODE
(the same ISO-date / pipe-med / compress / numbered transforms). This keeps the
measurement deterministic: same rules -> same reformat -> a clean, reproducible curve.
Two safety properties hold: (a) marker/flag sections are passed through untouched;
(b) a fact fence — if a reformat would drop a source fact (compared via reward.fact_key),
it is rejected and the original kept. So learning changes how a value is written, never
what it is.

4. Measured improvement (learn.py)  [outputs/part2/metrics.json, improvement_curve.png]
The memory learns on a train patient (patient_2) and is measured on a held-out patient
(patient_1) it never observed. Because the learned rules are about format
(patient-independent), a drop in edit burden on the held-out patient is a transferable
improvement, not memorised text. Result on the held-out patient:

    iteration 0 (cold start, 0 rules):  mean edit distance 0.119   fact_score 1.00
    iteration 1+ (4 rules learned):     mean edit distance ~0.089  fact_score 1.00
    => ~25% lower edit burden, with facts fully preserved (the safety guard holds).

Base drafts are cached to disk, so re-running the whole experiment takes ~12s and is
reproducible.

5. Limitations (and how safety is preserved)
  - Cold start / tiny data: only ~2 patients, so the curve is short and the LLM
    rule-summariser is slightly noisy (a learned rule set can wobble between iterations,
    giving a small bounce). More patients would smooth and deepen it. The system degrades
    gracefully to "no guidance" (= exact Part 1 behaviour) when nothing is learned.
  - Partial transfer: the deterministic formatter applies rules cleanly for dates (edit
    distance for that section -> ~0); some layouts (e.g. a comma-separated med list that
    does not round-trip perfectly) reduce edits less, which is why the gain is ~25% rather
    than ~100%. This is honest headroom, not a safety issue.
  - Gaming risk: optimising for fewer edits can reward vagueness. The fact_score guard
    penalises that directly, and the learnable surface is format only — the loop cannot
    alter a clinical value or fill a flagged gap. Safety markers are never learned from.
  - Reviewer is synthetic: a real clinician's edits would be noisier and multi-dimensional;
    the method generalises, but the absolute numbers are only as meaningful as the policy.

What we tried (and why the design landed here)
  - LLM-based reviewer AND LLM-based rule application: produced an unstable, sometimes
    flat curve (same input, different output each call) and was slow on free API tiers.
    -> moved both the reviewer and the rule-application to deterministic code; the LLM is
       kept only where it adds value — inferring the format rule from edits.
  - One LLM call per field for extraction: ~28 calls/patient, slow and rate-limited.
    -> batched into one extract_all_fields call (patient_1 draft went ~98s -> ~6s).
  - Free-tier rate limits / stalls: added per-request timeouts, retries with backoff,
    a Gemini-for-extraction / Groq-for-control split, OpenRouter for the learner with a
    Groq fallback, and on-disk caching so a re-run never depends on live API latency.


Status and what I'd do with more time
-------------------------------------

Done: Part 1 (all 10 hard requirements) on both patients — fast (~6s patient_1, ~16s the
large scanned patient_2), with conflicts, missing data, the unexplained med change, and a
drug interaction all correctly flagged/escalated. Part 2 runs end-to-end in ~12s and shows
a ~25% reduction in edit burden on a held-out patient with fact_score held at 1.0.

With more time: more synthetic patients to deepen and smooth the Part 2 curve; make the
deterministic formatter cover more layouts (e.g. comma-separated med lists) so the gain is
larger; a small unit-test suite for the guardrails (assert MISSING/CONFLICT always flag);
and a structured (JSON) draft output alongside the Markdown for EHR integration.

Synthetic data only. patient_1 is generated by scripts/make_patient_1.py; no real patient
data is used or submitted.
