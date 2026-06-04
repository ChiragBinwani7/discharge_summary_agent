# src/learn.py
# ---------------------------------------------------------------------------
# PART 2 — THE EXPERIMENT DRIVER.
#
# Ties the loop together and produces the before/after proof the task asks for:
#
#     agent drafts  ->  simulated doctor edits  ->  reward measures edit burden
#                          |                                   |
#                          +--> correction memory learns ------+
#                               fact-safe FORMAT rules
#                               (injected into the NEXT draft)
#
# FAIR MEASUREMENT (held-out set)
#   The memory LEARNS formatting rules from the TRAIN patient(s) only. We MEASURE
#   edit burden on a HELD-OUT patient the memory never observed. Because the
#   doctor's policy is about FORMAT (ISO dates, med layout, compressed course),
#   which is patient-independent, an improvement on a held-out patient proves the
#   agent learned a TRANSFERABLE format — not memorised text. That is the honest,
#   strong version of "show measurable improvement".
#
# THE CURVE
#   Iteration 0 = no learning yet (baseline edit burden on the held-out patient).
#   Iteration k = memory has observed k training passes; we re-draft the held-out
#   patient WITH the learned guidance and re-measure. Edit burden should fall while
#   fact_score stays ~1.0 (the safety guard holding).
# ---------------------------------------------------------------------------

import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pdf_reader import read_pdf
from agent import run_agent
from formatter import format_draft
from reviewer import review_draft
from reward import draft_reward, extract_facts
from memory import CorrectionMemory

OUT_DIR = os.path.join("outputs", "part2")
CACHE_DIR = os.path.join(OUT_DIR, "cache")

# Part 2 depends on live LLM calls (base extraction + the simulated reviewer). The
# reviewer runs at temperature 0, so its output for a given draft is deterministic;
# base drafts are likewise meant to be fixed across iterations. We therefore cache
# both to disk: the FIRST run computes and saves them; later runs replay instantly and
# IDENTICALLY. This makes the improvement curve fast to (re)generate and reproducible,
# instead of being at the mercy of API latency. Delete outputs/part2/cache to recompute.

def _cache_path(kind: str, key: str) -> str:
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"{kind}_{h}.json")


def _load_cache(kind: str, key: str):
    p = _cache_path(kind, key)
    if os.path.exists(p):
        with open(p) as fh:
            return json.load(fh)
    return None


def _save_cache(kind: str, key: str, value: dict):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(_cache_path(kind, key), "w") as fh:
        json.dump(value, fh, indent=2)


def cached_review(draft: dict) -> dict:
    """review_draft, memoised by draft content. Deterministic reviewer (temp 0) means
    the cached edit is exactly what a live call would return."""
    key = json.dumps(draft, sort_keys=True)
    hit = _load_cache("review", key)
    if hit is not None:
        return hit
    edited = review_draft(draft)
    _save_cache("review", key, edited)
    return edited

# train patients teach the memory; held-out patients are only ever measured.
# Held-out = patient_1 (small/fast) so each measurement iteration is quick and the
# improvement curve is cheap to (re)generate; train = patient_2 (the larger record).
TRAIN = [("patients/patient_2/patient_2.pdf", "patient_2")]
HELDOUT = [("patients/patient_1/patient_1.pdf", "patient_1")]

ITERATIONS = 4   # 0 = baseline (cold start), then 3 rounds of accumulated learning


def _base_draft(pdf_path, patient_id):
    """Run the full agent ONCE to get the base draft (facts, guidance-independent),
    cached to disk so the experiment is reproducible and fast to re-run.

    The agent extracts each section's CONTENT verbatim; guidance only ever changes
    PRESENTATION, applied later by format_draft(). So this expensive step is done a
    single time per patient and reused across every learning iteration.
    """
    hit = _load_cache("base", patient_id)
    if hit is not None:
        print(f"  (cached base draft: {patient_id})")
        return hit
    document = read_pdf(pdf_path)               # cached to disk after first read
    findings, _trace, _flags = run_agent(document, patient_id)
    _save_cache("base", patient_id, findings)
    return findings


def _source_facts(document_findings):
    """The facts each section must preserve = the facts in the agent's own draft of
    it. (We protect against the LEARNING loop later dropping a fact it once had.)"""
    return {section: extract_facts(text) for section, text in document_findings.items()}


def _measure(base_drafts, guidance):
    """Apply guidance to each held-out patient's base draft, have the doctor edit it,
    and score it. Facts are pinned to the BASE draft, so any fact lost by the
    formatting pass would show up as a fact_score drop (the safety guard)."""
    eds, rewards, facts = [], [], []
    for base in base_drafts:
        draft = format_draft(base, guidance)        # cheap formatting pass
        edited = cached_review(draft)
        scores = draft_reward(draft, edited, _source_facts(base))
        eds.append(scores["mean_edit_distance"])
        rewards.append(scores["mean_reward"])
        facts.append(scores["mean_fact_score"])
    n = len(base_drafts)
    return {
        "mean_edit_distance": round(sum(eds) / n, 4),
        "mean_reward": round(sum(rewards) / n, 4),
        "mean_fact_score": round(sum(facts) / n, 4),
    }


def run_experiment():
    os.makedirs(OUT_DIR, exist_ok=True)
    memory = CorrectionMemory()
    history = []

    print("\n=== PART 2: learning from doctor edits ===")
    print(f"train: {[p for _, p in TRAIN]}   held-out: {[p for _, p in HELDOUT]}\n")

    # Draft every patient's base ONCE up front (the only full agent runs we make).
    print("drafting base summaries once (reused across all iterations) ...")
    train_bases   = [_base_draft(p, pid) for p, pid in TRAIN]
    heldout_bases = [_base_draft(p, pid) for p, pid in HELDOUT]

    for it in range(ITERATIONS):
        guidance = memory.get_guidance()        # "" on iteration 0 (cold start)

        # 1) MEASURE on the held-out patient with the guidance learned SO FAR.
        m = _measure(heldout_bases, guidance)
        m["iteration"] = it
        m["rules_learned"] = len([l for l in guidance.splitlines() if l.startswith("- ")])
        history.append(m)
        print(f"[iter {it}] held-out edit_distance={m['mean_edit_distance']:.3f}  "
              f"reward={m['mean_reward']:.3f}  fact_score={m['mean_fact_score']:.3f}  "
              f"rules={m['rules_learned']}")

        # 2) LEARN from the train patient(s): format -> doctor edits -> memory observes.
        for base in train_bases:
            draft = format_draft(base, guidance)
            edited = cached_review(draft)
            memory.observe(draft, edited)

    memory.save()
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as fh:
        json.dump(history, fh, indent=2)

    _plot(history)
    _report(history)
    return history


def _plot(history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    its = [h["iteration"] for h in history]
    ed = [h["mean_edit_distance"] for h in history]
    fs = [h["mean_fact_score"] for h in history]

    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax1.plot(its, ed, "o-", color="#c0392b", label="mean edit distance (lower = better)")
    ax1.set_xlabel("learning iteration")
    ax1.set_ylabel("mean edit distance", color="#c0392b")
    ax1.set_ylim(0, max(ed) * 1.25 + 0.01)
    ax1.set_xticks(its)

    ax2 = ax1.twinx()
    ax2.plot(its, fs, "s--", color="#27ae60", label="mean fact score (safety guard)")
    ax2.set_ylabel("mean fact score", color="#27ae60")
    ax2.set_ylim(0, 1.05)

    heldout = ", ".join(pid for _, pid in HELDOUT)
    trained = ", ".join(pid for _, pid in TRAIN)
    plt.title(f"Part 2: edit burden drops while facts are preserved\n"
              f"measured on held-out {heldout}  (learned from {trained})")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "improvement_curve.png")
    plt.savefig(path, dpi=130)
    print(f"\ncurve -> {path}")


def _report(history):
    first, last = history[0], history[-1]
    drop = first["mean_edit_distance"] - last["mean_edit_distance"]
    pct = (drop / first["mean_edit_distance"] * 100) if first["mean_edit_distance"] else 0.0
    print("\n=== before / after (held-out) ===")
    print(f"  edit distance : {first['mean_edit_distance']:.3f} -> {last['mean_edit_distance']:.3f}"
          f"   ({pct:+.1f}% edit burden)")
    print(f"  fact score    : {first['mean_fact_score']:.3f} -> {last['mean_fact_score']:.3f}"
          f"   (safety preserved if ~1.0)")
    print(f"  metrics -> {os.path.join(OUT_DIR, 'metrics.json')}")
    print(f"  memory  -> {os.path.join('outputs', 'correction_memory.json')}\n")


if __name__ == "__main__":
    run_experiment()
    # The HTTP client libraries can keep non-daemon worker threads alive after the work
    # is done; we've written all outputs, so exit immediately rather than hang on them.
    os._exit(0)
