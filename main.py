# main.py — entry point.
#
#   uv run python main.py             -> Part 1 for every patient
#   uv run python main.py patient_2   -> Part 1 for just one patient (by id)
#   uv run python main.py --part2     -> Part 2 only: the learning experiment
#   uv run python main.py --all       -> Part 1 for every patient, then Part 2
#
# Part 1 reads each patient's PDF, runs the agent, and writes a draft + trace to
# outputs/. Part 2 (opt-in) runs the doctor-edit learning loop and writes the
# before/after curve + metrics to outputs/part2/. It is opt-in because it makes
# many more LLM calls than a single Part 1 pass.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))   # make src/ importable

from pdf_reader import read_pdf
from agent import run_agent
from output_builder import build_output

PATIENTS = [
    ("patients/patient_1/patient_1.pdf", "patient_1"),   # synthetic pneumonia case
    ("patients/patient_2/patient_2.pdf", "patient_2"),   # real scanned record (conflicts + gaps)
]


def process(pdf_path, patient_id):
    print(f"\n{'=' * 54}\nPROCESSING {patient_id}\n{'=' * 54}")
    print("[1/3] Reading PDF ...")
    document = read_pdf(pdf_path)

    print("[2/3] Running agent ...")
    findings, trace, flags = run_agent(document, patient_id)

    print("[3/3] Writing outputs ...")
    summary_path, trace_path = build_output(findings, trace, flags, patient_id)
    print(f"  summary -> {summary_path}")
    print(f"  trace   -> {trace_path}")
    print(f"  flags   -> {len(flags)}")


def run_part1(only=None):
    """Run Part 1 for every patient, or just one when `only` is a patient id."""
    selected = [(p, pid) for p, pid in PATIENTS if not only or pid == only]
    if not selected:
        print(f"No patient named {only!r}. Available: {[pid for _, pid in PATIENTS]}")
        return
    for pdf_path, patient_id in selected:
        process(pdf_path, patient_id)


def run_part2():
    # Imported lazily so a plain Part 1 run never needs matplotlib / the Part 2 deps.
    from learn import run_experiment
    run_experiment()


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--part2" in args:
        run_part2()
    elif "--all" in args:
        run_part1()
        run_part2()
    else:
        # An optional patient id (e.g. "patient_2") runs just that one; no arg = all.
        only = next((a for a in args if not a.startswith("-")), None)
        run_part1(only)
