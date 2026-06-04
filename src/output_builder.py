# src/output_builder.py
# ---------------------------------------------------------------------------
# Renders the agent's findings into two files per patient:
#   discharge_summary.md  - the draft, clearly marked for review
#   trace.txt             - the full step-by-step trace
# ---------------------------------------------------------------------------

import os
from datetime import datetime

# (internal key, display title) in the order they appear in the summary.
SECTION_ORDER = [
    ("patient demographics (name, age, gender)", "Patient Demographics"),
    ("admission and discharge dates",            "Admission & Discharge Dates"),
    ("principal diagnosis",                      "Principal Diagnosis"),
    ("secondary diagnoses",                      "Secondary Diagnoses"),
    ("hospital course",                          "Hospital Course"),
    ("procedures",                               "Procedures"),
    ("admission medications",                    "Admission Medications"),
    ("discharge medications",                    "Discharge Medications"),
    ("medication reconciliation",                "Medication Reconciliation"),
    ("allergies",                                "Allergies"),
    ("follow-up instructions",                   "Follow-Up Instructions"),
    ("pending results",                          "Pending Results"),
    ("discharge condition",                      "Discharge Condition"),
    ("drug interaction check",                   "Drug Interaction Check"),
]


def _render(value: str) -> str:
    """Turn a raw finding into clinician-facing text, making any gap obvious."""
    up = value.upper()
    if "NOT FOUND" in up:
        return "⚠ **MISSING — not documented in the record. Clinician review required.**"
    if up.startswith("CONFLICT"):
        return ("⚠ **CONFLICT — documents disagree; clinician must resolve (no value chosen).**\n\n"
                + value)
    if up.startswith("ERROR") or "CANNOT RECONCILE" in up or "NO CHECK POSSIBLE" in up:
        return "⚠ " + value
    return value


def build_output(findings, trace, flags, patient_id):
    out_dir = os.path.join("outputs", patient_id)
    os.makedirs(out_dir, exist_ok=True)

    # ----- discharge_summary.md -----
    md = [
        "# DISCHARGE SUMMARY",
        f"_Generated {datetime.now():%Y-%m-%d %H:%M}_",
        "**DRAFT — FOR CLINICIAN REVIEW ONLY. NOT A FINALISED CLINICAL DOCUMENT.**",
        "\n---\n",
    ]
    for key, title in SECTION_ORDER:
        md.append(f"## {title}")
        md.append(_render(findings.get(key, "NOT FOUND")))
        md.append("")

    md.append("\n---\n")
    md.append("## ⚠ Flags for Clinician Review")
    if flags:
        md += [f"- **{f['field']}** — {f['reason']}" for f in flags]
    else:
        md.append("None.")

    summary_path = os.path.join(out_dir, "discharge_summary.md")
    with open(summary_path, "w") as fh:
        fh.write("\n".join(md))

    # ----- trace.txt -----
    tlines = [
        f"AGENT TRACE — {patient_id}",
        f"Generated {datetime.now():%Y-%m-%d %H:%M}",
        "=" * 54,
    ]
    tlines += trace
    tlines += ["\n" + "=" * 54, f"Flags raised: {len(flags)}"]
    tlines += [f"  - {f['field']}: {f['reason']}" for f in flags]

    trace_path = os.path.join(out_dir, "trace.txt")
    with open(trace_path, "w") as fh:
        fh.write("\n".join(tlines))

    return summary_path, trace_path
