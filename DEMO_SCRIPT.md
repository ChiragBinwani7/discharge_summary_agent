VIDEO DEMO SCRIPT (Target: ~3-4 minutes)
==============================================================

*This script is designed for you to read out loud word-for-word. Actions you need to perform on your screen are marked with [ACTION]. Keep a steady pace. It's okay to pause or take a breath while typing.*

**BEFORE RECORDING:**
1. Open your terminal in the project directory.
2. Open your code editor so you have easy access to the `outputs/patient_1`, `outputs/patient_2`, and `outputs/part2` folders.
3. Start your screen recorder (e.g., Loom, OBS, QuickTime).

---

[ACTION] Start with your terminal visible on the screen.

**SAY:** 
"Hi! This is a demo of my Agentic AI System for Discharge Summaries. It reads messy, unstructured clinical documents, like scanned PDFs, and drafts a structured clinical discharge summary for a doctor to review. The most important rule of this system is that it never invents a clinical fact. Let me run the pipeline live so you can see it in action."

[ACTION] Type the following command in the terminal and hit Enter: `uv run python main.py --all`

**SAY:**
"I'm running the agent over all patients, including both Part 1 and Part 2. This is not a hardcoded pipeline. The agent uses a ReAct loop—it plans, extracts clinical data, observes the results, and then decides what to reconcile or flag. It's also hard-capped at 25 steps to ensure it never gets stuck in an infinite loop."

[ACTION] While the final outputs finish printing in the terminal, open the file: `outputs/patient_2/trace.txt`

**SAY:** 
"Let's look at Patient 2. This patient had a messy, scanned hospital record. Everything the agent does is observable. Here in the step trace, you can see the agent's reasoning, action, input, and result. At Step 1, it attempted to extract all fields but found that one section was missing, and another had conflicting information."

[ACTION] Switch to the file: `outputs/patient_2/discharge_summary.md` and scroll down to the "Flags" section at the very bottom.

**SAY:**
"Instead of hallucinating or making a dangerous guess, the agent safely escalates. If you look at the flags at the bottom of the drafted summary, it explicitly flagged that patient demographics were missing from the record. It also found conflicting values across different pages for the principal diagnosis, so it preserved both values and flagged it for the clinician to resolve."

[ACTION] Still in `outputs/patient_2/discharge_summary.md`, highlight or point to the Drug Interactions flag.

**SAY:**
"In addition to data extraction, we have strict safety backstops. You can see it correctly identified a dangerous drug interaction between Risperidone and Loperamide, alerting the clinician to the risk of QT prolongation."

[ACTION] Switch back to the terminal output to show the Part 2 metrics (e.g., Edit distance: 0.119 -> 0.089).

**SAY:**
"Finally, for Part 2, I implemented a learning loop. The system simulates a deterministic 'doctor' fixing the draft format. I built a memory system that extracts the doctor's formatting preferences and applies them to future patients."

[ACTION] Open `outputs/part2/improvement_curve.png` if it exists, otherwise just gesture to the numbers in the terminal.

**SAY:**
"As you can see from our held-out test patient, the learning loop reduced the clinician's needed edits by about 25 percent. But crucially, thanks to our fact-scoring reward system, the fact preservation score remained a perfect 1.0 the entire time. This proves the agent learned to format the output better strictly without dropping or inventing a single clinical metric to artificially lower the edit distance. 

Thank you for watching!"

---
END OF SCRIPT
