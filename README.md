# Discharge Summary AI Agent - My Submission

**Video Demo Link**: [https://www.loom.com/share/e968e77e449947c98864d40788618f48](https://www.loom.com/share/e968e77e449947c98864d40788618f48)

## 1. Agent Loop Design
Let's talk about how the agent actually works under the hood. Instead of a hardcoded, step-by-step pipeline, I built it around a dynamic ReAct (Reason, Act, Observe) loop. The controller LLM doesn't just blindly read the whole text at once; instead, it tracks its progress across the 12 required clinical sections. At each step, it looks at what's missing, picks a tool from a strict menu (like extracting fields or checking for drug interactions), and plans its next move. I capped it at 25 steps so it can never get stuck in an infinite loop. Once it finishes, I wrote a deterministic backstop script that double-checks everything was handled correctly.

## 2. Enforcing the "No-Fabrication" Guardrail
In healthcare, hallucinating facts is a complete dealbreaker. To make sure the agent is safe, I built several layers of defense so it never relies entirely on the LLM's imagination:
* **Verbatim Extractions:** The extraction tools are strictly prompted to copy exact quotes or return `NOT FOUND`.
* **Restricted Context:** The main agent loop never actually sees the raw PDF text. It only sees the parsed results from its tools, completely removing its ability to invent random details.
* **Hardcoded Backstops:** Once the LLM finishes, pure Python logic audits the output. It forces medication reconciliation, checks for interactions, and explicitly flags any `NOT FOUND` or `CONFLICT` states for the doctor.
* **Strict Scoring (Part 2):** My reward system immediately penalizes the agent if it drops any numbers, dosages, or lab values from the original text.

## 3. Handling Failures and Conflicts
The system is designed to fail safely. For example, `patient_2` is missing demographics and has conflicting baseline diagnoses in the notes. When the tools spot this, the loop recognizes the `NOT FOUND` and `CONFLICT` states. Instead of guessing or picking a favorite diagnosis, it stops trying to extract those fields and uses the `flag_for_review` tool to escalate them directly to the doctor in the final draft. It behaves similarly in `patient_1` when it notices a medication was stopped without any documented reason.

## 4. Part 2: Reward Design and Results
For the stretch goal, I wanted the agent to learn a doctor's formatting preferences without accidentally learning to make things up just to please them. I bypassed subjective LLM graders and wrote a pure Python reward function that tracks two things: an edit reward (reducing the edit distance to the doctor's preferred format) and a strict fact score (ensuring no original clinical facts got dropped).
* **The Results:** Over a few iterations, the simulated learning loop reduced the doctor's required edits by about **25%**! Even better, the fact preservation score stayed at a perfect `1.0`. It successfully learned how to format the data better without dropping or hallucinating a single medical detail.

## 5. Limitations & What I Would Do With More Time
* **Limitations:** Right now, the factual extraction relies on regexes to track numbers and specific words. It works fast but lacks deep medical understanding (for example, it doesn't know Acetaminophen and Paracetamol are the exact same medication). Also, the system uses standard Python dictionaries instead of typed schemas like Pydantic, and multimodal OCR can sometimes make transcription mistakes with messy handwriting.
* **If I had more time:** Given another week, I'd integrate a medical knowledge graph (like UMLS) so the system actually understands the clinical concepts it's handling. I'd also format the final outputs into standard FHIR JSON objects and build a clean front-end UI for doctors to review and approve the drafts easily. Adding more synthetic patients to expand the Part 2 training would be awesome too!