# Project Plan — Bearing Fault Diagnosis & Predictive Maintenance

**Edge AI + Reinforcement Learning + Conversational AI, for a PA (Primary Air) Fan**

Status: **DRAFT — awaiting review.** No code will be written until this plan is approved.

---

## Part A — The project in plain words

### A1. The problem

A power plant has large fans called **PA (Primary Air) fans**. They blow air that carries
coal powder into the boiler. If a PA fan stops unexpectedly, the boiler loses fuel feed,
output drops, and the plant can trip. Repairs during an unplanned stop are expensive and rushed.

The part that most often fails in such a fan is the **ball bearing** — the ring of steel balls
that lets the shaft spin smoothly. A bearing does not usually break suddenly. It starts with a
tiny crack or pit on one of its surfaces, and that damage grows over weeks or months.

A damaged bearing **vibrates differently**. Every time a ball rolls over a crack, it produces a
tiny "knock". These knocks repeat at a rate that depends on *where* the crack is:

| Where the damage is | Name | Knocks per shaft revolution (6205 bearing) |
|---|---|---|
| Outer ring (stationary) | Outer race fault | ≈ 3.58 |
| Inner ring (rotating) | Inner race fault | ≈ 5.42 |
| A ball | Ball fault | ≈ 4.71 |

So by listening to the vibration carefully, you can tell **whether** a bearing is damaged,
**where** the damage is, and roughly **how bad** it is — long before it fails.

### A2. What this project builds

A software system that acts like an **automated maintenance engineer**:

1. **Listens** — takes a vibration recording from a sensor on the bearing.
2. **Analyses** — computes measurements from the signal (how spiky it is, which frequencies are strong, etc.).
3. **Diagnoses** — a Machine Learning model (Random Forest) says: *Normal*, *Inner race fault*, *Outer race fault*, or *Ball fault*, and how confident it is.
4. **Judges severity** — a second model estimates how big the damage is, and rules turn this into **Healthy / Warning / Critical**.
5. **Decides** — a Reinforcement Learning agent recommends what to do: *keep running*, *monitor more often*, *schedule an inspection*, *reduce load*, or *stop for maintenance*.
6. **Explains** — a chat assistant (Google Gemini) explains the result and answers questions in plain language, using only the real numbers from steps 1–5.
7. **Shows it all** — a web dashboard (Streamlit) where the engineer does all of this by clicking.

It is also designed so the diagnosis part is **small and fast enough to run on a small device**
(like a Raspberry Pi) mounted next to the fan — that is the **Edge AI** part.

### A3. Where the data comes from (and an honest caveat)

We do not have recordings from a real PA fan. We use the **CWRU Bearing Dataset** from Case
Western Reserve University — a well-known public dataset where researchers deliberately damaged
bearings (6205 type, the same family used in many fans and motors) and recorded vibration at
different loads and speeds.

- CWRU is a **lab test rig** (a 2 HP motor), not a PA fan.
- The PA fan is the **target application** — the system is built so it *could* be fitted to one,
  but it has **not** been tested on one.
- The maintenance agent is trained in a **simulation** of how a bearing wears out. The costs,
  failure chances and outcomes it uses are **simulated**, not real plant data.

The dashboard and README will state these points clearly.

### A4. What each "AI" part actually does

| Part | Simple description | Why it is there |
|---|---|---|
| **Machine Learning (Random Forest)** | Learns from thousands of labelled vibration snippets what each fault "looks like" in numbers | To diagnose the fault automatically |
| **Edge AI** | Makes that model small and fast, exports it in a portable format, and measures its speed/size/memory | So it could run on a cheap device at the machine, without the cloud |
| **Reinforcement Learning (PPO)** | An agent plays a simulated "maintenance game" millions of times — running the fan earns money, breakdowns cost a lot, maintenance costs a little — and learns the best action for each situation | To recommend *what to do*, not just *what is wrong* |
| **Conversational AI (Gemini)** | A chatbot that receives the real results as data and explains them | So engineers can ask "why?" in plain language |

### A5. How someone will use it

**One-time setup (the developer, ~30–60 minutes, mostly waiting):**

1. Install Python and the project's packages.
2. Run the scripts that download the CWRU data, train the models, and train the RL agent.
3. Put a Gemini API key in a `.env` file (optional — chat works in a simpler offline mode without it).

**Day-to-day use (the engineer, a few minutes):**

1. Open the dashboard in a web browser (`streamlit run app/Home.py`).
2. In the sidebar, **upload a vibration recording** (or pick one of the bundled sample recordings),
   and enter the sampling rate, shaft speed (RPM) and load.
3. Click **Analyze**. Within a second or two the system shows:
   - the signal and its frequency spectrum, with the bearing fault frequencies marked;
   - the diagnosis, e.g. **"Outer race fault — 94% confidence"**;
   - the estimated damage size and a health status, e.g. **"Warning"**;
   - the RL agent's recommendation, e.g. **"Schedule inspection"**, and how it compares to a simple rule-based policy.
4. Go to the **AI Assistant** page and ask, for example:
   - *"Why do you think it's an outer race fault?"* → "The envelope spectrum has a strong peak at 107 Hz, which is 3.58× the shaft speed — the outer race defect frequency…"
   - *"What happens if we keep running?"* → the system runs the simulator forward and reports the **simulated** risk.
   - *"Explain this to a non-engineer."*
5. Check the **Edge AI** page to see how large and fast the model is, and download it.

---

## Part B — Decisions already made

| Topic | Decision |
|---|---|
| Dataset | CWRU, drive-end 6205 bearing, 4 classes (Normal, Inner race, Outer race, Ball), all 4 loads (0–3 HP ↔ ~1797/1772/1750/1730 RPM), fault sizes 0.007"/0.014"/0.021"/0.028" where available |
| Sampling rate | Everything at **12 kHz**. The CWRU healthy recordings are at 48 kHz and will be properly down-sampled (anti-alias filtered) to 12 kHz. 12 kHz is used because the 0.028" faults only exist at 12 kHz, and a lower rate is cheaper on edge devices |
| ML model | scikit-learn Random Forest (fault type) + a second Random Forest (fault size) |
| RL | Custom Gymnasium environment + PPO from Stable-Baselines3 |
| LLM | Google Gemini via the official `google-genai` SDK; model name configurable; API key supplied later via `.env` |
| Dashboard | Streamlit (multipage) + Plotly |
| Platform | Windows (developed and tested on Windows 11). Should also run on Linux, but not required |
| Language | Python 3.11 |

---

## Part C — System architecture

### C1. Two phases

**Phase 1 — Training (run once, offline, via scripts):**

```
CWRU .mat files
   │  load drive-end signal, resample to 12 kHz
   ▼
Windowing (2048 samples, 50% overlap)
   ▼
Feature extraction (time + frequency + envelope)  ──►  data/features/features.csv
   ▼
Train Fault RF  +  Severity RF   (evaluated with leave-one-load-out)
   │                                   │
   ├──► models/fault_rf.joblib, fault_rf.onnx, severity_rf.joblib, ...
   └──► test confusion matrices ──────────────┐
                                              ▼
                       Gymnasium degradation simulator
                       (uses the real confusion matrices to make
                        the agent's "observations" realistically noisy)
                                              ▼
                       PPO training  ──►  models/ppo_agent.zip
                                              ▼
                       Evaluate vs baselines  ──►  reports/
```

**Phase 2 — Runtime (every time the engineer clicks Analyze):**

```
Uploaded signal + fs + RPM + load
   ▼
analyze(signal, fs, rpm, load)            ← ONE function, in pdm/pipeline.py
   ├─ resample + window + features
   ├─ Fault RF      → class + probabilities (averaged over windows)
   ├─ Severity RF   → estimated fault size
   ├─ Health rules  → Healthy / Warning / Critical
   ├─ PPO agent     → recommended action (+ rule-based policy's action for comparison)
   └─ Simulator     → "what if we keep running" risk (simulated)
   ▼
AnalysisReport  (one Python object containing all of the above)
   ├──► Dashboard pages (charts, tables, cards)
   └──► Gemini chat (receives the report as JSON — nothing else)
```

**Key rule:** the dashboard and the chat never compute or invent anything themselves. They only
display / explain the `AnalysisReport`. This makes hardcoded or made-up results structurally impossible.

### C2. Folder structure

```
new proj/
├── plan.md                    ← this file
├── README.md                  ← install + run instructions
├── requirements.txt
├── config.yaml                ← all settings: window size, RF params, RL costs, Gemini model, health rules
├── .env.example               ← GEMINI_API_KEY=  (real .env is never committed)
│
├── pdm/                       ← the core Python package (no UI code here)
│   ├── config.py              ← loads config.yaml
│   ├── data/
│   │   ├── cwru_catalog.py    ← table of every CWRU file: class, size, load, rpm, fs, URL
│   │   ├── download.py        ← downloads CWRU files into data/raw/
│   │   └── loader.py          ← reads .mat/.csv/.npy, extracts drive-end signal, resamples
│   ├── signal/
│   │   ├── windowing.py
│   │   ├── features.py        ← time, frequency, envelope features (NumPy/SciPy only)
│   │   └── bearing.py         ← 6205 geometry + defect frequencies (BPFO/BPFI/BSF/FTF)
│   ├── ml/
│   │   ├── train.py           ← fault + severity RF training
│   │   ├── evaluate.py        ← metrics, confusion matrices, feature-group comparison
│   │   └── predict.py         ← loads models, predicts per recording
│   ├── health.py              ← rule-based Healthy/Warning/Critical
│   ├── edge/
│   │   ├── export.py          ← ONNX export
│   │   └── benchmark.py       ← size, latency, memory measurements
│   ├── rl/
│   │   ├── env.py             ← Gymnasium BearingMaintenanceEnv
│   │   ├── baselines.py       ← rule-based, run-to-failure, periodic, random policies
│   │   ├── train.py           ← PPO training
│   │   └── evaluate.py        ← policy comparison
│   ├── llm/
│   │   ├── context.py         ← AnalysisReport → JSON for the LLM
│   │   ├── gemini_client.py   ← Gemini calls
│   │   ├── fallback.py        ← template answers when no API key / offline
│   │   └── guard.py           ← checks numbers in answers against the report
│   └── pipeline.py            ← analyze() → AnalysisReport
│
├── scripts/                   ← run these in order (or run_all.py)
│   ├── 01_download_data.py
│   ├── 02_build_features.py
│   ├── 03_train_classifier.py
│   ├── 04_edge_benchmark.py
│   ├── 05_train_rl.py
│   ├── 06_evaluate_rl.py
│   ├── 07_make_figures.py     ← all report plots + architecture diagram
│   └── run_all.py
│
├── edge/
│   └── edge_infer.py          ← standalone: needs only numpy + scipy + onnxruntime
│
├── app/                       ← Streamlit dashboard
│   ├── Home.py
│   ├── components/            ← shared sidebar, cards, chart helpers
│   └── pages/
│       ├── 1_Signal_Analysis.py
│       ├── 2_Fault_Diagnosis.py
│       ├── 3_Predictive_Maintenance.py
│       ├── 4_AI_Assistant.py
│       └── 5_Edge_AI.py
│
├── data/
│   ├── raw/                   ← downloaded CWRU .mat files (not committed)
│   ├── features/              ← features.csv
│   └── samples/               ← a few small demo recordings for the dashboard
├── models/                    ← trained models (.joblib, .onnx, .zip)
├── reports/
│   ├── figures/               ← PNG plots
│   └── metrics/               ← JSON/CSV results
└── tests/                     ← pytest tests for every module
```

### C3. Tech stack

| Purpose | Packages |
|---|---|
| Core | numpy, pandas, scipy, pyyaml, python-dotenv |
| ML | scikit-learn, joblib |
| Edge | skl2onnx, onnxruntime, psutil |
| RL | gymnasium, stable-baselines3 (PyTorch CPU) |
| LLM | google-genai |
| Dashboard | streamlit, plotly |
| Plots/reports | matplotlib |
| Testing | pytest |

Exact versions will be pinned in `requirements.txt` after they are installed and tested together.

---

## Part D — Module-by-module plan

Each module: **what it does → what it produces → how we know it works.**

### Module 0 — Project setup
- Create virtual environment, `requirements.txt`, `config.yaml`, folder structure, `.env.example`.
- **Done when:** `pip install -r requirements.txt` succeeds on Windows and `pytest` runs (empty).

### Module 1 — Data & signal processing
**Tasks**
1. `cwru_catalog.py`: a table of every CWRU file we use — fault type, fault size, load (HP),
   RPM, sampling rate, file number, URL. Outer race uses the **@6 o'clock** position (the
   centred/standard one). Each entry is verified against the official CWRU site.
2. `download.py`: download into `data/raw/`, skip files already present, verify each file opens
   and contains a drive-end (`_DE_time`) signal. If the website blocks scripted downloads, the
   README explains how to download manually.
3. `loader.py`: read CWRU `.mat` (handles the known quirks — e.g. some files contain more than
   one `_DE_time` variable), plus user uploads in `.mat`, `.csv`, `.npy`. Resample anything to
   12 kHz with an anti-aliasing filter.
4. `windowing.py`: split into windows of 2048 samples (≈0.17 s, ≈5 shaft revolutions) with 50% overlap.
5. `features.py` — per window:
   - **Time domain:** mean, std, RMS, min, max, peak-to-peak, **kurtosis, skewness, crest factor, impulse factor, shape factor**
   - **Frequency domain (FFT):** spectral centroid, dominant frequency, energy in fixed frequency bands (e.g. 8 bands up to 6 kHz)
   - **Envelope domain:** band-pass filter → Hilbert envelope → spectrum; energy at **BPFO, BPFI, BSF** (and their 2nd/3rd harmonics) and at 1× shaft speed, using the RPM
6. `02_build_features.py` → `data/features/features.csv` (one row per window, with recording ID, fault, size, load, RPM columns).

**Done when**
- Unit tests on **synthetic signals** pass (e.g. a sine wave gives the right RMS and dominant
  frequency; a signal with impulses at 107 Hz lights up the BPFO feature).
- Feature CSV exists; a sanity plot shows faults clearly separated on kurtosis/envelope features.

### Module 2 — Fault diagnosis (Random Forest)
**Tasks**
1. Train an RF to predict the 4 classes from the features. `class_weight="balanced"` because there is less healthy data.
2. **Evaluation — leave-one-load-out (LOLO):** train on 3 loads, test on the 4th, repeat for all 4
   → report accuracy, precision, recall, F1 (per class + macro) and confusion matrix, as mean ± std over folds.
   Every recording is entirely in either train or test — no leakage.
3. **Leakage demonstration:** also report the score from a naive random-window split, to show
   why recording-level separation matters (expected: naive ≈ 100%, LOLO somewhat lower).
4. **Feature-group comparison:** train/evaluate with (a) time only, (b) frequency only,
   (c) envelope only, (d) all → bar chart.
5. **Recording-level prediction:** at runtime, average window probabilities over the whole
   uploaded recording → final class + confidence.
6. **Feature importance** plot (also used by the chat to explain "why").
7. Final model is trained on all loads and saved to `models/fault_rf.joblib`.

**Done when:** metrics JSON + confusion-matrix and comparison plots exist in `reports/`; tests pass.

### Module 3 — Severity & health assessment
**Tasks**
1. **Severity RF:** trained only on faulty windows, predicting fault size
   (0.007" / 0.014" / 0.021" / 0.028"), evaluated with LOLO the same way. Confusion matrix reported.
   *Severity is never derived from classifier confidence.*
2. **Health rules** (`health.py`, rules stored in `config.yaml`, clearly labelled "rule-based estimate"):

   | Condition | Health |
   |---|---|
   | Predicted Normal with confidence ≥ 0.70 | **Healthy** |
   | Fault predicted, estimated size 0.007" | **Warning** (early stage) |
   | Fault predicted, estimated size 0.014" | **Warning** (moderate) — raised to **Critical** at loads ≥ 2 HP |
   | Fault predicted, estimated size ≥ 0.021" | **Critical** |
   | Top-class confidence < 0.60 (any class) | Status shown + **"Low confidence — re-measure"** flag |

**Done when:** severity metrics/plots in `reports/`; unit tests cover every rule.

### Module 4 — Edge AI
**Tasks**
1. **Size vs accuracy study:** train RFs with trees ∈ {10, 25, 50, 100, 200} and max depth ∈
   {8, 12, 16, unlimited}; plot LOLO F1 against model size. Pick the smallest model within ~1% F1 of the best.
2. **Export** the chosen model to **ONNX** (`skl2onnx`) and check ONNX predictions match scikit-learn predictions exactly on the test set.
3. **Benchmark** (1000 repeated runs, report median and 95th percentile):
   - feature extraction time per window
   - model inference time per window (joblib and ONNX)
   - total end-to-end time per window
   - model file sizes on disk
   - peak memory (`tracemalloc`) and process memory (`psutil`)
4. **`edge/edge_infer.py`:** standalone script that only needs NumPy, SciPy and onnxruntime —
   reads a signal file, prints the diagnosis. This is what would be copied to a Raspberry Pi.
5. All numbers labelled **"measured on development laptop (Windows, CPU), not on edge hardware."**

**Done when:** benchmark JSON + plots exist; `edge_infer.py` gives the same answer as the main pipeline.

### Module 5 — Reinforcement Learning maintenance agent (simulation)
**The simulated "maintenance game":**

- **One step = one simulated day. One episode = 180 days** (or until failure).
- **Hidden truth** (the agent cannot see it directly):
  - fault type (fixed per episode; can start healthy and develop a fault later)
  - degradation level 0 → 1 (grows randomly each day; grows **faster at higher load**)
  - degradation maps onto the CWRU fault sizes (0.007" → 0.028")
  - daily failure probability rises sharply as degradation approaches 1
- **What the agent observes** (like a real engineer with a sensor):
  - diagnosed fault type + confidence — **sampled from the real fault-RF confusion matrix**, so the agent experiences realistic misdiagnoses
  - estimated severity — sampled from the real severity-RF confusion matrix
  - health status (from the same rules as Module 3)
  - current load and RPM
  - days since last inspection, and whether frequent monitoring is on
- **Actions:**

  | # | Action | Effect in simulation |
  |---|---|---|
  | 0 | Continue operation | Full production, normal wear |
  | 1 | Monitor more frequently | Small cost; observations become more accurate |
  | 2 | Schedule inspection | Moderate cost, short downtime; reveals true fault and degradation |
  | 3 | Reduce operating load | Lower production; slower wear and lower failure risk |
  | 4 | Stop for maintenance | Large cost + downtime; bearing replaced (degradation reset). Extra penalty if the bearing was actually healthy (unnecessary maintenance) |

- **Reward (per day)** = production value − maintenance/inspection/monitoring cost − downtime loss
  − **failure cost** (very large, ends episode) − **safety penalty** (for running while truly critical).
  All cost values are in `config.yaml` and labelled as illustrative assumptions.

**Tasks**
1. Implement `BearingMaintenanceEnv` (Gymnasium) and validate it with Gymnasium's / SB3's environment checkers.
2. Baseline policies: **rule-based** (Healthy→continue, Warning→monitor/inspect, Critical→stop), **run-to-failure**, **periodic maintenance** (every N days, N tuned), **random**.
3. Train **PPO** (`MlpPolicy`, ~500k steps, 3 random seeds, CPU).
4. Evaluate every policy on the same 1000 simulated episodes: average reward, failures per episode,
   unnecessary maintenance, downtime days, availability %.
5. Outputs: training reward curve, policy comparison bar charts, a sample episode timeline
   (degradation + actions over 180 days), and an **example decisions table** (state → PPO action vs rule action).
6. *Optional:* compute the exact optimal policy for a simplified discrete version (value iteration) and show PPO comes close.

**At runtime:** the dashboard's real `AnalysisReport` is converted into the same observation format and
fed to the trained PPO agent → recommended action + action probabilities.

**Done when:** env passes checks; PPO beats run-to-failure and random clearly, and matches or
beats the rule-based policy on average reward; all plots in `reports/`. If PPO does *not* beat the
rule-based policy, that is reported honestly with analysis, not hidden.

### Module 6 — Conversational AI (Gemini)
**Tasks**
1. `context.py`: convert the `AnalysisReport` into a compact JSON: diagnosis, probabilities,
   severity, health, top contributing features (compared to healthy-bearing averages from the
   training data), defect-frequency peaks, RL action + its probabilities, rule-based action,
   simulated "continue operating" risk, and the data/simulation caveats.
2. `gemini_client.py`: send system prompt + JSON context + chat history + question to Gemini
   (low temperature). System prompt rules: *use only the provided data; never invent readings,
   predictions or actions; say "not available" if something is missing; always label simulated numbers as simulated.*
3. **What-if questions** ("what if we keep running?") are answered by actually running the
   simulator from the current state with "continue" (e.g. 500 rollouts) and giving the result to Gemini.
4. `guard.py`: extract numbers from Gemini's answer and check they appear in the context; if not, show a warning under the answer.
5. `fallback.py`: if there is no API key or no internet, answer the standard questions from
   templates filled with the real report values, so the demo always works.
6. Supported questions (shown as quick buttons): bearing condition · why this fault · recommended
   action · why the RL agent chose it · what if we keep running · explain simply · compare with a healthy bearing.

**Done when:** tests run the chat in fallback mode for all 7 question types; with a key, a manual
check confirms Gemini answers use only report values.

### Module 7 — Streamlit dashboard
**Sidebar (all pages):** upload file **or** pick a bundled sample · sampling rate · RPM · load · **Analyze** button.

| Page | Contents |
|---|---|
| **Home** | Project overview · machine status card · health gauge · latest diagnosis · recommended action · simulation/data disclaimer |
| **Signal Analysis** | Raw waveform (zoomable) · FFT spectrum · envelope spectrum with BPFO/BPFI/BSF lines · feature table vs healthy averages |
| **Fault Diagnosis** | Predicted class · probability bars · estimated severity · per-window prediction timeline · model evaluation: LOLO metrics, confusion matrices, feature-group comparison, feature importance |
| **Predictive Maintenance** | Machine state · PPO recommendation + action probabilities · explanation · rule-based comparison · training reward curve · policy comparison · example decisions · "SIMULATION" banner |
| **AI Assistant** | Chat (Gemini or fallback mode indicator) · quick-question buttons · context panel showing exactly what data the AI received |
| **Edge AI** | Model sizes · latency (median/p95) · memory · size-vs-accuracy chart · download buttons for `.onnx` / `.joblib` / `edge_infer.py` |

- Models loaded once (`st.cache_resource`); the current `AnalysisReport` lives in `st.session_state`.
- Pages show a friendly "upload or pick a sample first" message if no analysis has been run.

**Done when:** Streamlit's built-in `AppTest` smoke-tests every page; manual walkthrough of the demo scenario works end-to-end.

### Module 8 — Documentation & deliverables
- `README.md`: what it is, architecture diagram, Windows install steps, how to run each script, how to start the dashboard, how to add the Gemini key, results summary, limitations.
- Architecture diagram: Mermaid version in README + PNG version in `reports/figures/`.
- All evaluation plots and metrics in `reports/`.
- `run_all.py` to rebuild everything from scratch with one command.

---

## Part E — Build order & review checkpoints

| Step | Module | Checkpoint (I report to you) |
|---|---|---|
| 1 | M0 Setup + M1 Data & features | Data downloaded, feature CSV built, sanity plots |
| 2 | M2 Fault RF + M3 Severity & health | **Checkpoint 1:** classification results (LOLO metrics, confusion matrices, feature comparison) |
| 3 | M4 Edge AI | Benchmark numbers |
| 4 | M5 RL | **Checkpoint 2:** RL training curve + comparison vs baselines |
| 5 | M6 Conversational AI | Fallback mode working; Gemini plugged in once you provide the key |
| 6 | M7 Dashboard | **Checkpoint 3:** full demo walkthrough |
| 7 | M8 Docs & final results | Final deliverables |

Every module is tested before the next one starts.

---

## Part F — How the final demo will go

1. Engineer opens the dashboard and picks/uploads a recording (e.g. a CWRU outer race 0.014" recording at 2 HP, held out from training).
2. Signal Analysis shows the waveform and a clear envelope-spectrum peak at the outer race frequency (~105 Hz at 1750 RPM).
3. Fault Diagnosis shows **Outer Race Fault** with its confidence and estimated size.
4. Health shows **Warning/Critical** per the rules.
5. Predictive Maintenance shows the PPO agent's action (e.g. **Schedule inspection** or **Reduce load**) — whatever it actually learned — next to the rule-based action.
6. AI Assistant explains *why*, and answers follow-up questions using only these results.

None of these values are hardcoded — they all come from `analyze()`.

---

## Part G — Limitations (will be stated in README and dashboard)

1. Trained and tested only on **CWRU lab data** (a 2 HP motor test rig), not on a real PA fan. Real fans have different speeds, structures and noise; retraining/validation on fan data would be needed.
2. CWRU faults are **artificially made** (machined pits), not naturally grown damage.
3. The RL agent is trained in a **simulation** with **assumed** costs and wear rates. Its decisions show the method, not validated plant policy.
4. **No real edge hardware** is used; Edge AI numbers are measured on a laptop.
5. Health status is a **rule-based estimate**.
6. Gemini requires internet and an API key; without it the chat uses template answers.

---

## Part H — Risks & how they are handled

| Risk | Handling |
|---|---|
| CWRU website blocks scripted download | Manual download instructions; loader works with any files placed in `data/raw/` |
| PPO doesn't beat the rule-based policy | Tune rewards/training; if still not better, report honestly with analysis |
| LOLO accuracy lower than published "99%" | Expected and fine — it is the honest number; the naive split is shown alongside for comparison |
| Gemini model names/SDK change | Model name in `config.yaml`; fallback mode always available |
| Uploaded file in an unexpected format | Clear error messages; supported formats listed in UI |

---

## Part I — Questions for you to confirm before I start

1. **Outer race position:** only the standard @6 o'clock data (proposed), or also @3 and @12?
2. **Health rules (Module 3):** OK as proposed, or do you want different thresholds?
3. **RL costs:** fine for me to choose reasonable illustrative values (all editable in `config.yaml`)?
4. **Checkpoints:** should I stop and wait for your OK at each checkpoint, or just report and keep going?
5. **Git:** should I initialise a git repository for this project?
