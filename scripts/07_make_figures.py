"""Step 7: architecture diagram (PNG) for the README / reports. (All evaluation plots are produced by
scripts 02-06 next to the numbers they show.)

Output: reports/figures/architecture.png
"""
import _bootstrap  # noqa: F401

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from pdm.config import path

BOX = dict(boxstyle="round,pad=0.02,rounding_size=0.015", linewidth=1.2)


def box(ax, x, y, w, h, text, fc, ec="#333333", fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, facecolor=fc, edgecolor=ec, **BOX))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x1, y1, x2, y2, style="-|>", color="#333333", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=12, color=color,
                                 linewidth=1.2, linestyle=ls))


def main() -> None:
    fig, ax = plt.subplots(figsize=(15, 8.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.25, 0.97, "PHASE 1 — training (offline scripts, run once)", ha="center", fontsize=12, fontweight="bold")
    ax.text(0.75, 0.97, "PHASE 2 — runtime (every Analyze click, < 1 s)", ha="center", fontsize=12,
            fontweight="bold")
    ax.axvline(0.5, color="#bbbbbb", lw=1, ls="--")

    data_c, ml_c, rl_c, edge_c, llm_c, ui_c = "#e3f2fd", "#e8f5e9", "#fff3e0", "#f3e5f5", "#fce4ec", "#eceff1"
    # phase 1
    box(ax, 0.03, 0.83, 0.2, 0.08, "CWRU .mat files (48)\n12 kHz faults, 48 kHz normal", data_c)
    box(ax, 0.27, 0.83, 0.2, 0.08, "Resample to 12 kHz → windows\n(2048 samples, 50% overlap)", data_c)
    box(ax, 0.27, 0.70, 0.2, 0.08, "Features: time · FFT · envelope\n→ features.csv (5,577 × 31)", data_c)
    box(ax, 0.03, 0.55, 0.2, 0.1, "Fault RF + Severity RF\n(scikit-learn)\nLOLO / unseen-bearing evaluation", ml_c)
    box(ax, 0.27, 0.55, 0.2, 0.1, "Edge AI: size study,\nONNX export + parity,\nlatency / memory benchmark", edge_c)
    box(ax, 0.03, 0.36, 0.2, 0.12, "Out-of-fold predictions on\nUNSEEN bearings\n(realistic misdiagnoses)", ml_c)
    box(ax, 0.27, 0.36, 0.2, 0.12, "Gymnasium simulator\nBearingMaintenanceEnv\n1 step = 1 day, 180 days\n"
        "(SIMULATED costs)", rl_c)
    box(ax, 0.27, 0.19, 0.2, 0.1, "PPO (Stable-Baselines3)\n3 seeds, 500k steps\nvs rule / periodic / RTF / random",
        rl_c)
    box(ax, 0.03, 0.04, 0.44, 0.09, "models/  fault_rf · severity_rf · *_edge.onnx · ppo_agent.zip      "
        "reports/  metrics + figures", "#ffffff", bold=True, fs=7.5)
    arrow(ax, 0.23, 0.87, 0.27, 0.87); arrow(ax, 0.37, 0.83, 0.37, 0.78)
    arrow(ax, 0.30, 0.70, 0.18, 0.65); arrow(ax, 0.40, 0.70, 0.38, 0.65)
    arrow(ax, 0.13, 0.55, 0.13, 0.48); arrow(ax, 0.23, 0.42, 0.27, 0.42); arrow(ax, 0.37, 0.36, 0.37, 0.29)
    arrow(ax, 0.13, 0.36, 0.13, 0.13); arrow(ax, 0.37, 0.19, 0.37, 0.13);
    # phase 2
    box(ax, 0.53, 0.83, 0.2, 0.08, "Upload (.mat/.csv/.npy) or sample\n+ sampling rate, RPM, load", ui_c)
    box(ax, 0.56, 0.42, 0.38, 0.36,
        "analyze(signal, fs, rpm, load)   — pdm/pipeline.py\n\n"
        "• validate input, resample, window, features\n"
        "• Fault RF → class + probabilities (avg. over windows)\n"
        "• Severity RF → estimated fault size\n"
        "• Health rules → Healthy / Warning / Critical (rule-based)\n"
        "• PPO agent → action + probabilities (+ rule-based action)\n"
        "• Simulator → 'what if we keep running' risk (SIMULATED)\n"
        "• training-recording check, caveats, warnings",
        "#fffde7", fs=9)
    box(ax, 0.62, 0.30, 0.26, 0.07, "AnalysisReport (one object)", "#ffffff", bold=True, fs=10)
    box(ax, 0.53, 0.06, 0.2, 0.16, "Streamlit dashboard\n(5 pages, Plotly)\ndisplays the report only", ui_c)
    box(ax, 0.77, 0.06, 0.2, 0.16, "AI Assistant\nGemini (or offline templates)\nreceives report JSON only\n"
        "+ number guard", llm_c)
    box(ax, 0.77, 0.83, 0.2, 0.08, "edge/edge_infer.py\nnumpy + scipy + onnxruntime only", edge_c)
    arrow(ax, 0.63, 0.83, 0.70, 0.78); arrow(ax, 0.75, 0.42, 0.75, 0.37)
    arrow(ax, 0.68, 0.30, 0.63, 0.22); arrow(ax, 0.82, 0.30, 0.87, 0.22)
    arrow(ax, 0.47, 0.085, 0.56, 0.42, color="#888888", ls="--")
    ax.text(0.5, 0.005, "Data: CWRU lab rig (not a PA fan) · RL costs & failure rates are simulated · health is a "
            "rule-based estimate · edge numbers measured on a laptop", ha="center", fontsize=8.5, style="italic")
    out = path("figures_dir") / "architecture.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
