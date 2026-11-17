"""Matplotlib report figures (PNG files for reports/figures). No UI code."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def confusion_plot(cm, labels, title: str, out: Path) -> None:
    cm = np.asarray(cm, dtype=float)
    rows = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm, rows, out=np.zeros_like(cm), where=rows > 0)
    fig, ax = plt.subplots(figsize=(1.4 * len(labels) + 2.5, 1.2 * len(labels) + 1.8))
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{pct[i, j]:.0%}\n({int(cm[i, j])})", ha="center", va="center",
                    color="white" if pct[i, j] > 0.6 else "black", fontsize=9)
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def bar_plot(values: dict[str, float], errors: dict[str, float] | None, title: str, ylabel: str, out: Path,
             ylim=(0, 1.05)) -> None:
    fig, ax = plt.subplots(figsize=(1.3 * len(values) + 2, 4))
    names = list(values)
    ax.bar(names, [values[n] for n in names], yerr=[errors[n] for n in names] if errors else None,
           capsize=4, color="#457b9d")
    for i, n in enumerate(names):
        ax.text(i, values[n] + 0.01, f"{values[n]:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(*ylim); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def importance_plot(names, importances, out: Path, top: int = 20) -> None:
    order = np.argsort(importances)[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(order) + 1.2))
    ax.barh([names[i] for i in order], np.asarray(importances)[order], color="#e76f51")
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_title(f"Fault RF - top {len(order)} feature importances", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
