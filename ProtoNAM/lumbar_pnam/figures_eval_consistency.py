from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def lin_ccc(*, y_true: np.ndarray, y_cont: np.ndarray) -> float:
    """Lin's Concordance Correlation Coefficient (CCC) between u=y_true and v=y_cont."""
    u = np.asarray(y_true, dtype=float)
    v = np.asarray(y_cont, dtype=float)
    if u.size == 0:
        return 0.0

    mu_u = float(u.mean())
    mu_v = float(v.mean())
    var_u = float(u.var())
    var_v = float(v.var())
    std_u = float(np.sqrt(var_u))
    std_v = float(np.sqrt(var_v))

    if std_u <= 0.0 or std_v <= 0.0:
        rho = 0.0
    else:
        rho_val = float(np.corrcoef(u, v)[0, 1])
        rho = rho_val if np.isfinite(rho_val) else 0.0

    denom = var_u + var_v + (mu_u - mu_v) ** 2
    if denom <= 0.0 or not np.isfinite(denom):
        return 0.0

    ccc_val = (2.0 * rho * std_u * std_v) / denom
    return float(ccc_val) if np.isfinite(ccc_val) else 0.0


def save_bland_altman(
    *,
    y_true: np.ndarray,
    y_cont: np.ndarray,
    out_path: Path,
    title: str,
) -> dict[str, float]:
    """Bland–Altman plot: diff vs mean, with 95% limits of agreement."""
    y_true_f = np.asarray(y_true, dtype=float)
    y_cont_f = np.asarray(y_cont, dtype=float)
    mean = 0.5 * (y_true_f + y_cont_f)
    diff = y_cont_f - y_true_f

    bias = float(np.mean(diff)) if diff.size else 0.0
    sd = float(np.std(diff, ddof=1)) if diff.size >= 2 else 0.0
    loa_low = bias - 1.96 * sd
    loa_high = bias + 1.96 * sd
    ccc = lin_ccc(y_true=y_true_f, y_cont=y_cont_f)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(mean, diff, s=14, alpha=0.35, color="tab:blue", edgecolors="none")
    ax.axhline(bias, color="tab:red", linestyle="-", linewidth=1.5, label=f"bias={bias:.3f}")
    ax.axhline(loa_low, color="tab:red", linestyle="--", linewidth=1.2, label=f"LoA low={loa_low:.3f}")
    ax.axhline(loa_high, color="tab:red", linestyle="--", linewidth=1.2, label=f"LoA high={loa_high:.3f}")
    ax.set_xlabel("Mean of (y_true, y_cont)")
    ax.set_ylabel("Difference (y_cont - y_true)")
    ax.set_title(title)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.legend(loc="best", frameon=True, fontsize=8)
    ax.text(
        0.02,
        0.98,
        f"n={int(diff.size)}\nCCC={ccc:.4f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, linewidth=0.5),
    )
    _ensure_dir(out_path.parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    return {"bias": bias, "loa_low": loa_low, "loa_high": loa_high, "ccc": ccc, "n": float(diff.size)}

