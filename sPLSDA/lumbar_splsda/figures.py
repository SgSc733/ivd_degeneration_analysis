from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from lumbar_splsda.figures_eval import generate_eval_figures
from lumbar_splsda.figures_explain import generate_explain_figures


def generate_all_figures(
    *,
    run_dir: str | Path,
    X_raw: pd.DataFrame,
    meta: pd.DataFrame | None,
    tasks: Any,
    task_names: list[str],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """
    Generate post-training figures for a run directory.

    Key principle (aligned with the ProtoNAM main plan): all prediction-based figures use the
    calibrated discrete grade.
    """
    run_dir = Path(run_dir)
    out: dict[str, Any] = {}
    out["eval"] = generate_eval_figures(run_dir=run_dir)
    out["explain"] = generate_explain_figures(
        run_dir=run_dir,
        X_raw=X_raw,
        meta=meta,
        tasks=tasks,
        task_names=task_names,
        cfg=cfg,
    )
    return out

