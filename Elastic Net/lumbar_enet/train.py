"""Training entrypoint (public API).

The active implementation lives in `lumbar_enet.train_path`:
- Outer GroupKFold (patient-level)
- Inner GroupKFold lambda path selection (lambda_min / lambda_1se)
- Exact penalty factor objective with unpenalized gamma (v(γ)=0 by default)

This module only re-exports the symbols to keep the stable import path:
    from lumbar_enet.train import train_group_kfold

The previous sklearn GridSearchCV prototype is kept in
`lumbar_enet.train_sklearn_legacy` for historical reference.
"""
from __future__ import annotations

from lumbar_enet.train_path import CVResult, FoldResult, train_group_kfold

__all__ = [
    "CVResult",
    "FoldResult",
    "train_group_kfold",
]
