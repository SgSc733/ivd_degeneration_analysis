"""EBM baseline pipeline for lumbar disc degeneration quantification.

This package mirrors the structure of `ProtoNAM/lumbar_pnam` but uses:
- InterpretML ExplainableBoostingRegressor (EBM) as Stage-1
- A CORAL-style ordinal calibrator as Stage-2
"""

from .data import LoadedTabular, attach_pfirrmann_from_xlsx, load_model_input_csv

__all__ = [
    "LoadedTabular",
    "attach_pfirrmann_from_xlsx",
    "load_model_input_csv",
]
