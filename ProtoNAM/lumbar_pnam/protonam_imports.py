from __future__ import annotations

import sys
from pathlib import Path


def ensure_protonam_src_on_path() -> Path:
    """Make ProtoNAM/src importable and return the resolved path."""
    root = Path(__file__).resolve().parents[1]
    proto_src = root / "ProtoNAM" / "src"
    if not proto_src.exists():
        raise FileNotFoundError(f"ProtoNAM src directory not found: {proto_src}")
    proto_src_str = str(proto_src)
    if proto_src_str not in sys.path:
        sys.path.insert(0, proto_src_str)
    return proto_src


# NOTE: call ensure_protonam_src_on_path() before importing these.

