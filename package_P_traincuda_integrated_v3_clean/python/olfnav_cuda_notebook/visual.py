from __future__ import annotations

from typing import Any
import json


def show_cuda_training_report(result: Any, *, display_rows: int = 10) -> None:
    """Render a compact traincuda report in notebooks or terminals."""
    summary = getattr(result, "summary", None)
    rows = getattr(result, "rows", None)
    if summary is not None:
        print("CUDA traincuda summary:")
        print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    if rows is None:
        return
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        if len(df):
            try:
                from IPython.display import display
                display(df.tail(max(1, int(display_rows))))
            except Exception:
                print(df.tail(max(1, int(display_rows))).to_string(index=False))
    except Exception as exc:
        print(f"[WARN] could not render CUDA rows: {type(exc).__name__}: {exc}")
