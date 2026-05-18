from __future__ import annotations

from typing import Any


def _display(obj: Any) -> None:
    try:
        from IPython.display import display
        display(obj)
    except Exception:
        print(obj)


def show_cuda_training_report(result: Any, display_rows: int = 10):
    """Display a compact notebook report for a traincuda result."""
    try:
        import pandas as pd
    except Exception:
        print(getattr(result, "summary", result))
        return None

    summary = getattr(result, "summary", {}) or {}
    rows = getattr(result, "rows", []) or []
    print("CUDA traincuda summary")
    for key in [
        "completed", "requested_expansions", "completed_expansions",
        "cuda_version_requested", "total_wall_s", "sum_expand_ms",
        "sum_backup_ms", "sum_update_ms", "last_alpha_after",
        "last_belief_total_after",
    ]:
        if key in summary:
            print(f"  {key}: {summary[key]}")
    df = pd.DataFrame(rows)
    if len(df):
        keep = [c for c in [
            "iter", "nB", "nG_in", "actual_version", "expand_ms",
            "backup_ms", "backup_wall_ms", "update_ms", "iter_total_ms",
            "alpha_after", "belief_total_after", "pruned",
        ] if c in df.columns]
        _display(df[keep].tail(int(display_rows)))
    return df
