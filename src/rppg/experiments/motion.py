"""PURE six-condition evaluation from actual predicted session HR values."""
from collections import defaultdict
import numpy as np
from .robustness import condition_from_session, PURE_CONDITIONS


def pure_motion_table(sessions):
    by_condition = defaultdict(list)
    steady = {}
    for row in sessions:
        condition = condition_from_session(row["session"])
        by_condition[condition].append(row)
        if condition == "steady":
            steady[row["subject"]] = abs(row["pred_hr"] - row["gt_hr"])
    result = []
    for condition in PURE_CONDITIONS.values():
        data = by_condition.get(condition, [])
        if not data:
            continue
        errs = np.array([r["pred_hr"] - r["gt_hr"] for r in data])
        paired = [abs(r["pred_hr"] - r["gt_hr"]) - steady[r["subject"]]
                  for r in data if r["subject"] in steady]
        result.append({"condition": condition, "n_sessions": len(data),
                       "mae_bpm": float(np.abs(errs).mean()),
                       "rmse_bpm": float(np.sqrt(np.mean(errs ** 2))),
                       "paired_delta_ae_bpm": float(np.mean(paired)) if paired else None})
    return result
