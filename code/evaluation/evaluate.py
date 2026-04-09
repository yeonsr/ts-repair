import json
import math
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


INPUT_COLS = ["Case", "Activity", "Timestamp"]
EVAL_COLS = ["Timestamp_original", "is_polluted"]


def _normalize_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    as_str = series.astype(str).str.strip().str.lower()
    return as_str.isin({"1", "true", "t", "yes", "y"})


def _safe_rate(num: float, den: float) -> Optional[float]:
    if den == 0:
        return None
    return float(num / den)


def _safe_mean_bool(series: pd.Series) -> Optional[float]:
    if len(series) == 0:
        return None
    return float(series.astype(bool).mean())


def _metrics_from_error_array(arr: pd.Series) -> Dict[str, Optional[float]]:
    if len(arr) == 0:
        return {"MAE": None, "MedAE": None, "RMSE": None}
    arr = arr.astype(float)
    return {
        "MAE": float(arr.mean()),
        "MedAE": float(arr.median()),
        "RMSE": float(np.sqrt((arr ** 2).mean())),
    }


def _pairwise_counts(order: List[int]) -> Tuple[int, int]:
    total = 0
    correct = 0
    pos = {rid: i for i, rid in enumerate(order)}
    sorted_ids = sorted(order)

    for i in range(len(sorted_ids)):
        for j in range(i + 1, len(sorted_ids)):
            a = sorted_ids[i]
            b = sorted_ids[j]
            total += 1
            if pos[a] < pos[b]:
                correct += 1

    return correct, total


def _order_metrics_from_block_orders(gt_order: List[int], pred_order: List[int]) -> Dict[str, float]:
    if len(gt_order) != len(pred_order) or len(gt_order) == 0:
        return {
            "valid_blocks": 0,
            "exact_blocks": 0,
            "pairwise_correct": 0,
            "pairwise_total": 0,
            "position_correct": 0,
            "position_total": 0,
        }

    exact = int(gt_order == pred_order)

    gt_pos = {rid: i for i, rid in enumerate(gt_order)}
    pred_pos = {rid: i for i, rid in enumerate(pred_order)}
    position_correct = sum(int(gt_pos[rid] == pred_pos.get(rid, -1)) for rid in gt_order)
    position_total = len(gt_order)

    pairwise_total = 0
    pairwise_correct = 0
    for i in range(len(gt_order)):
        for j in range(i + 1, len(gt_order)):
            a = gt_order[i]
            b = gt_order[j]
            pairwise_total += 1
            if pred_pos.get(a, math.inf) < pred_pos.get(b, math.inf):
                pairwise_correct += 1

    return {
        "valid_blocks": 1,
        "exact_blocks": exact,
        "pairwise_correct": pairwise_correct,
        "pairwise_total": pairwise_total,
        "position_correct": position_correct,
        "position_total": position_total,
    }


def _detect_polluted_blocks(eval_df: pd.DataFrame) -> List[pd.DataFrame]:
    blocks = []
    for _, case_df in eval_df.groupby("Case", sort=False):
        case_df = case_df.sort_values(["_row_id"], kind="stable").reset_index(drop=True)
        i = 0
        while i < len(case_df):
            if not bool(case_df.loc[i, "is_polluted_bool"]):
                i += 1
                continue

            start = i
            while i < len(case_df) and bool(case_df.loc[i, "is_polluted_bool"]):
                i += 1
            end = i - 1

            block = case_df.iloc[start : end + 1].copy()
            if len(block) >= 2:
                blocks.append(block)

    return blocks


def compute_ordering_metrics(eval_df: pd.DataFrame) -> Dict[str, Optional[float]]:
    blocks = _detect_polluted_blocks(eval_df)

    total_blocks = len(blocks)
    baseline_valid_blocks = 0
    baseline_exact_blocks = 0
    baseline_pairwise_correct = 0
    baseline_pairwise_total = 0
    baseline_position_correct = 0
    baseline_position_total = 0

    repaired_valid_blocks = 0
    repaired_exact_blocks = 0
    repaired_pairwise_correct = 0
    repaired_pairwise_total = 0
    repaired_position_correct = 0
    repaired_position_total = 0

    for block in blocks:
        if block["gt_dt"].isna().any():
            continue

        gt_order = block.sort_values(["gt_dt", "_row_id"], kind="stable")["_row_id"].tolist()

        if not block["obs_dt"].isna().any():
            obs_order = block.sort_values(["obs_dt", "_row_id"], kind="stable")["_row_id"].tolist()
            stats = _order_metrics_from_block_orders(gt_order, obs_order)
            baseline_valid_blocks += stats["valid_blocks"]
            baseline_exact_blocks += stats["exact_blocks"]
            baseline_pairwise_correct += stats["pairwise_correct"]
            baseline_pairwise_total += stats["pairwise_total"]
            baseline_position_correct += stats["position_correct"]
            baseline_position_total += stats["position_total"]

        if not block["pred_dt"].isna().any():
            pred_order = block.sort_values(["pred_dt", "_row_id"], kind="stable")["_row_id"].tolist()
            stats = _order_metrics_from_block_orders(gt_order, pred_order)
            repaired_valid_blocks += stats["valid_blocks"]
            repaired_exact_blocks += stats["exact_blocks"]
            repaired_pairwise_correct += stats["pairwise_correct"]
            repaired_pairwise_total += stats["pairwise_total"]
            repaired_position_correct += stats["position_correct"]
            repaired_position_total += stats["position_total"]

    metrics = {
        "polluted_block_count": total_blocks,
        "baseline_valid_order_eval_block_count": baseline_valid_blocks,
        "baseline_block_exact_order_match_rate": _safe_rate(
            baseline_exact_blocks, baseline_valid_blocks
        ),
        "baseline_pairwise_order_accuracy": _safe_rate(
            baseline_pairwise_correct, baseline_pairwise_total
        ),
        "baseline_position_accuracy": _safe_rate(
            baseline_position_correct, baseline_position_total
        ),
        "valid_order_eval_block_count": repaired_valid_blocks,
        "block_exact_order_match_rate": _safe_rate(
            repaired_exact_blocks, repaired_valid_blocks
        ),
        "pairwise_order_accuracy": _safe_rate(
            repaired_pairwise_correct, repaired_pairwise_total
        ),
        "position_accuracy": _safe_rate(
            repaired_position_correct, repaired_position_total
        ),
    }

    if (
        metrics["block_exact_order_match_rate"] is not None
        and metrics["baseline_block_exact_order_match_rate"] is not None
    ):
        metrics["block_exact_order_match_improvement"] = (
            metrics["block_exact_order_match_rate"]
            - metrics["baseline_block_exact_order_match_rate"]
        )
    else:
        metrics["block_exact_order_match_improvement"] = None

    if (
        metrics["pairwise_order_accuracy"] is not None
        and metrics["baseline_pairwise_order_accuracy"] is not None
    ):
        metrics["pairwise_order_accuracy_improvement"] = (
            metrics["pairwise_order_accuracy"]
            - metrics["baseline_pairwise_order_accuracy"]
        )
    else:
        metrics["pairwise_order_accuracy_improvement"] = None

    if (
        metrics["position_accuracy"] is not None
        and metrics["baseline_position_accuracy"] is not None
    ):
        metrics["position_accuracy_improvement"] = (
            metrics["position_accuracy"] - metrics["baseline_position_accuracy"]
        )
    else:
        metrics["position_accuracy_improvement"] = None

    return metrics


def evaluate(original_path: str, repaired_path: str) -> Dict:
    original = pd.read_csv(original_path)
    repaired = pd.read_csv(repaired_path)

    required_original = set(INPUT_COLS + EVAL_COLS)
    missing_original = sorted(required_original - set(original.columns))
    if missing_original:
        raise ValueError(f"Original file is missing required columns: {missing_original}")

    missing_repaired = sorted(set(INPUT_COLS) - set(repaired.columns))
    if missing_repaired:
        raise ValueError(f"Repaired file is missing required columns: {missing_repaired}")

    if len(original) != len(repaired):
        raise ValueError(
            f"Row count mismatch: original={len(original)}, repaired={len(repaired)}"
        )

    eval_df = original.copy().reset_index(drop=True)
    repaired = repaired.reset_index(drop=True)

    eval_df["_row_id"] = np.arange(len(eval_df))
    eval_df["is_polluted_bool"] = _normalize_bool_series(eval_df["is_polluted"])

    eval_df["orig_timestamp_obs"] = eval_df["Timestamp"].astype(str)
    eval_df["pred_timestamp"] = repaired["Timestamp"].astype(str)
    eval_df["gt_timestamp"] = eval_df["Timestamp_original"].astype(str)

    eval_df["obs_dt"] = pd.to_datetime(eval_df["orig_timestamp_obs"], errors="coerce", utc=True)
    eval_df["pred_dt"] = pd.to_datetime(eval_df["pred_timestamp"], errors="coerce", utc=True)
    eval_df["gt_dt"] = pd.to_datetime(eval_df["gt_timestamp"], errors="coerce", utc=True)

    polluted_mask = eval_df["is_polluted_bool"]
    clean_mask = ~polluted_mask

    eval_df["string_exact_match"] = eval_df["pred_timestamp"] == eval_df["gt_timestamp"]
    eval_df["datetime_exact_match"] = (
        eval_df["pred_dt"].notna()
        & eval_df["gt_dt"].notna()
        & (eval_df["pred_dt"] == eval_df["gt_dt"])
    )
    eval_df["changed_from_observed"] = eval_df["pred_timestamp"] != eval_df["orig_timestamp_obs"]

    valid_polluted = polluted_mask & eval_df["obs_dt"].notna() & eval_df["pred_dt"].notna() & eval_df["gt_dt"].notna()
    eval_df["error_before"] = np.nan
    eval_df["error_after"] = np.nan

    eval_df.loc[valid_polluted, "error_before"] = (
        eval_df.loc[valid_polluted, "obs_dt"] - eval_df.loc[valid_polluted, "gt_dt"]
    ).abs().dt.total_seconds()

    eval_df.loc[valid_polluted, "error_after"] = (
        eval_df.loc[valid_polluted, "pred_dt"] - eval_df.loc[valid_polluted, "gt_dt"]
    ).abs().dt.total_seconds()

    before_metrics = _metrics_from_error_array(
        eval_df.loc[polluted_mask, "error_before"].dropna()
    )
    after_metrics = _metrics_from_error_array(
        eval_df.loc[polluted_mask, "error_after"].dropna()
    )

    results = {
        "row_count": int(len(eval_df)),
        "polluted_row_count": int(polluted_mask.sum()),
        "clean_row_count": int(clean_mask.sum()),
        "valid_polluted_datetime_count": int(valid_polluted.sum()),
        "BEFORE": before_metrics,
        "AFTER": after_metrics,
        "IMPROVEMENT": {
            "MAE_reduction": (
                None
                if before_metrics["MAE"] is None or after_metrics["MAE"] is None
                else float(before_metrics["MAE"] - after_metrics["MAE"])
            ),
            "RMSE_reduction": (
                None
                if before_metrics["RMSE"] is None or after_metrics["RMSE"] is None
                else float(before_metrics["RMSE"] - after_metrics["RMSE"])
            ),
        },
        "polluted_string_exact_match_rate": (
            _safe_mean_bool(eval_df.loc[polluted_mask, "string_exact_match"])
            if polluted_mask.any()
            else None
        ),
        "polluted_datetime_exact_match_rate": (
            _safe_mean_bool(eval_df.loc[polluted_mask, "datetime_exact_match"])
            if polluted_mask.any()
            else None
        ),
        "clean_string_exact_match_rate": (
            _safe_mean_bool(eval_df.loc[clean_mask, "string_exact_match"])
            if clean_mask.any()
            else None
        ),
        "clean_datetime_exact_match_rate": (
            _safe_mean_bool(eval_df.loc[clean_mask, "datetime_exact_match"])
            if clean_mask.any()
            else None
        ),
        "overall_string_exact_match_rate": _safe_mean_bool(eval_df["string_exact_match"]),
        "overall_datetime_exact_match_rate": _safe_mean_bool(eval_df["datetime_exact_match"]),
        "clean_changed_rate": (
            _safe_mean_bool(eval_df.loc[clean_mask, "changed_from_observed"])
            if clean_mask.any()
            else None
        ),
        "polluted_changed_rate": (
            _safe_mean_bool(eval_df.loc[polluted_mask, "changed_from_observed"])
            if polluted_mask.any()
            else None
        ),
        "overall_changed_rate": _safe_mean_bool(eval_df["changed_from_observed"]),
    }

    tmp = repaired.copy()
    tmp["Timestamp_dt"] = pd.to_datetime(tmp["Timestamp"], errors="coerce", utc=True)

    case_ok = []
    for _, g in tmp.groupby("Case", sort=False):
        if g["Timestamp_dt"].isna().any():
            case_ok.append(False)
            continue
        diffs = g["Timestamp_dt"].diff().dropna()
        case_ok.append(bool((diffs > pd.Timedelta(0)).all()))

    results["strict_monotonic_case_rate"] = (
        float(sum(case_ok) / len(case_ok)) if case_ok else None
    )

    results["ORDERING"] = compute_ordering_metrics(eval_df)
    return results


def main():
    if len(sys.argv) < 3:
        print("Usage: python evaluate_repair.py <original_csv> <repaired_csv>")
        sys.exit(1)

    original_path = sys.argv[1]
    repaired_path = sys.argv[2]

    results = evaluate(original_path, repaired_path)
    if len(sys.argv) >= 4:
        output_path = sys.argv[3]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    else:
        print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()