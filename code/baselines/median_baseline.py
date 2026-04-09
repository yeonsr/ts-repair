import sys
import math
import itertools
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["Case", "Activity", "Timestamp"]
FALLBACK_SECONDS = 60.0
BRUTE_FORCE_MAX_LEN = 8
BEAM_SIZE = 10
MIN_DURATION_SECONDS = 1.0


def validate_input_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def parse_timestamp_series(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce", utc=True)
    if ts.isna().any():
        bad = series[ts.isna()].head(5).tolist()
        raise ValueError(f"Failed to parse Timestamp values. Examples: {bad}")
    return ts


def detect_same_timestamp_blocks(case_df: pd.DataFrame) -> List[Tuple[int, int]]:
    """
    Detect contiguous same-timestamp blocks within a case.
    A candidate block is repairable when:
      - size >= 2
      - activities are not all identical
    Returns local index ranges over case_df after sorting by (ts, _orig_order).
    """
    blocks = []
    i = 0
    n = len(case_df)

    while i < n:
        curr_ts = case_df.iloc[i]["_ts"]
        start = i
        while i < n and case_df.iloc[i]["_ts"] == curr_ts:
            i += 1
        end = i - 1

        if end - start + 1 >= 2:
            acts = case_df.iloc[start : end + 1]["Activity"].astype(str).tolist()
            if len(set(acts)) > 1:
                blocks.append((start, end))

    return blocks


def compute_activity_pair_median(df: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    """
    Learn pairwise median durations from non-collapsed transitions only.
    """
    rows = []

    for _, g in df.groupby("Case", sort=False):
        g = g.sort_values(["_ts", "_orig_order"], kind="stable").reset_index(drop=True)
        for i in range(len(g) - 1):
            ts1 = g.loc[i, "_ts"]
            ts2 = g.loc[i + 1, "_ts"]
            if ts1 == ts2:
                continue

            dt = (ts2 - ts1).total_seconds()
            if dt <= 0:
                continue

            act = str(g.loc[i, "Activity"])
            nxt = str(g.loc[i + 1, "Activity"])
            rows.append((act, nxt, float(dt)))

    if not rows:
        return {}

    pair_df = pd.DataFrame(rows, columns=["act", "next", "duration"])
    return pair_df.groupby(["act", "next"])["duration"].median().to_dict()


def train_transition_probability_model(
    df: pd.DataFrame,
    smoothing: float = 1.0,
) -> Dict[str, object]:
    """
    Learn transition probabilities from non-collapsed transitions only.
    """
    trans_counts = defaultdict(Counter)
    out_counts = Counter()
    activities = sorted(df["Activity"].astype(str).unique().tolist())

    for _, g in df.groupby("Case", sort=False):
        g = g.sort_values(["_ts", "_orig_order"], kind="stable").reset_index(drop=True)
        for i in range(len(g) - 1):
            ts1 = g.loc[i, "_ts"]
            ts2 = g.loc[i + 1, "_ts"]
            if ts1 == ts2:
                continue

            a = str(g.loc[i, "Activity"])
            b = str(g.loc[i + 1, "Activity"])
            trans_counts[a][b] += 1
            out_counts[a] += 1

    vocab_size = max(len(activities), 1)
    uniform_logp = -math.log(vocab_size)
    log_trans = {}

    for a in activities:
        total = out_counts[a] + smoothing * vocab_size
        for b in activities:
            c = trans_counts[a][b]
            p = (c + smoothing) / total
            log_trans[(a, b)] = math.log(p)

    return {
        "activities": activities,
        "log_trans": log_trans,
        "uniform_logp": uniform_logp,
    }


def score_sequence_with_boundaries(
    order: Sequence[str],
    prev_act: Optional[str],
    next_act: Optional[str],
    model: Dict[str, object],
) -> float:
    log_trans = model["log_trans"]
    uniform_logp = model["uniform_logp"]

    seq = []
    if prev_act is not None:
        seq.append(str(prev_act))
    seq.extend([str(a) for a in order])
    if next_act is not None:
        seq.append(str(next_act))

    score = 0.0
    for a, b in zip(seq[:-1], seq[1:]):
        score += log_trans.get((a, b), uniform_logp)
    return score


def reorder_block_bruteforce(
    block_activities: Sequence[str],
    prev_act: Optional[str],
    next_act: Optional[str],
    model: Dict[str, object],
) -> List[str]:
    best_order = list(block_activities)
    best_score = float("-inf")

    for perm in itertools.permutations(block_activities):
        score = score_sequence_with_boundaries(perm, prev_act, next_act, model)
        if score > best_score:
            best_score = score
            best_order = list(perm)

    return best_order


def reorder_block_beam(
    block_activities: Sequence[str],
    prev_act: Optional[str],
    next_act: Optional[str],
    model: Dict[str, object],
    beam_size: int = 10,
) -> List[str]:
    """
    Deterministic beam search for larger blocks.
    """
    log_trans = model["log_trans"]
    uniform_logp = model["uniform_logp"]

    states = [([], list(block_activities), 0.0)]

    while True:
        expanded = []
        done = True

        for prefix, remaining, score in states:
            if not remaining:
                expanded.append((prefix, remaining, score))
                continue

            done = False
            prev = prev_act if not prefix else prefix[-1]

            for i, act in enumerate(remaining):
                trans_score = 0.0
                if prev is not None:
                    trans_score = log_trans.get((str(prev), str(act)), uniform_logp)
                new_prefix = prefix + [act]
                new_remaining = remaining[:i] + remaining[i + 1 :]
                expanded.append((new_prefix, new_remaining, score + trans_score))

        if done:
            break

        expanded.sort(key=lambda x: x[2], reverse=True)
        states = expanded[:beam_size]

    rescored = []
    for prefix, _, score in states:
        final_score = score
        if prefix and next_act is not None:
            final_score += log_trans.get((str(prefix[-1]), str(next_act)), uniform_logp)
        rescored.append((prefix, final_score))

    rescored.sort(key=lambda x: x[1], reverse=True)
    return rescored[0][0] if rescored else list(block_activities)


def reorder_block(
    block_activities: Sequence[str],
    prev_act: Optional[str],
    next_act: Optional[str],
    model: Dict[str, object],
    brute_force_max_len: int = BRUTE_FORCE_MAX_LEN,
    beam_size: int = BEAM_SIZE,
) -> List[str]:
    if len(block_activities) <= 1:
        return list(block_activities)

    if len(block_activities) <= brute_force_max_len:
        return reorder_block_bruteforce(block_activities, prev_act, next_act, model)

    return reorder_block_beam(block_activities, prev_act, next_act, model, beam_size=beam_size)


def _get_pair_duration(
    pair_median: Dict[Tuple[str, str], float],
    a: str,
    b: str,
    fallback_seconds: float,
) -> float:
    d = float(pair_median.get((str(a), str(b)), fallback_seconds))
    return max(d, MIN_DURATION_SECONDS)


def assign_block_timestamps(
    block_acts: Sequence[str],
    left_act: Optional[str],
    left_ts: Optional[pd.Timestamp],
    right_act: Optional[str],
    right_ts: Optional[pd.Timestamp],
    pair_median: Dict[Tuple[str, str], float],
    fallback_seconds: float = FALLBACK_SECONDS,
) -> Optional[List[pd.Timestamp]]:
    """
    Reconstruct timestamps sequentially from learned pairwise medians.

    Case 1: both anchors exist
    - Use bridge durations [left->a1, a1->a2, ..., ak->right]
    - Scale only internal block durations to fit within the available window

    Case 2: only left anchor exists
    - Forward cumulative assignment

    Case 3: only right anchor exists
    - Backward cumulative assignment

    Case 4: no anchors
    - Conservative skip
    """
    k = len(block_acts)
    if k == 0:
        return []

    if left_ts is not None and right_ts is not None:
        window = (right_ts - left_ts).total_seconds()
        if window <= 0:
            return None

        bridge = []
        prev = left_act
        for act in block_acts:
            if prev is None:
                bridge.append(fallback_seconds)
            else:
                bridge.append(_get_pair_duration(pair_median, prev, act, fallback_seconds))
            prev = act

        if right_act is not None:
            tail = _get_pair_duration(pair_median, block_acts[-1], right_act, fallback_seconds)
        else:
            tail = fallback_seconds

        total_pred = sum(bridge) + tail
        if total_pred <= 0:
            return None

        scale = min(1.0, window / total_pred)
        bridge = [max(d * scale, MIN_DURATION_SECONDS) for d in bridge]

        current = left_ts
        repaired = []
        for d in bridge:
            current = current + pd.to_timedelta(d, unit="s")
            repaired.append(current)

        if repaired[-1] >= right_ts:
            repaired[-1] = right_ts - pd.to_timedelta(MIN_DURATION_SECONDS, unit="s")
            for j in range(len(repaired) - 2, -1, -1):
                repaired[j] = min(
                    repaired[j],
                    repaired[j + 1] - pd.to_timedelta(MIN_DURATION_SECONDS, unit="s"),
                )
            if repaired[0] <= left_ts:
                return None

        return repaired

    if left_ts is not None:
        repaired = []
        current = left_ts
        prev = left_act
        for act in block_acts:
            d = (
                _get_pair_duration(pair_median, prev, act, fallback_seconds)
                if prev is not None
                else fallback_seconds
            )
            current = current + pd.to_timedelta(d, unit="s")
            repaired.append(current)
            prev = act
        return repaired

    if right_ts is not None:
        repaired_rev = []
        current = right_ts
        next_act_local = right_act
        for act in reversed(block_acts):
            d = (
                _get_pair_duration(pair_median, act, next_act_local, fallback_seconds)
                if next_act_local is not None
                else fallback_seconds
            )
            current = current - pd.to_timedelta(d, unit="s")
            repaired_rev.append(current)
            next_act_local = act

        repaired = list(reversed(repaired_rev))
        if len(repaired) >= 2:
            for i in range(1, len(repaired)):
                if repaired[i] <= repaired[i - 1]:
                    repaired[i] = repaired[i - 1] + pd.to_timedelta(MIN_DURATION_SECONDS, unit="s")
        if repaired and repaired[-1] >= right_ts:
            return None
        return repaired

    return None


def repair_log(
    df: pd.DataFrame,
    pair_median: Dict[Tuple[str, str], float],
    trans_model: Dict[str, object],
    fallback_seconds: float = FALLBACK_SECONDS,
) -> pd.DataFrame:
    out = df.copy()

    for case_id, g in out.groupby("Case", sort=False):
        g = g.sort_values(["_ts", "_orig_order"], kind="stable").reset_index()
        local_blocks = detect_same_timestamp_blocks(g)

        for start, end in local_blocks:
            block_global_idxs = g.loc[start:end, "index"].tolist()
            block_acts_observed = g.loc[start:end, "Activity"].astype(str).tolist()

            left_row = g.iloc[start - 1] if start - 1 >= 0 else None
            right_row = g.iloc[end + 1] if end + 1 < len(g) else None

            left_act = None if left_row is None else str(left_row["Activity"])
            left_ts = None if left_row is None else left_row["_ts"]

            right_act = None if right_row is None else str(right_row["Activity"])
            right_ts = None if right_row is None else right_row["_ts"]

            reordered_acts = reorder_block(
                block_activities=block_acts_observed,
                prev_act=left_act,
                next_act=right_act,
                model=trans_model,
            )

            repaired_ts = assign_block_timestamps(
                block_acts=reordered_acts,
                left_act=left_act,
                left_ts=left_ts,
                right_act=right_act,
                right_ts=right_ts,
                pair_median=pair_median,
                fallback_seconds=fallback_seconds,
            )

            if repaired_ts is None:
                continue

            for idx, ts in zip(block_global_idxs, repaired_ts):
                out.loc[idx, "_repaired_ts"] = ts

    out["_final_ts"] = out["_repaired_ts"].where(out["_repaired_ts"].notna(), out["_ts"])
    out["Timestamp"] = out["_final_ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return out[df.columns]


def main():
    if len(sys.argv) < 3:
        print("Usage: python median_baseline.py <input_csv> <output_csv>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    df = pd.read_csv(input_path)
    validate_input_columns(df)

    original_columns = df.columns.tolist()
    work = df.copy()
    work["_orig_order"] = np.arange(len(work))
    work["_ts"] = parse_timestamp_series(work["Timestamp"])
    work["_repaired_ts"] = pd.NaT

    pair_median = compute_activity_pair_median(work)
    trans_model = train_transition_probability_model(work)

    repaired = repair_log(work, pair_median=pair_median, trans_model=trans_model)
    repaired = repaired[original_columns]
    repaired.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()