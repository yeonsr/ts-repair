import sys
import pandas as pd
from collections import defaultdict, Counter
import itertools
import numpy as np
from datetime import timedelta

def main():
    if len(sys.argv) < 3:
        print("Usage: python repair.py <input_csv> <output_csv>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    df = pd.read_csv(input_file)
    df['original_timestamp'] = df['Timestamp']
    df['ts'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df['orig_index'] = df.index

    # Build clean reference model from non-corrupted transitions
    transition_counts = defaultdict(Counter)
    duration_stats = defaultdict(list)

    for _, case_group in df.groupby('Case'):
        sorted_case = case_group.sort_values(by=['ts', 'orig_index'])
        sorted_records = sorted_case.to_dict('records')
        for i in range(1, len(sorted_records)):
            prev = sorted_records[i - 1]
            curr = sorted_records[i]
            if prev['ts'] != curr['ts']:
                act1 = prev['Activity']
                act2 = curr['Activity']
                delta = (curr['ts'] - prev['ts']).total_seconds()
                if delta >= 0:
                    transition_counts[act1][act2] += 1
                    duration_stats[(act1, act2)].append(delta)

    # Average durations (median is more robust)
    avg_duration = {}
    for pair, deltas in duration_stats.items():
        if deltas:
            avg_duration[pair] = float(np.median(deltas))

    # Repair only corrupted segments
    repaired_ts = [None] * len(df)

    for _, case_group in df.groupby('Case'):
        sorted_case = case_group.sort_values(by=['ts', 'orig_index'])
        sorted_records = sorted_case.to_dict('records')
        n = len(sorted_records)
        i = 0
        while i < n:
            curr_ts = sorted_records[i]['ts']
            start = i
            while i < n and sorted_records[i]['ts'] == curr_ts:
                i += 1
            end = i
            group_records = sorted_records[start:end]
            activities = [r['Activity'] for r in group_records]
            unique_acts = set(activities)

            if len(group_records) >= 2 and len(unique_acts) > 1:
                left = sorted_records[start - 1] if start > 0 else None
                right = sorted_records[end] if end < n else None

                if left is None or right is None or len(group_records) > 6:
                    # Conservative fallback: do not repair if anchors missing or block too large
                    continue

                # Find best permutation using anchor and internal transitions
                best_score = -1
                best_order = group_records
                for perm in itertools.permutations(range(len(group_records))):
                    ordered_acts = [group_records[p]['Activity'] for p in perm]
                    score = 0
                    if left:
                        score += transition_counts[left['Activity']].get(ordered_acts[0], 0)
                    for k in range(len(ordered_acts) - 1):
                        score += transition_counts[ordered_acts[k]].get(ordered_acts[k + 1], 0)
                    if right:
                        score += transition_counts[ordered_acts[-1]].get(right['Activity'], 0)
                    if score > best_score:
                        best_score = score
                        best_order = [group_records[p] for p in perm]

                # Reconstruct with weighted timestamps
                ordered_acts = [r['Activity'] for r in best_order]
                expected_intervals = []
                prev_act = left['Activity']
                for act in ordered_acts:
                    expected_intervals.append(avg_duration.get((prev_act, act), 30.0))
                    prev_act = act
                expected_intervals.append(avg_duration.get((prev_act, right['Activity']), 30.0))

                total_exp = sum(expected_intervals)
                left_ts = left['ts']
                right_ts = right['ts']
                actual_gap = (right_ts - left_ts).total_seconds()
                if actual_gap <= 0 or total_exp <= 0:
                    actual_gap = max(actual_gap, total_exp, len(group_records) * 60.0)
                    scale = 1.0
                else:
                    scale = actual_gap / total_exp

                curr_ts = left_ts
                for idx in range(len(best_order)):
                    delta = expected_intervals[idx] * scale
                    curr_ts += timedelta(seconds=delta)
                    orig_idx = best_order[idx]['orig_index']
                    repaired_ts[orig_idx] = curr_ts

    # Apply repairs
    has_millis = df['original_timestamp'].str.contains(r'\.\d', na=False).any()
    ts_format = '%Y-%m-%d %H:%M:%S.%f' if has_millis else '%Y-%m-%d %H:%M:%S'

    for i in range(len(df)):
        if repaired_ts[i] is not None:
            df.at[i, 'Timestamp'] = repaired_ts[i].strftime(ts_format)

    # Output only required columns, preserving row order
    output_df = df[['Case', 'Activity', 'Timestamp', 'Resource']]
    output_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    main()