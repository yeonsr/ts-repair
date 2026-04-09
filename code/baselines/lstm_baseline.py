import sys
import math
import itertools
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Concatenate, Dense, Embedding, Input, LSTM, Reshape
from tensorflow.keras.losses import Huber
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences


REQUIRED_COLUMNS = ["Case", "Activity", "Timestamp"]

MAX_PREFIX_LEN = 20
MAX_SYN_BLOCK_LEN = 4
EMBED_DIM = 32
LSTM_UNITS = 64
DENSE_UNITS = 64
BATCH_SIZE = 128
EPOCHS = 10
RANDOM_STATE = 42

FALLBACK_SECONDS = 60.0
MIN_DURATION_SECONDS = 1.0
SMOOTHING = 1.0
BRUTE_FORCE_MAX_LEN = 8
BEAM_SIZE = 10


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


def safe_log1p(x: float) -> float:
    return float(np.log1p(max(float(x), 0.0)))


def detect_same_timestamp_blocks(case_df: pd.DataFrame) -> List[Tuple[int, int]]:
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


def compute_pair_median(df: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    rows = []
    for _, g in df.groupby("Case", sort=False):
        g = g.sort_values(["_ts", "_orig_order"], kind="stable").reset_index(drop=True)
        for i in range(len(g) - 1):
            if g.loc[i, "_ts"] == g.loc[i + 1, "_ts"]:
                continue
            a = str(g.loc[i, "Activity"])
            b = str(g.loc[i + 1, "Activity"])
            dt = (g.loc[i + 1, "_ts"] - g.loc[i, "_ts"]).total_seconds()
            if dt > 0:
                rows.append((a, b, float(dt)))

    if not rows:
        return {}
    pair_df = pd.DataFrame(rows, columns=["act", "next", "duration"])
    return pair_df.groupby(["act", "next"])["duration"].median().to_dict()


def train_transition_probability_model(
    df: pd.DataFrame,
    smoothing: float = SMOOTHING,
) -> Dict[str, object]:
    activities = sorted(df["Activity"].astype(str).unique().tolist())
    trans_counts = defaultdict(Counter)
    out_counts = Counter()

    for _, g in df.groupby("Case", sort=False):
        g = g.sort_values(["_ts", "_orig_order"], kind="stable").reset_index(drop=True)
        for i in range(len(g) - 1):
            if g.loc[i, "_ts"] == g.loc[i + 1, "_ts"]:
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
    beam_size: int = BEAM_SIZE,
) -> List[str]:
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
                expanded.append(
                    (prefix + [act], remaining[:i] + remaining[i + 1 :], score + trans_score)
                )

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


def build_activity_vocab(df: pd.DataFrame) -> Dict[str, int]:
    acts = sorted(df["Activity"].astype(str).unique().tolist())
    return {a: i + 1 for i, a in enumerate(acts)}  # 0 is padding/unknown


def build_clean_training_sequences(
    df: pd.DataFrame,
) -> List[Dict[str, object]]:
    """
    Build clean sequences by removing same-timestamp ambiguity.
    Only non-collapsed transitions are used for training instances.
    """
    sequences = []

    for _, g in df.groupby("Case", sort=False):
        g = g.sort_values(["_ts", "_orig_order"], kind="stable").reset_index(drop=True)

        clean_rows = []
        for i in range(len(g)):
            row = g.iloc[i]
            clean_rows.append(
                {
                    "Activity": str(row["Activity"]),
                    "ts": row["_ts"],
                }
            )

        if len(clean_rows) >= 2:
            sequences.append(clean_rows)

    return sequences


def make_training_rows(
    clean_sequences: List[List[Dict[str, object]]],
    pair_median: Dict[Tuple[str, str], float],
    trans_model: Dict[str, object],
    act2id: Dict[str, int],
) -> List[Dict[str, object]]:
    rows = []

    for seq in clean_sequences:
        acts = [r["Activity"] for r in seq]
        tss = [r["ts"] for r in seq]

        deltas = [0.0]
        for i in range(1, len(seq)):
            dt = (tss[i] - tss[i - 1]).total_seconds()
            deltas.append(max(float(dt), MIN_DURATION_SECONDS))

        for i in range(1, len(seq)):
            prev_clean_act = acts[i - 1] if i - 1 >= 0 else None
            next_clean_act = acts[i + 1] if i + 1 < len(seq) else None
            source_act = acts[i - 1]
            target_act = acts[i]

            prefix_acts = acts[:i]
            prefix_deltas = deltas[:i]

            pair_prior = pair_median.get((source_act, target_act), FALLBACK_SECONDS)
            trans_logp = trans_model["log_trans"].get(
                (source_act, target_act),
                trans_model["uniform_logp"],
            )

            rows.append(
                {
                    "prefix_acts": [act2id.get(a, 0) for a in prefix_acts][-MAX_PREFIX_LEN:],
                    "prefix_deltas": [safe_log1p(x) for x in prefix_deltas][-MAX_PREFIX_LEN:],
                    "prev_clean": act2id.get(prev_clean_act, 0) if prev_clean_act else 0,
                    "next_clean": act2id.get(next_clean_act, 0) if next_clean_act else 0,
                    "source_act": act2id.get(source_act, 0),
                    "target_act": act2id.get(target_act, 0),
                    "num_features": [
                        safe_log1p(pair_prior),
                        float(trans_logp),
                        0.0,
                        0.0,
                        1.0 / MAX_SYN_BLOCK_LEN,
                    ],
                    "target_log_delta": safe_log1p(deltas[i]),
                }
            )

    return rows


def prepare_training_arrays(
    train_rows: List[Dict[str, object]],
):
    if not train_rows:
        raise ValueError("No training rows were generated for LSTM baseline.")

    Xp = pad_sequences(
        [r["prefix_acts"] for r in train_rows],
        maxlen=MAX_PREFIX_LEN,
        padding="pre",
        truncating="pre",
        value=0,
    ).astype("int32")

    Xd = pad_sequences(
        [r["prefix_deltas"] for r in train_rows],
        maxlen=MAX_PREFIX_LEN,
        padding="pre",
        truncating="pre",
        value=0.0,
    ).astype("float32")
    Xd = np.expand_dims(Xd, axis=-1)

    Xpc = np.asarray([[r["prev_clean"]] for r in train_rows], dtype="int32")
    Xnc = np.asarray([[r["next_clean"]] for r in train_rows], dtype="int32")
    Xsa = np.asarray([[r["source_act"]] for r in train_rows], dtype="int32")
    Xta = np.asarray([[r["target_act"]] for r in train_rows], dtype="int32")
    Xn = np.asarray([r["num_features"] for r in train_rows], dtype="float32")
    y = np.asarray([r["target_log_delta"] for r in train_rows], dtype="float32")

    return Xp, Xd, Xpc, Xnc, Xsa, Xta, Xn, y


def build_repair_model(vocab_size: int) -> Model:
    prefix_act_input = Input(shape=(MAX_PREFIX_LEN,), dtype="int32", name="prefix_act_input")
    prefix_delta_input = Input(shape=(MAX_PREFIX_LEN, 1), dtype="float32", name="prefix_delta_input")
    prev_clean_input = Input(shape=(1,), dtype="int32", name="prev_clean_input")
    next_clean_input = Input(shape=(1,), dtype="int32", name="next_clean_input")
    source_act_input = Input(shape=(1,), dtype="int32", name="source_act_input")
    target_act_input = Input(shape=(1,), dtype="int32", name="target_act_input")
    num_input = Input(shape=(5,), dtype="float32", name="num_input")

    seq_emb = Embedding(input_dim=vocab_size, output_dim=EMBED_DIM, mask_zero=True)(prefix_act_input)
    seq_x = Concatenate(axis=-1)([seq_emb, prefix_delta_input])
    seq_x = LSTM(LSTM_UNITS)(seq_x)

    ctx_emb = Embedding(input_dim=vocab_size, output_dim=EMBED_DIM)
    prev_emb = Reshape((EMBED_DIM,))(ctx_emb(prev_clean_input))
    next_emb = Reshape((EMBED_DIM,))(ctx_emb(next_clean_input))
    src_emb = Reshape((EMBED_DIM,))(ctx_emb(source_act_input))
    tgt_emb = Reshape((EMBED_DIM,))(ctx_emb(target_act_input))

    x = Concatenate()([seq_x, prev_emb, next_emb, src_emb, tgt_emb, num_input])
    x = Dense(DENSE_UNITS, activation="relu")(x)
    x = Dense(DENSE_UNITS, activation="relu")(x)
    delta_out = Dense(1, activation="linear", name="delta_out")(x)

    model = Model(
        inputs=[
            prefix_act_input,
            prefix_delta_input,
            prev_clean_input,
            next_clean_input,
            source_act_input,
            target_act_input,
            num_input,
        ],
        outputs=delta_out,
    )
    model.compile(optimizer=Adam(1e-3), loss=Huber(), metrics=["mae"])
    return model


def build_prefix_inputs_for_repair(
    prefix_acts: List[str],
    prefix_deltas: List[float],
    prev_clean_act: Optional[str],
    next_clean_act: Optional[str],
    source_act: str,
    target_act: str,
    block_len: int,
    step_idx: int,
    pair_median: Dict[Tuple[str, str], float],
    trans_model: Dict[str, object],
    act2id: Dict[str, int],
):
    x_prefix_act = [act2id.get(a, 0) for a in prefix_acts][-MAX_PREFIX_LEN:]
    x_prefix_delta = [safe_log1p(x) for x in prefix_deltas][-MAX_PREFIX_LEN:]

    x_prefix_act = pad_sequences(
        [x_prefix_act],
        maxlen=MAX_PREFIX_LEN,
        padding="pre",
        truncating="pre",
        value=0,
    ).astype("int32")

    x_prefix_delta = pad_sequences(
        [x_prefix_delta],
        maxlen=MAX_PREFIX_LEN,
        padding="pre",
        truncating="pre",
        value=0.0,
    ).astype("float32")
    x_prefix_delta = np.expand_dims(x_prefix_delta, axis=-1)

    pair_prior = pair_median.get((source_act, target_act), FALLBACK_SECONDS)
    trans_logp = trans_model["log_trans"].get(
        (source_act, target_act),
        trans_model["uniform_logp"],
    )

    remaining_steps = block_len - step_idx
    step_frac = step_idx / max(block_len - 1, 1)
    remaining_frac = remaining_steps / max(block_len, 1)
    block_len_norm = min(block_len / MAX_SYN_BLOCK_LEN, 1.0)

    x_prev_clean = np.asarray([[act2id.get(prev_clean_act, 0) if prev_clean_act else 0]], dtype="int32")
    x_next_clean = np.asarray([[act2id.get(next_clean_act, 0) if next_clean_act else 0]], dtype="int32")
    x_source = np.asarray([[act2id.get(source_act, 0)]], dtype="int32")
    x_target = np.asarray([[act2id.get(target_act, 0)]], dtype="int32")

    x_num = np.asarray(
        [[
            safe_log1p(pair_prior),
            float(trans_logp),
            float(step_frac),
            float(remaining_frac),
            float(block_len_norm),
        ]],
        dtype="float32",
    )

    return [x_prefix_act, x_prefix_delta, x_prev_clean, x_next_clean, x_source, x_target, x_num]


def predict_block_timestamps_with_lstm(
    reordered_acts: List[str],
    left_act: Optional[str],
    left_ts: Optional[pd.Timestamp],
    right_act: Optional[str],
    right_ts: Optional[pd.Timestamp],
    prefix_acts: List[str],
    prefix_deltas: List[float],
    repair_model: Model,
    pair_median: Dict[Tuple[str, str], float],
    trans_model: Dict[str, object],
    act2id: Dict[str, int],
) -> Optional[List[pd.Timestamp]]:
    if left_ts is None:
        return None

    pred_durations = []
    dyn_prefix_acts = list(prefix_acts)
    dyn_prefix_deltas = list(prefix_deltas)
    prev_clean_act = left_act
    next_clean_act = right_act

    for step_idx, target_act in enumerate(reordered_acts):
        source_act = dyn_prefix_acts[-1] if dyn_prefix_acts else (left_act or target_act)

        inputs = build_prefix_inputs_for_repair(
            prefix_acts=dyn_prefix_acts,
            prefix_deltas=dyn_prefix_deltas,
            prev_clean_act=prev_clean_act,
            next_clean_act=next_clean_act,
            source_act=source_act,
            target_act=target_act,
            block_len=len(reordered_acts),
            step_idx=step_idx,
            pair_median=pair_median,
            trans_model=trans_model,
            act2id=act2id,
        )

        pred_log = repair_model(inputs, training=False).numpy()[0, 0]
        pred_sec = max(float(np.expm1(pred_log)), MIN_DURATION_SECONDS)
        pred_durations.append(pred_sec)

        dyn_prefix_acts.append(target_act)
        dyn_prefix_deltas.append(pred_sec)

    if right_ts is not None:
        window = (right_ts - left_ts).total_seconds()
        if window <= 0:
            return None

        bridge_prior = pair_median.get((reordered_acts[-1], right_act), FALLBACK_SECONDS) if right_act else FALLBACK_SECONDS
        total_pred = sum(pred_durations) + max(float(bridge_prior), MIN_DURATION_SECONDS)

        if total_pred > 0:
            scale = min(1.0, window / total_pred)
            pred_durations = [max(d * scale, MIN_DURATION_SECONDS) for d in pred_durations]

    repaired = []
    current_time = left_ts
    for d in pred_durations:
        current_time = current_time + pd.to_timedelta(d, unit="s")
        repaired.append(current_time)

    if right_ts is not None and repaired:
        if repaired[-1] >= right_ts:
            repaired[-1] = right_ts - pd.to_timedelta(MIN_DURATION_SECONDS, unit="s")
            for i in range(len(repaired) - 2, -1, -1):
                repaired[i] = min(
                    repaired[i],
                    repaired[i + 1] - pd.to_timedelta(MIN_DURATION_SECONDS, unit="s"),
                )
            if repaired[0] <= left_ts:
                return None

    return repaired


def estimate_prefix_before_block(case_df: pd.DataFrame, start_local_idx: int) -> Tuple[List[str], List[float]]:
    prefix = case_df.iloc[:start_local_idx].copy()
    if prefix.empty:
        return [], []

    acts = prefix["Activity"].astype(str).tolist()
    deltas = [0.0]
    for i in range(1, len(prefix)):
        dt = (prefix.iloc[i]["_ts"] - prefix.iloc[i - 1]["_ts"]).total_seconds()
        deltas.append(max(float(dt), MIN_DURATION_SECONDS))
    return acts, deltas


def repair_log_with_lstm(
    df: pd.DataFrame,
    repair_model: Model,
    pair_median: Dict[Tuple[str, str], float],
    trans_model: Dict[str, object],
    act2id: Dict[str, int],
) -> pd.DataFrame:
    out = df.copy()

    for _, g in out.groupby("Case", sort=False):
        g = g.sort_values(["_ts", "_orig_order"], kind="stable").reset_index()
        local_blocks = detect_same_timestamp_blocks(g)

        for start, end in local_blocks:
            left_row = g.iloc[start - 1] if start - 1 >= 0 else None
            right_row = g.iloc[end + 1] if end + 1 < len(g) else None

            # LSTM repair needs a left boundary for autoregressive reconstruction.
            if left_row is None:
                continue

            block_global_idxs = g.loc[start:end, "index"].tolist()
            observed_acts = g.loc[start:end, "Activity"].astype(str).tolist()

            left_act = str(left_row["Activity"])
            left_ts = left_row["_ts"]
            right_act = None if right_row is None else str(right_row["Activity"])
            right_ts = None if right_row is None else right_row["_ts"]

            reordered_acts = reorder_block(
                observed_acts,
                prev_act=left_act,
                next_act=right_act,
                model=trans_model,
            )

            prefix_acts, prefix_deltas = estimate_prefix_before_block(g, start)

            repaired_ts = predict_block_timestamps_with_lstm(
                reordered_acts=reordered_acts,
                left_act=left_act,
                left_ts=left_ts,
                right_act=right_act,
                right_ts=right_ts,
                prefix_acts=prefix_acts,
                prefix_deltas=prefix_deltas,
                repair_model=repair_model,
                pair_median=pair_median,
                trans_model=trans_model,
                act2id=act2id,
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
        print("Usage: python lstm_baseline.py <input_csv> <output_csv>")
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

    pair_median = compute_pair_median(work)
    trans_model = train_transition_probability_model(work)
    act2id = build_activity_vocab(work)

    clean_sequences = build_clean_training_sequences(work)
    train_rows = make_training_rows(
        clean_sequences=clean_sequences,
        pair_median=pair_median,
        trans_model=trans_model,
        act2id=act2id,
    )
    Xp, Xd, Xpc, Xnc, Xsa, Xta, Xn, y = prepare_training_arrays(train_rows)

    (
        Xp_tr, Xp_va,
        Xd_tr, Xd_va,
        Xpc_tr, Xpc_va,
        Xnc_tr, Xnc_va,
        Xsa_tr, Xsa_va,
        Xta_tr, Xta_va,
        Xn_tr, Xn_va,
        y_tr, y_va,
    ) = train_test_split(
        Xp, Xd, Xpc, Xnc, Xsa, Xta, Xn, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    vocab_size = max(act2id.values(), default=0) + 1
    repair_model = build_repair_model(vocab_size=vocab_size)
    repair_model.fit(
        [Xp_tr, Xd_tr, Xpc_tr, Xnc_tr, Xsa_tr, Xta_tr, Xn_tr],
        y_tr,
        validation_data=(
            [Xp_va, Xd_va, Xpc_va, Xnc_va, Xsa_va, Xta_va, Xn_va],
            y_va,
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    repaired = repair_log_with_lstm(
        work,
        repair_model=repair_model,
        pair_median=pair_median,
        trans_model=trans_model,
        act2id=act2id,
    )
    repaired = repaired[original_columns]
    repaired.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()