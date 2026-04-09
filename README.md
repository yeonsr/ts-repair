This repository contains supplementary material for a submitted paper on timestamp repair in process mining.

## Overview

We study timestamp repair under form-based event capture, where multiple events share identical timestamps and their internal order is ambiguous.

We compare three approaches:

- Statistical baseline (median-based)
- Predictive model (LSTM)
- LLM-based program synthesis (proposed method)

## Task Definition

Given an event log with columns:

- Case
- Activity
- Timestamp
- Resource

The goal is to:

- Detect form-based corrupted segments
- Reconstruct event ordering
- Repair timestamps
- Preserve row order

## Repository Structure
code/baselines/median_baseline.py
code/baselines/lstm_baseline.py
llm/prompt.txt
run_codegen_experiment.py
example_generated_repair.py

data/Credit
data/Pub

results/
