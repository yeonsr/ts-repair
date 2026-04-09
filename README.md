This repository contains the supplementary material for a submitted paper on timestamp repair in process mining.

---

## Overview

We address the problem of repairing **form-based timestamp errors** in event logs, where multiple consecutive events share identical timestamps and their internal order is ambiguous.

The goal is to reconstruct:

- the **correct event ordering**
- realistic **inter-event timing**

while preserving:

- original row order
- non-corrupted timestamps

---

## Problem Definition

Input event logs contain the following columns:

- `Case`: case identifier  
- `Activity`: activity label  
- `Timestamp`: observed timestamp  

A **form-based error** occurs when:

- multiple consecutive events share the same timestamp  
- their internal ordering is uncertain or incorrect  

The repair task is to:

1. detect such segments  
2. reconstruct the most plausible ordering  
3. assign consistent timestamps  

---

## Methods

We compare three approaches:

### 1. Median baseline
- Learns median inter-event durations for activity pairs from non-collapsed transitions  
- Repairs segments using **anchor-based scaling** within the available time window  

### 2. LSTM baseline
- Learns transition probabilities from clean trace fragments  
- Reorders segments using probabilistic scoring  
- Predicts durations using a **process-aware LSTM model** based on prefix context  

### 3. LLM-based approach (proposed)
- Uses a prompt-driven code generation paradigm  
- The model generates an executable repair script  
- Integrates ordering and timing reconstruction under structural constraints  

---

## Repository Structure
The repository is structured as follows:
```
.
├── README.md
├── requirements.txt
│
├── code/
│ ├── baselines/
│ │ ├── median_baseline.py # Median-based repair with anchor scaling
│ │ └── lstm_baseline.py # Process-aware LSTM repair
│ │
│ ├── llm/
│ │ ├── prompt.txt # Prompt used for code generation
│ │ ├── run_codegen_experiment.py # Executes generated repair script
│ │ └── example_generated_repair.py # Example LLM-generated repair code, by Grok-4.20-reasoning
│ │
│ └── evaluation/
│ └── evaluate.py # Evaluation (MAE, RMSE, ordering metrics)
│
├── data/
│ └── Credit_Form_0.3_s
│ └── Pub_Form_0.3_s
│
├── results/
│ ├── Credit_eval.json # Evaluation results (Credit dataset)
│ └── Pub_eval.json # Evaluation results (Pub dataset)
```


---

## Installation
``` bash
pip install -r requirements.txt
```

---
## Usage
### Run median baseline
```bash
python code/baselines/median_baseline.py input.csv output.csv
```
### Run LSTM baseline
```bash
python code/baselines/lstm_baseline.py input.csv output.csv
  ```
### Run LLM-based Method
```bash
python code/llm/run_codegen_experiment.py input.csv output.csv
```
### Evaluation
Evaluation results can be exported to a JSON file using:
```bash
python code/evaluation/evaluate_repair.py original.csv repaired.csv > result.json
```
Required columns for evaluation:
- Timestamp_original
- is_polluted
These columns are used only for evaluation and are not required for the repair methods.


---
## Results
Evaluation outputs are provided in the `results/` directory.

For the final code-generation experiment:
- Credit_Form_0.3_s
  - MAE: 696.55 → 524.81
  - RMSE: 7457.69 → 4283.05
  - Exact block order match: 1.000
- Pub_Form_0.3_s
  - MAE: 134898.03 → 18759.47
  - RMSE: 288739.80 → 43635.88
  - Exact block order match: 0.927

These results show substantial improvements in both temporal accuracy and ordering reconstruction.
