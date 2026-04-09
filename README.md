This repository contains the supplementary material for the submitted paper on timestamp repair in process mining.

## Contents
- Prompt used for LLM-based code generation
- Execution pipeline for code-generation-based repair
- Baseline implementations:
  - median-based repair
  - process-aware LSTM repair
- Example input/output files
- Reproduction instructions

## Task
The goal is to repair form-based timestamp errors in event logs while preserving row order and modifying only timestamps inside corrupted blocks.

## Repository Structure
- `code/baselines/`: baseline implementations
- `code/llm/`: prompt and code-generation pipeline
- `data/sample/`: small anonymized example input
- `results/`: example outputs and summary metrics
- `docs/reproduction.md`: step-by-step reproduction guide

## Requirements
Install dependencies:

```bash
pip install -r requirements.txt
