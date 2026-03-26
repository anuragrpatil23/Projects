# Autoresearch

Autonomous LLM pretraining research experiments on Minerva HPC, based on [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

## Overview

An AI agent (Claude Code) autonomously modifies `train.py` to explore model architecture, optimizer, and hyperparameter changes. Each experiment runs for a fixed 5-minute time budget on a single H100 80GB GPU. The agent keeps improvements and discards regressions, advancing the branch incrementally.

## Setup

- **Platform:** Minerva HPC, NVIDIA H100 80GB HBM3
- **PyTorch:** 2.10.0+cu130
- **Model:** GPT with Muon+AdamW optimizer, Flash Attention 3, torch.compile
- **Metric:** val_bpb (bits per byte on validation set — lower is better)
- **Baseline:** val_bpb = 1.016074 (50.3M params, depth=8)

## Results

Experiment results are logged in `results.tsv` during runs. See the autoresearch branch for the latest state of `train.py`.
