# CLAUDE.md — Learn-AI

This repository contains multiple independent machine learning projects. Work should stay scoped to the relevant subdirectory rather than assuming a single repo-wide environment.

## How To Work Here

- Start by reading this repository's local docs, especially `README.md` and any project-specific README files.
- Treat each subproject as its own environment and workflow unless the code clearly shows otherwise.
- Prefer small, local validation first before proposing heavier runs.

## Minerva HPC Escalation

When work needs GPUs, longer training runs, larger datasets, or more compute than is reasonable locally, check the Mount Sinai Minerva instructions before proceeding.

Primary references:

- `/Users/patila06/Documents/MountSinaiGit/CLAUDE.md`
- `/Users/patila06/Documents/MountSinaiGit/minerva-workspace/CLAUDE.md`

Key rule:

- For Minerva work, use the documented SSH ControlMaster socket workflow at `/tmp/minerva-sock`.
- Prefer reading remote state before changing anything on the cluster.
- Be careful with shared HPC resources and confirm before destructive remote actions.

## Practical Default

Use this repository for code, local iteration, and project context. Use the Mount Sinai Minerva instructions whenever the task crosses into HPC execution, GPU scheduling, or remote cluster operations.
