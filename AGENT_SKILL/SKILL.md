---
name: agent-julia-workflows
description: Understand, run, and maintain the neural PGF models, training data, parameter inference, feedback and toggle extensions, capture-rate integration, and real-data analysis in AGENT.jl and AGENT_analysis. Use for code navigation, experiment reproduction, data audits, and methods documentation in these repositories.
---

# AGENT Repository Workflow Guide

AGENT (Agent-guided Generating-function Estimation via Neural Transfer) maps an analytical reduced-model PGF and additional kinetic parameters to the full model's joint PGF. The network is then frozen to infer kinetic parameters from empirical PGFs computed from single-cell counts.

## Locate the repositories and select references

Default package path: `/Users/x-y-zhou/Documents/GitHub/AGENT.jl`. Default analysis path: `/Users/x-y-zhou/Documents/GitHub/AGENT_analysis`. On another machine, locate the package through this skill's parent directory and then locate the sibling analysis repository, rather than depending on the original user's paths.

Start by inspecting the current `Project.toml`, README, relevant source code, and input files. Dimensions and defaults below describe a snapshot and do not replace current checks. When code, data, and README disagree, use the actual executable code and files as evidence and explain the discrepancy.

Read references according to the task:

- Repository overview, environment, and source-file responsibilities: [Repository map](references/repository-map.md).
- PGFs, networks, basic inference, LMA, and capture-rate integration: [Models and inference](references/model-inference.md).
- Training, weight export, and data splits: [Training workflow](references/training.md).
- Synthetic and real data, the notebook, illustration, and comparison methods: [Data and analysis](references/data-analysis.md).
- Validation scope and known inconsistencies: [Validation and limitations](references/validation.md).
- Individual repository files and snapshot hashes: [File inventory](references/file-inventory.tsv). Search for specific files as needed instead of loading all numerical data at once.

## Preserve scientific semantics

1. The released 2D network is `8 → 80 → 49`, Float64; the released 3D network is `10 → 160 → 343`, Float32 by default. Both current training scripts train in Float64. Distinguish training precision from released inference precision.
2. Raw network inputs consist of seven reduced PGF values plus additional parameters. Input standardization is already folded into the released weights; do not standardize again during inference. The hidden activation is tanh, followed by a direct sigmoid output, with no residual connection or output normalization.
3. The 2D PGF uses `vec(G')`, with z2 varying fastest. The 3D PGF uses `vec(G)`, with z1 fastest, followed by z2 and z3. Distribution-array indices equal counts plus one. Do not change transposes or summation order merely for stylistic consistency.
4. Basic parameter order is `[σon, σoff, ρ, dm]` or `[σon, σoff, ρ, dm, λ, dp]`; the default fixed delay is `τ=1`. Standard inference starts from all-one physical parameters and optimizes in log space.
5. Preserve the target-only constant in `int_dist`. Its value is not conventional MSE; compute PGF MSE and relative parameter errors separately.
6. For linear feedback and toggle models, retain both equivalent linear parameters and converted physical parameters. Predict PGFs using equivalent parameters. Toggle inference consists of two 3D marginal fits, not a 6D joint PGF fit.
7. Capture-rate integration uses paired beta samples, a joint KDE, and 7×7 quadrature. Preserve the current unnormalized integration mass. Changing the domain, weights, or normalization changes model semantics.
8. Training-parameter rows correspond one-to-one to PGF columns. When changing the sample set, update all related matrices together and retain selection indices. Keep inference truth out of training, and store released weights separately from new training outputs.
9. Retain every requested group, failure record, and raw result. If inference error guides hyperparameter selection, identify the data as tuning data rather than independent validation.
10. Match execution scope to the request. Summarization or documentation alone does not call for full training, ABC, FSP, or NNCME runs. When fixing numerical algorithms, record differences from the baseline; do not silently introduce retries, clipping, or filtering into reproduction workflows.

## Working procedure

Choose either a package API example or the explicit implementation in the analysis repository. Record Julia/Flux versions, weight files, dimensions, input order, and precision. Execute within the project environment. Run the five AGENT examples in separate Julia processes to prevent collisions between top-level variables, functions, and training constants.

First check dimensions, finiteness, parameter counts, PGF ordering, and input/output semantics; then run examples appropriate to the task. Identify the truth source when reporting parameter errors: example truth values are rounded to four decimal places. State whether timings include compilation, count-to-PGF conversion, quadrature preparation, and postprocessing.

Deliver the changed-file list, data/weight provenance, commands actually executed, completed checks, and remaining limitations. Do not present documentation checks, historical test records, or cached notebook outputs as a successful experiment from the current run.
