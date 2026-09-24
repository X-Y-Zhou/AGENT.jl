# Validation Results and Scope

## Snapshot coverage

This skill was created from the two local working trees on 2026-09-24. Inspection covered all Julia/Python source files, both READMEs and Projects, manifest metadata and dependency entries, all notebook source cells, auxiliary configuration, table dimensions and weights, internal loom dataset structure, and a visual inspection of illustrate.png. The complete file list is in `file-inventory.tsv`.

Numerical text files were traversed to confirm consistent row/column structure, and weights and duplicate example files were compared byte for byte. Loom inspection covered internal structure, dimensions, and types, not full-matrix numerical statistics or validation of biological annotations. `.git` internals and Finder caches are excluded from scientific content.

Many project files were untracked in the initial Git status, so HEAD alone did not represent the current code. The inventory records SHA-256 hashes of actual working-tree contents. Skill creation made no Git commit and changed no other source files.

## Checks completed during skill creation

The following checks were executed during the original 2026-09-24 inspection:

- Julia executable version was 1.8.0; `using AGENT, Optim, LinearAlgebra` succeeded.
- Both released models loaded successfully, with Float64/Float32 network weights and 4689/56983 parameters respectively.
- `compute_full_pgf` evaluated all-one physical parameters and returned lengths 49/343.
- Basic 2D/3D quadrature weights summed to approximately 1.0000000000000007 / 1.0000000000000013.
- Using each prediction as its own target gave `int_dist` values 0.21561529997142526 / 0.23711321692843607, confirming the target-only constant and showing why this objective is not zero-residual MSE.
- Capture-rate KDE constructed from the existing beta data had integration mass `0.9182808143876362`, without normalization.
- An Optim call without an explicit algorithm used NelderMead in the locked environment.
- Current 2D/3D training parameters had no exact overlap with corresponding 400-row inference-truth tables, and no complete parameter-vector overlap within an absolute difference of 1e-12 in every coordinate.
- The package and analysis copies of both weight files, all five count files, and the beta file matched.

During 3D loading/prediction, Flux warned that Float32 parameters received Float64 inputs and converted them. This was the observed default mixed-precision behavior, not a loading failure.

The inspection did not rerun all five parameter optimizations, full training, 400-group inference, real-data clustering, or comparison methods. Historical test results were not counted as checks from that inspection. `Pkg.test()` was not run because the package had no `test/runtests.jl`. A remembered historical test pass cannot compensate for an absent current test directory.

## Boundaries of the current contents

1. The package README mentions “complete reference results,” but the inspected working trees contain no complete batch-inference result tables, training logs/outputs, or observations for all 400 groups. Locate actual files or obtain the missing source from the user when needed.
2. Notebook equations and code differ in their transpose/constant-term descriptions, and the pretraining diagram shows only Adam. Base methods writing on executable code; update original documentation when that is part of the requested task.
3. Real-data RNA, ADT, and cluster-label counts differ. Recheck mappings rather than aligning by position alone.
4. Comparison-method includes point to the existing utils.jl. Do not repeat older claims that this file is missing. The principal issues are incomplete direct dependencies and the absence of executed algorithm validation during this inspection.
5. New training outputs contain folded weights but lack the model_config.toml required by the custom-directory loader; the 2D filename also differs.
6. This skill summarizes repository contents and the inspected implementation. It does not incorporate external BRIDGE/HPO history, checkpoints not supplied with the repositories, additional data, or unverifiable performance figures.

## Selecting future checks

For numerical changes, start with small behavioral checks: asymmetric count examples to detect PGF flattening axes, old/new predictions on identical inputs, weight counts and dtypes, and equivalence after folding standardization. Training changes can first use a short run that exercises export. Complete full-scale experiments when the user requests full training or performance conclusions.

For basic inference, report actual opt status, measured runtime, and physical parameter errors; retain equivalent parameters for LMA. Establish a separate baseline for distribution or capture-weight changes, documenting mass, boundaries, and normalization behavior. Sigmoid outputs in [0,1] do not establish every physical property required of a PGF.
