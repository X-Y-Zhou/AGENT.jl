# Repository Map and Environment

## Responsibilities

`AGENT.jl` is the Julia package. `AGENT_analysis` contains training, explicit inference demonstrations, comparison methods, a notebook, and datasets. The package name is `AGENT`, version `0.1.0`, UUID `67660b2f-b957-44fd-967a-aed50f0343fb`. Both repositories use the MIT license.

The snapshot contains 78 project files: 31 in the package and 47 in the analysis repository, including two empty `.vscode/settings.json` files and excluding `.git` internals, `.DS_Store`, and this skill. See `file-inventory.tsv` for file sizes and SHA-256 hashes. Numerical datasets are not duplicated inside the skill.

## AGENT.jl

| Path | Contents and entry points |
|---|---|
| `src/AGENT.jl` | Module, imports, include order, and public exports; inspect when changing the API |
| `src/reduced_model.jl` | `G_tele_delay(σon,σoff,ρ,τ,z)`: analytical reduced PGF using a complex square root and taking the real part |
| `src/quadrature.jl` | Gauss–Legendre nodes on `[0,1]` and 2D/3D tensor-product weights |
| `src/network.jl` | `AGENTModel`, weight CSV loading, network reconstruction, default/custom-directory loading, and `compute_full_pgf` |
| `src/pgf_utils.jl` | Count readers, empirical joint distributions, 1D/2D/3D distribution-to-PGF and count-to-PGF conversion |
| `src/inference.jl` | `int_dist` and positive-parameter inference with Optim in log space |
| `src/feedback.jl` | Linear-LMA feedback off-rate conversion and inference wrapper |
| `src/toggle.jl` | Six-column counts, two 3D marginal PGFs, two fits, and coupled off-rate conversion |
| `src/capture_rate.jl` | Beta samples, joint KDE quadrature object, and capture-aware PGF/loss/inference |
| `src/nonlinear_feedback.jl` | Inner nonlinear LMA with NLsolve, outer optimization, and inner-solver diagnostics |
| `src/data.jl` | `Matrix <id>` distribution blocks, batch conversion to 2D PGFs, and precomputed PGF matrix reading |
| `examples/Inference_AGENT2d.jl` | Four-parameter API demonstration |
| `examples/Inference_AGENT3d.jl` | Six-parameter API demonstration |
| `examples/Inference_AGENT_feedback.jl` | Feedback API demonstration |
| `examples/Inference_AGENT_toggle.jl` | Toggle API demonstration |
| `examples/Inference_AGENT_capture_rate.jl` | Capture-rate API demonstration |
| `examples/parameters_trained/` | Two released weight CSVs with `.txt` extensions and a `params` column |
| `examples/synthetic_data/` | Five count files and `β1β2.txt` |

The five package examples call APIs such as `build_agent_model*` and `infer_*`, and display parameter errors and PGF scatterplots. They do not automatically save batch results.

## AGENT_analysis

| Path | Contents and boundaries |
|---|---|
| `run_AGENT/Inference_AGENT*.jl` | Five similarly named demonstrations that explicitly construct Flux networks, objectives, and Optim calls; not verbatim copies of the package examples |
| `run_AGENT/Inference_ABC.jl` | Delay SSA and ABC rejection |
| `run_AGENT/Inference_Exact.jl` | Analytical 2D PGF inference using hypergeometric functions |
| `run_AGENT/Inference_FSP.jl` | Delayed-model distribution calculation and maximum likelihood on a 60×60 truncated support |
| `run_AGENT/Inference_MOM.jl` | Mean and variance matching |
| `run_AGENT/Inference_NNCME.jl` | Steady-state equations, neural effective rates, and implicit differentiation |
| `train_AGENT/2d/`, `train_AGENT/3d/` | Each contains a training script, parameter table, reduced PGFs, and full PGFs; see the training reference |
| `parameters_trained/` | Released 2D/3D weights, byte-identical to the package copies |
| `dataset/synthetic_data/` | Example counts, 400-row 2D/3D inference-truth tables, and paired beta samples |
| `dataset/real_data/` | Two loom files and two ADT CSVs |
| `dataset/cluster/` | CTCL/control Python clustering scripts and two label files |
| `utils.jl` | Reduced PGF, histograms, padding/truncation, normalization, moments, and PGF conversion; shared by comparison scripts, not part of the AGENT package |
| `Tutorials.ipynb` | 17 cells: 6 markdown and 11 code; Julia 1.8.0; explicit introductory 2D workflow |
| `illustrate.png` | 2845×3660 bitmap with four sections: network structure, pretraining, inference, and agent-guided hyperparameter configuration; no editable vector source is included |
| `setup.jl` | Activates the analysis environment, calls `Pkg.develop(path="../AGENT.jl")`, then `Pkg.instantiate()` |

## Environment and auxiliary files

Both manifests declare Julia `1.8.0` and manifest format `2.0`, with 301 and 302 dependency entries respectively. The analysis manifest references AGENT through `../AGENT.jl`. Do not manually insert absolute paths into the manifest or refresh lockfiles during a summarization task.

Key exact compat versions: Flux 0.13.17, Optim 1.11.0, FastGaussQuadrature 1.0.2, CSV 0.10.15, DataFrames 1.7.1, KernelDensity 0.6.10, Interpolations 0.15.1, NLsolve 4.5.1, and Plots 1.40.20. Training uses older Flux APIs; upgrading requires more than mechanical substitutions.

`.gitattributes` contains `* text=auto`. `.gitignore` excludes `.DS_Store`, `.vscode/`, `__pycache__/`, `*.pyc`, and `outputs/`, so new training results are ignored by Git by default. `.git` contains version-control metadata; `.DS_Store` is a Finder cache. Both `AGENT_SKILL/` directories were empty before this skill was created; only the package directory received the skill.

## Reusable execution commands

Run from the analysis repository; use the local Julia executable if `julia` is absent from PATH:

```sh
cd /Users/x-y-zhou/Documents/GitHub/AGENT_analysis
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. run_AGENT/Inference_AGENT2d.jl
```

To restore the local package association, run `julia --project=. setup.jl`; this changes the environment and is not a read-only check. Replace the filename to run another AGENT example. From the package directory, use `julia --project=. examples/Inference_AGENT2d.jl` for the package example.

The Julia 1.8.0 executable located on this machine is `/Applications/Julia-1.8.app/Contents/Resources/julia/bin/julia`. This is a machine-specific convenience path. Start the notebook with the analysis repository as its working directory and execute cells in order.
