# Data, Notebook, Illustration, and Comparison Methods

Paths below are relative to AGENT_analysis. Dimensions were read directly from files on 2026-09-24.

## Synthetic data and weights

| File | Data dimensions | Notes |
|---|---|---|
| `dataset/synthetic_data/counts_example2d.txt` | 10000×2 | U,S |
| `dataset/synthetic_data/counts_example3d.txt` | 10000×3 | U,S,P |
| `dataset/synthetic_data/counts_example_feedback.txt` | **2000×3** | Feedback U,S,P; do not describe every example as having 10000 cells |
| `dataset/synthetic_data/counts_example_toggle.txt` | 10000×6 | U1,S1,P1,U2,S2,P2 |
| `dataset/synthetic_data/counts_example_capture_rate.txt` | 10000×3 | U,S,P with capture effects |
| `dataset/synthetic_data/β1β2.txt` | 10000×2 | Preserve within-row pairing |
| `dataset/synthetic_data/ps_forinfer_2d.txt` | 400×4 | Four-parameter truth, one group per row |
| `dataset/synthetic_data/ps_forinfer_3d.txt` | 400×6 | Six-parameter truth, one group per row |
| `parameters_trained/params_trained2d.txt` | 4689 weights + one header row | params CSV |
| `parameters_trained/params_trained3d.txt` | 56983 weights + one header row | params CSV |

The five count files, beta file, and two released weight files have identical SHA-256 hashes to their counterparts under the package's `examples/`. The 400-row truth files exist only in the analysis repository; their presence does not imply that corresponding counts/PGFs are supplied for every group. Only five example count files are present. Do not substitute these examples for observations corresponding to all 400 groups.

Truth values annotated in example code:

| Example | Physical parameters, rounded to four decimal places |
|---|---|
| 2D | [3.0634, 3.2981, 21.4090, 0.7231] |
| 3D | [0.7080, 1.2732, 4.0171, 0.5903, 0.8317, 1.3848] |
| feedback | [1.0156, 0.0206, 4.0630, 0.7276, 3.2615, 0.6603] |
| capture | [1.0856, 0.1959, 6.8676, 0.4923, 2.1462, 1.5242] |
| toggle gene1 | [1.0946, 0.1216, 2.5215, 0.4321, 1.8173, 1.2662] |
| toggle gene2 | [1.4658, 0.0469, 2.6785, 0.3962, 2.4908, 1.1572] |

These values support example error reporting; they are not full-precision generating parameters.

## Real data

| File | Contents/dimensions |
|---|---|
| `dataset/real_data/190426_CTCL.loom` | matrix and three RNA-layer arrays: 32738 genes×6049 cells |
| `dataset/real_data/190426_ctrl.loom` | matrix and three RNA-layer arrays: 32738×5530 |
| `dataset/real_data/GSM3596101_CTCL-ADT-count.csv` | 52 ADT features×6500 cell columns; including the header row and label column, the file is 53×6501 |
| `dataset/real_data/GSM3596096_ctrl-ADT-count.csv` | Likewise, a 52×6500 numerical region |
| `dataset/cluster/cluster_ID/CTCL.txt` | 5317 integer labels |
| `dataset/cluster/cluster_ID/ctrl.txt` | 5064 integer labels |

In both loom files, `layers/spliced`, `layers/unspliced`, and `layers/ambiguous` are uint16. `matrix` is Float32; do not substitute it for raw count layers without checking its meaning. `row_attrs` contains Gene, Accession, Chromosome, Start, End, and Strand. `col_attrs` contains CellID, Clusters, _X, and _Y.

RNA, ADT, and labels have different cell counts. Align by CellID/barcode, shared-cell filtering, and the original script's ordering. Do not attach labels directly to original loom column positions. Label text files contain no barcode column, so their lengths alone cannot recover correspondence. Do not infer biological label names without an explicit mapping.

## Clustering scripts

`cluster_CTCL.py` and `cluster_ctrl.py` use numpy, matplotlib, velocyto, and protaccel. They import ADT and loom data, filter by expression/detection levels, enforce target retention using a built-in protein-to-gene dictionary, filter to shared cells, then impute in protein space and cluster.

Shared settings: `score_cv_vs_mean(3000,max_expr_avg=100,min_expr_cells=20)`; `score_detection_levels(min_expr_counts=40,min_cells_express=30)`; barcode matching uses the substring after the colon in loom CellID and excludes its final character; `impute(k=800,impute_in_prot_space=True,size_norm=False,impute_in_pca_space=False)`.

CTCL uses `ModularityVertexPartition` and tag correction `[0,0,1,2,1]`. Control uses `RBERVertexPartition` and tag correction `[3,0,3,1,2,0,0]`. The scripts produce runtime objects but contain no code to save results automatically into `cluster_ID/`. Do not claim that rerunning overwrites or exactly reproduces the existing labels.

Comments identify the ECCITE notebooks in pachterlab/GSP_2019 as the source of the adapted code. This is a provenance clue recorded in local source. Formal claims about patient clinical status, sampled tissue, enrollment, or diagnostic details require independent verification against original sources; local filenames do not establish those facts.

## Notebook and workflow illustration

`Tutorials.ipynb` covers the reaction model, reduced-to-full mapping, released architecture, training recipe, objective, and explicit 2D inference code. It checks that the working directory contains Project.toml and the notebook, loads released parameters, and starts inference from ones. Embedded historical outputs do not replace re-execution.

`illustrate.png` presents four workflow levels: structure, pretraining, kinetic inference, and agent-guided hyperparameter configuration. The pretraining panel shows only Adam, whereas current scripts also use L-BFGS. The automatic-configuration panel is conceptual; these repositories contain no complete automated HPO driver. Use current code as the implementation reference.

The notebook's mathematical discussion needs checking: one expression uses transposed 2D vec, while a later error expression omits the transpose; the prose describes weighted squared error, while code retains the divergence constant. Reconcile definitions and code before quoting equations rather than copying them and claiming term-by-term equivalence.

## Five comparison scripts

| Method | Implementation and key settings | Reproduction considerations |
|---|---|---|
| ABC | DelaySSAToolkit/Catalyst; tf=15, delay=1, 10000 trajectories per evaluation; KS distance after concatenating U/S; ABCRejection ε=0.2, at most 10^6 attempts; priors Uniform(0,4), Uniform(0,4), Uniform(0,50), Uniform(0,2) | Not a 2D joint KS statistic; expensive, so establish a budget; dependencies are incomplete |
| Exact | Analytical 2D PGF using HypergeometricFunctions.pFq, seven-node quadrature, log-space Optim, 1000 iterations, g_tol=1e-11 | Both target and prediction use vec without transpose internally; explicitly reorder when exchanging outputs with AGENT |
| FSP | Auxiliary ODE uses Tsit5; main distribution comes from a linear system; 60×60 support; negative model values are clipped to zero and normalized; loss `-sum(hist.*log.(output.+1e-12))` | Observation resizing can truncate without renormalization; large dense matrices are constructed; clipping/normalization belong to this method's existing implementation |
| MOM | Matches U/S means and variances using the mean of four squared relative deviations; 1000 iterations, g_tol=1e-11 | Sample var follows Julia defaults; zero observed moments cause division by zero |
| NNCME | 60×60 support, two gene states and 7200 unknowns; network 3600→5→3600; softplus rates; NLsolve steady state with custom adjoint/GMRES; Adam 1e-3, 10 epochs, λreg=1e-8 | Observations are truncated then normalized; no fixed random seed; loss calls used for training gradients do not receive the warm start, while postprocessing solves receive current_x0 |

Comparison scripts use `dataset/...` paths relative to the analysis working directory. In ABC/Exact/FSP/MOM, `include("../utils.jl")` resolves from the run_AGENT script to the existing root utils.jl; that include currently resolves correctly.

The current Project does not fully declare direct comparison-method dependencies such as ApproxBayes, Catalyst, DelaySSAToolkit, DifferentialEquations, HypergeometricFunctions, LinearMaps, IterativeSolvers, Zygote, StatsBase, and Distributions. Some may appear as indirect manifest dependencies. Prepare the environment for the specific method before running it; file presence does not imply immediate reproducibility.

In `utils.jl`, `embeding_dist` pads or truncates, `set_one` takes absolute values before normalization, and `P2mean/P2var/P2sm` interpret i−1 as the count. Do not silently apply these corrective operations to AGENT's empirical distribution reader or capture-rate integration.
