# AGENT.jl

`AGENT.jl` provides parameter inference for stochastic biochemical reaction systems using frozen AGENT neural probability-generating-function (PGF) models. Single-cell counts are converted into empirical PGFs, and kinetic parameters are estimated by optimization in PGF space.

## Features

- Gaussian quadrature for PGF-space inference
- Count-to-PGF conversion for two- and three-species gene-expression models
- Frozen AGENT networks with explicit weight loading
- Positive, log-space parameter inference using `Optim.jl`
- Linear-LMA feedback and toggle-switch extensions
- A reusable AGENT skill documenting model conventions, training, inference, and data analysis across `AGENT.jl` and `AGENT_analysis`

## Installation

The tested environment is **Julia 1.8.0**. In the Julia REPL, install the package with:

```julia
] add https://github.com/X-Y-Zhou/AGENT.jl
```

Here, `]` enters package mode; then type `add https://github.com/X-Y-Zhou/AGENT.jl`.

Press Backspace to return to the Julia prompt, then load the package:

```julia
using AGENT
```

## Basic 2D Example

The following example performs parameter inference for a two-dimensional stochastic gene expression model.

### Full example

```julia
using AGENT, Plots, Statistics, LinearAlgebra

# Match the numerical execution used to validate these examples.
BLAS.set_num_threads(1)

# True parameters: [3.0634, 3.2981, 21.4090, 0.7231]
# Order: [sigma_on, sigma_off, rho, dm].

# 1. Load the frozen weights and construct the model.
trained_params = load_trained_params(joinpath(joinpath(pkgdir(AGENT), "examples"), "parameters_trained", "params_trained2d.txt"))
agent = build_agent_model2d(7; hidden_channels=80, params=trained_params, τ=1.0)

# 2. Read counts and evaluate their empirical PGF on the model grid.
counts_path = joinpath(joinpath(pkgdir(AGENT), "examples"), "synthetic_data", "counts_example2d.txt")
U, S = read_counts2d(counts_path)
SSA_PGF = counts_to_pgf2d(U, S, agent.z1, agent.z2)

# 3. Infer positive parameters in log space, starting from all ones.
result = infer_parameters(agent, SSA_PGF;
    init=ones(4), a=1.0, iterations=1000,
    show_trace=false, g_tol=1e-11)
inferred_params = result.inferred_params

# 4. Compare physical parameters with the annotated truth.
true_params = [3.0634, 3.2981, 21.4090, 0.7231]
relative_errors = abs.(inferred_params .- true_params) ./ abs.(true_params)
mean_relative_error = mean(relative_errors)
println("True parameters:     ", true_params)
println("Inferred parameters: ", inferred_params)
println("Mean relative error: ", 100 * mean_relative_error, "%")

# 5. Plot the fitted PGF against its empirical target.
inferred_PGF = compute_full_pgf(agent, inferred_params)
pgf_plot = scatter(SSA_PGF, inferred_PGF; label="AGENT", xlabel="SSA PGF", ylabel="Inferred PGF")
plot!(pgf_plot, [0, 1], [0, 1]; label="y = x", linestyle=:dash)
display(pgf_plot)
```

## Main Workflow

1. **Load trained parameters:** `load_trained_params(path)` reads the bundled weight vector.
2. **Build the model:** `build_agent_model2d(7; hidden_channels=80, params=weights, τ=1.0)` or `build_agent_model3d(7; hidden_channels=160, params=weights, τ=1.0)`. Omitting `params` loads the corresponding frozen weights automatically. 
3. **Read counts:** `read_counts2d(path)` returns `(U, S)`; `read_counts3d(path)` returns `(U, S, P)`. Each file row is one independent cell/trajectory.
4. **Compute the empirical PGF:** call `counts_to_pgf2d` or `counts_to_pgf3d` with the model's quadrature nodes.
5. **Infer parameters:** `infer_parameters(model, observed; init=ones(4), a=1.0, iterations=1000, g_tol=1e-11)` for 2D.
6. **Compare predictions:** `compute_full_pgf(model, result.inferred_params)` evaluates the fitted PGF. For feedback and toggle, use the returned PGFs computed from the equivalent linear parameters.
7. **Plot and inspect:** compare inferred and empirical PGFs against the identity line; when synthetic truth is available, report the parameter error separately.

## Important Functions

| Function | Purpose |
|---|---|
| `load_trained_params` | Read exported network weights |
| `build_agent_model2d`, `build_agent_model3d` | Construct the frozen network and quadrature grid |
| `load_agent_model` | Load bundled weights with the released defaults; optionally load a custom model directory |
| `read_counts2d`, `read_counts3d`, `read_counts_toggle` | Read count columns |
| `counts_to_pgf2d`, `counts_to_pgf3d`, `counts_to_toggle_pgfs` | Evaluate empirical PGFs |
| `compute_full_pgf` | Predict a full-model PGF |
| `infer_parameters` | Infer basic model parameters |
| `infer_feedback_parameters` | Infer equivalent parameters and convert the feedback rate with linear LMA |
| `infer_toggle_parameters` | Infer both marginals and convert both toggle rates |
| `read_capture_rates`, `build_capture_rate_quadrature` | Read paired beta samples and construct joint KDE quadrature |
| `infer_parameters_capture` | Infer parameters with capture-rate integration |
| `infer_nonlinear_feedback_parameters` | Original nonlinear-LMA extension, separate from the five examples |

## Example Directory

```text
examples/
  Inference_AGENT2d.jl
  Inference_AGENT3d.jl
  Inference_AGENT_feedback.jl
  Inference_AGENT_toggle.jl
  Inference_AGENT_capture_rate.jl
  parameters_trained/
    params_trained2d.txt
    params_trained3d.txt
  synthetic_data/
    counts_example2d.txt
    counts_example3d.txt
    counts_example_feedback.txt
    counts_example_toggle.txt
    counts_example_capture_rate.txt
    β1β2.txt
```

## Parameter Ordering

- **2D:** `[sigma_on, sigma_off, rho, dm]`; counts `[U, S]`.
- **3D / feedback / capture:** `[sigma_on, sigma_off, rho, dm, lambda, dp]`; counts `[U, S, P]`.
- **Toggle:** the six parameters of gene 1 followed by those of gene 2; counts `[U1, S1, P1, U2, S2, P2]`.

For feedback, the second returned physical parameter is a protein-dependent off-rate coefficient. For toggle, both off-rate coefficients are converted together. The equivalent linear parameters remain available in the result for PGF prediction.

The companion `AGENT_analysis` repository contains the main experimental datasets, complete reference results, and training/reproduction code. 

## License

MIT; see [LICENSE](LICENSE).
