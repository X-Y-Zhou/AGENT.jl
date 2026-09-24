# Models, Numerical Conventions, and Inference APIs

## Networks and inputs

The 2D model consists of gene switching, transcription of U by the active gene, conversion of U to S after a fixed delay τ, and degradation of S at rate dm. The 3D model adds protein production parameter λ and degradation parameter dp. The reduced PGF is computed analytically from `(σon,σoff,ρ,τ)`; preserve its `sqrt(Complex(...))` and `real(...)` handling.

| Property | 2D | 3D and related extensions |
|---|---|---|
| Parameter order | σon, σoff, ρ, dm | σon, σoff, ρ, dm, λ, dp |
| Network input | 7 reduced PGF values + dm | 7 reduced PGF values + dm,λ,dp |
| Dense architecture | 8→80→49 | 10→160→343 |
| Total weights and biases | 4689 | 56983 |
| Released default weight precision | Float64 | Float32 |
| PGF flattening | `vec(G')`, z2 fastest | `vec(G)`, z1 fastest, then z2 and z3 |
| Default iteration limit | 1000 | 2000 |
| Default g_tol | 1e-11 | 1e-20 |

Read weights with `CSV.read(...,DataFrame).params`, not a headerless `readdlm`. Reconstruct the two-layer core through Flux.destructure, then append sigmoid exactly once. `AGENTModel` stores dim, z1/z2/z3, W, network, params, τ, and n_extra_params. In 2D, z3 is nothing. Grid nodes and quadrature weights are Float64; not every 3D intermediate is forced to Float32.

`build_agent_model2d()` / `build_agent_model3d()` automatically read package example weights and accept explicit `params`, `hidden_channels`, `precision`, and `τ`. The released loader requires n=7 and a weight-vector length matching the architecture. Although the API accepts a different τ, this does not establish that the released network has been validated at that delay.

For `load_agent_model(dim; directory=...)`, the custom directory must contain `model_config.toml` with hidden="tanh", output="sigmoid", residual=false, standardization_folded=true, width, inference_precision, and output_order. The order strings must be `z2 fastest, then z1` or `z1 fastest, then z2 then z3`. The expected weight filename is **params_trained.txt** for 2D and **params_trained3d.txt** for 3D. Training does not generate this configuration automatically, and its 2D output filename differs. Passing weights directly through the builder's params argument is the clearest loading route.

## Empirical PGFs and objective

Each count row represents an independent cell/trajectory. `read_counts2d/3d` take the first two/three columns and convert them to Int. First verify nonnegative integers, nonempty data, and equal column lengths; the readers are not comprehensive data validators.

`counts_to_joint_prob*` constructs a dense histogram extending to the maximum observed counts and divides by sample size. The PGF is the empirical mean `mean(z1^U*z2^S[ * z3^P])`. Dense arrays can be large for sparse data with high counts. When optimizing this implementation, compare results and numerical differences caused by summation order.

Gauss–Legendre nodes are mapped from [-1,1] to [0,1], with seven nodes per axis and no endpoints. In 2D, `W=vec(w*w')`; in 3D, concatenate `vec(w*w')*w[k]` for each layer. Basic quadrature weights sum to approximately one.

For prediction p and observation y, the implemented objective is:

```julia
sum(W .* (p.^(1+a) .- p.^a .* y .* (1+1/a) .+ y/a))
```

Here a>0, with a=1 by default. At a=1 this equals weighted squared error plus `sum(W.*(y-y.^2))`, so it is generally nonzero even when p=y. Do not remove the constant and claim numerical identity with the baseline. The returned `mse` field, where available, is unweighted PGF MSE and differs from the optimization objective.

Optim receives only the objective, initial values, and Options, with no explicit gradient or algorithm. It uses the default derivative-free method in the locked version. Model differentiability or the presence of g_tol does not mean inference uses gradient descent. Initial values are `log.(ones(...))`; inferred physical parameters are `exp.(Optim.minimizer(opt))`.

## Basic API example

```julia
using AGENT, LinearAlgebra
BLAS.set_num_threads(1)
m = build_agent_model2d()
counts = joinpath(pkgdir(AGENT), "examples", "synthetic_data", "counts_example2d.txt")
U, S = read_counts2d(counts)
y = counts_to_pgf2d(U, S, m.z1, m.z2)
r = infer_parameters(m, y; init=ones(4), a=1.0,
                     iterations=1000, g_tol=1e-11)
pred = compute_full_pgf(m, r.inferred_params)
```

`infer_parameters` returns `(inferred_params,opt,time)`, without inferred_PGF or mse fields. The timer surrounds only the Optim call, excluding model construction, data loading, and empirical PGF calculation. The first call may include compilation.

## Feedback and toggle

- `infer_feedback_parameters(m,y)` first fits an equivalent linear 3D model, then replaces the second parameter through `convert_LMA_feedback`. The conversion is `σoff*g/mg`, where g is the active fraction and mg is the protein–active-gene mixed-moment expression used in the code. It returns physical parameters, `inferred_params_equal`, `inferred_PGF` computed from equivalent parameters, mse, and `result_equal`. Optimization timing is available as `result_equal.time`.
- `read_counts_toggle` reads `[U1,S1,P1,U2,S2,P2]`. `counts_to_toggle_pgfs(m,counts...)` produces two marginal vectors of length 343. The init argument of `infer_toggle_parameters` has length six and is used for two independent optimizations; it does not have length twelve.
- `convert_LMA_toggle(vcat(equal1,equal2))` converts both off-rates using protein-related moments of the other gene. Physical parameters are ordered as six for gene1 followed by six for gene2. Returns include `inferred_params_equal1/2`, `inferred_PGF1/2`, `mse1/2`, and `result1/2`. End-to-end timing must additionally include conversion and prediction.

## Capture-rate integration

`read_capture_rates` returns paired β1 and β2. `build_capture_rate_quadrature(β1,β2,7)` constructs a two-dimensional KDE and linear interpolation with flat boundary extrapolation. Each integration interval spans that column's observed minimum and maximum.

β1 replaces the reduced model's ρ with ρβ1; β2 replaces the network's additional parameter λ with λβ2. dm and dp remain unchanged. The 49 network-input columns are ordered with β1 fastest, followed by β2. Integration returns `network(X)*vec(weighted_density)*scale`, where scale is the product of the two half-interval lengths.

The code does not normalize `scale*sum(weighted_density)`. Measure and record this mass during reproduction rather than silently dividing by it. `infer_parameters_capture` returns inferred_params, inferred_PGF, mse, opt, and time, and supports only 3D models.

## Original nonlinear LMA

`infer_nonlinear_feedback_parameters` is separate from the five examples. The outer optimization fits physical parameters; the inner `nlsolve` solves six variables `[g,m,p,mg,pg,soff]`, always starting from six ones. Its sixth equation is `soff-sof*pg/g=0`. The code takes `result.zero[6]` as equivalent soff, then evaluates the neural PGF.

Preserve the existing semantics: two predict calls per objective, no rejection of unconverged roots, and no automatic retries. Exceptions are counted and rethrown. Diagnostics returned are inner_calls, inner_not_converged, inner_nonpositive_soff, inner_nonfinite_soff, inner_exceptions, and inner_max_residual. Introduce more robust root-solving strategies explicitly as a new method, not as unchanged reproduction.

## Other data APIs

`read_distributions` reads headers such as `Matrix 1` followed by comma-separated rows. IDs must be consecutive from one; probabilities must be finite and nonnegative; each block's sum must differ from one by less than 1e-12. It neither reorders nor renormalizes values. `distributions_to_pgfs2d` returns 49×number-of-groups.

`read_pgf_matrix(path,dim;groups=nothing)` uses whitespace `readdlm`, reshapes to 7^dim rows, and checks finiteness and optionally the group count. Reshaping cannot establish the source orientation or probability range; inspect the source first.
