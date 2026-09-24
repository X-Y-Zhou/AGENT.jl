using AGENT, Plots, Statistics, LinearAlgebra

# Match the numerical execution used to validate these examples.
BLAS.set_num_threads(1)

# True parameters: [3.0634, 3.2981, 21.4090, 0.7231]
# Order: [sigma_on, sigma_off, rho, dm].

# 1. Load the frozen weights and construct the model.
trained_params = load_trained_params(joinpath(@__DIR__, "parameters_trained", "params_trained2d.txt"))
agent = build_agent_model2d(7; hidden_channels=80, params=trained_params, τ=1.0)

# 2. Read counts and evaluate their empirical PGF on the model grid.
counts_path = joinpath(@__DIR__, "synthetic_data", "counts_example2d.txt")
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
