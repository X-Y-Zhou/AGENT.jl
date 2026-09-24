using AGENT, Plots, Statistics, LinearAlgebra

# Match the numerical execution used to validate these examples.
BLAS.set_num_threads(1)

# True parameters: [1.0856, 0.1959, 6.8676, 0.4923, 2.1462, 1.5242]
# Order: [sigma_on, sigma_off, rho, dm, lambda, dp].

# 1. Load the frozen weights and construct the model.
trained_params = load_trained_params(joinpath(@__DIR__, "parameters_trained", "params_trained3d.txt"))
agent = build_agent_model3d(7; hidden_channels=160, params=trained_params, τ=1.0)

# 2. Read counts and evaluate their empirical PGF on the model grid.
counts_path = joinpath(@__DIR__, "synthetic_data", "counts_example_capture_rate.txt")
U, S, P = read_counts3d(counts_path)
SSA_PGF = counts_to_pgf3d(U, S, P, agent.z1, agent.z2, agent.z3)
beta1, beta2 = read_capture_rates(joinpath(@__DIR__, "synthetic_data", "β1β2.txt"))
# Original paired beta samples and unnormalized 7 x 7 joint-KDE quadrature.
capture = build_capture_rate_quadrature(beta1, beta2, 7)

# 3. Infer positive parameters in log space, starting from all ones.
result = infer_parameters_capture(agent, capture, SSA_PGF;
    init=ones(6), a=1.0, iterations=2000,
    show_trace=false, g_tol=1e-20)
inferred_params = result.inferred_params

# 4. Compare physical parameters with the annotated truth.
true_params = [1.0856, 0.1959, 6.8676, 0.4923, 2.1462, 1.5242]
relative_errors = abs.(inferred_params .- true_params) ./ abs.(true_params)
mean_relative_error = mean(relative_errors)
println("True parameters:     ", true_params)
println("Inferred parameters: ", inferred_params)
println("Mean relative error: ", 100 * mean_relative_error, "%")

# 5. Plot the fitted PGF against its empirical target.
inferred_PGF = result.inferred_PGF
pgf_plot = scatter(SSA_PGF, inferred_PGF; label="AGENT", xlabel="SSA PGF", ylabel="Inferred PGF")
plot!(pgf_plot, [0, 1], [0, 1]; label="y = x", linestyle=:dash)
display(pgf_plot)
