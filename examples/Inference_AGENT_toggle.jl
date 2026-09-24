using AGENT, Plots, Statistics, LinearAlgebra

# Match the numerical execution used to validate these examples.
BLAS.set_num_threads(1)

# True parameters: [1.0946, 0.1216, 2.5215, 0.4321, 1.8173, 1.2662, 1.4658, 0.0469, 2.6785, 0.3962, 2.4908, 1.1572]
# Order: [sigma_on, sigma_off, rho, dm, lambda, dp] for gene 1, then gene 2.

# 1. Load the frozen weights and construct the model.
trained_params = load_trained_params(joinpath(@__DIR__, "parameters_trained", "params_trained3d.txt"))
agent = build_agent_model3d(7; hidden_channels=160, params=trained_params, τ=1.0)

# 2. Read counts and evaluate their empirical PGF on the model grid.
counts_path = joinpath(@__DIR__, "synthetic_data", "counts_example_toggle.txt")
U1, S1, P1, U2, S2, P2 = read_counts_toggle(counts_path)
SSA_PGF1, SSA_PGF2 = counts_to_toggle_pgfs(agent, U1, S1, P1, U2, S2, P2)

# 3. Infer positive parameters in log space, starting from all ones.
result = infer_toggle_parameters(agent, SSA_PGF1, SSA_PGF2;
    init=ones(6), a=1.0, iterations=2000,
    show_trace=false, g_tol=1e-20)
inferred_params = result.inferred_params

# 4. Compare physical parameters with the annotated truth.
true_params = [1.0946, 0.1216, 2.5215, 0.4321, 1.8173, 1.2662, 1.4658, 0.0469, 2.6785, 0.3962, 2.4908, 1.1572]
relative_errors = abs.(inferred_params .- true_params) ./ abs.(true_params)
mean_relative_error = mean(relative_errors)
println("True parameters:     ", true_params)
println("Inferred parameters: ", inferred_params)
println("Mean relative error: ", 100 * mean_relative_error, "%")
# The physical feedback/off rates above have undergone the LMA conversion.
println("Equivalent gene 1 parameters: ", result.inferred_params_equal1)
println("Equivalent gene 2 parameters: ", result.inferred_params_equal2)

# 5. Plot each marginal PGF against its empirical target.
p1 = scatter(SSA_PGF1, result.inferred_PGF1; label="AGENT", xlabel="SSA PGF", ylabel="Inferred PGF", title="Gene 1")
plot!(p1, [0, 1], [0, 1]; label="y = x", linestyle=:dash)
p2 = scatter(SSA_PGF2, result.inferred_PGF2; label="AGENT", xlabel="SSA PGF", ylabel="Inferred PGF", title="Gene 2")
plot!(p2, [0, 1], [0, 1]; label="y = x", linestyle=:dash)
pgf_plot = plot(p1, p2; layout=(1, 2), size=(900, 400), margin=5 * Plots.PlotMeasures.mm)
display(pgf_plot)
