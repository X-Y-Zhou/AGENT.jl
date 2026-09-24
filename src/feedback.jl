function convert_LMA_feedback(ps::AbstractVector)
    if length(ps) != 6
        error(
            "convert_LMA_feedback expects a parameter vector of length 6: " *
            "[σ_on, σ_off, ρ, d, λ, dp]."
        )
    end

    σ_on, σ_off, ρ, d, λ, dp = ps

    g = σ_on / (σ_off + σ_on)

    mg =
        λ * ρ * σ_on *
        (
            d * dp +
            d * σ_on +
            dp * σ_on +
            σ_off * σ_on +
            σ_on^2
        ) /
        (
            d * dp *
            (σ_off + σ_on) *
            (d + σ_off + σ_on) *
            (dp + σ_off + σ_on)
        )

    converted_σ_off = σ_off * g / mg

    return converted_σ_off
end


function infer_feedback_parameters(
    bridge::AGENTModel,
    SSA_PGF::AbstractVector;
    init = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    a::Real = 1.0,
    iterations::Int = 2000,
    show_trace::Bool = false,
    g_tol::Real = 1e-20
)
    if bridge.dim != 3
        error("Feedback-model inference requires a 3D AGENT model.")
    end

    result_equal = infer_parameters(
        bridge,
        SSA_PGF;
        init = init,
        a = a,
        iterations = iterations,
        show_trace = show_trace,
        g_tol = g_tol
    )

    inferred_params_equal = result_equal.inferred_params

    converted_σ_off = convert_LMA_feedback(inferred_params_equal)

    inferred_params = [
        inferred_params_equal[1];
        converted_σ_off;
        inferred_params_equal[3:end]
    ]

    inferred_PGF = compute_full_pgf(
        bridge,
        inferred_params_equal
    )

    mse = mean((inferred_PGF .- SSA_PGF) .^ 2)

    return (
        inferred_params = inferred_params,
        inferred_params_equal = inferred_params_equal,
        inferred_PGF = inferred_PGF,
        mse = mse,
        result_equal = result_equal
    )
end