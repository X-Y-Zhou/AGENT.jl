function read_counts_toggle(path::AbstractString)
    counts = readdlm(path)

    if size(counts, 2) < 6
        error(
            "Toggle-switch counts file should contain at least 6 columns: " *
            "U1, S1, P1, U2, S2, P2."
        )
    end

    U1_sample = Int.(counts[:, 1])
    S1_sample = Int.(counts[:, 2])
    P1_sample = Int.(counts[:, 3])

    U2_sample = Int.(counts[:, 4])
    S2_sample = Int.(counts[:, 5])
    P2_sample = Int.(counts[:, 6])

    return U1_sample, S1_sample, P1_sample,
           U2_sample, S2_sample, P2_sample
end


function counts_to_toggle_pgfs(
    bridge::AGENTModel,
    U1_counts::AbstractVector{<:Integer},
    S1_counts::AbstractVector{<:Integer},
    P1_counts::AbstractVector{<:Integer},
    U2_counts::AbstractVector{<:Integer},
    S2_counts::AbstractVector{<:Integer},
    P2_counts::AbstractVector{<:Integer}
)
    if bridge.dim != 3
        error("Toggle-switch inference requires a 3D AGENT model.")
    end

    z3 = bridge.z3 === nothing ? error("bridge.z3 is missing.") : bridge.z3

    SSA_PGF1 = counts_to_pgf3d(
        U1_counts,
        S1_counts,
        P1_counts,
        bridge.z1,
        bridge.z2,
        z3
    )

    SSA_PGF2 = counts_to_pgf3d(
        U2_counts,
        S2_counts,
        P2_counts,
        bridge.z1,
        bridge.z2,
        z3
    )

    return SSA_PGF1, SSA_PGF2
end


function convert_LMA_toggle(ps::AbstractVector)
    if length(ps) != 12
        error(
            "convert_LMA_toggle expects a parameter vector of length 12: " *
            "[σ_on1, σ_off1, ρ1, d1, λ1, dp1, " *
            "σ_on2, σ_off2, ρ2, d2, λ2, dp2]."
        )
    end

    σ_on1, σ_off1, ρ1, d1, λ1, dp1,
    σ_on2, σ_off2, ρ2, d2, λ2, dp2 = ps

    g1 = σ_on1 / (σ_off1 + σ_on1)
    g2 = σ_on2 / (σ_off2 + σ_on2)

    m1g2 =
        λ1 * ρ1 * σ_on1 * σ_on2 /
        (
            dp1 * d1 *
            (σ_off1 + σ_on1) *
            (σ_off2 + σ_on2)
        )

    m2g1 =
        λ2 * ρ2 * σ_on1 * σ_on2 /
        (
            dp2 * d2 *
            (σ_off1 + σ_on1) *
            (σ_off2 + σ_on2)
        )

    converted_σ_off1 = σ_off1 * g1 / m2g1
    converted_σ_off2 = σ_off2 * g2 / m1g2

    return [converted_σ_off1, converted_σ_off2]
end


function infer_toggle_parameters(
    bridge::AGENTModel,
    SSA_PGF1::AbstractVector,
    SSA_PGF2::AbstractVector;
    init = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    a::Real = 1.0,
    iterations::Int = 2000,
    show_trace::Bool = false,
    g_tol::Real = 1e-20
)
    if bridge.dim != 3
        error("Toggle-switch inference requires a 3D AGENT model.")
    end

    result1 = infer_parameters(
        bridge,
        SSA_PGF1;
        init = init,
        a = a,
        iterations = iterations,
        show_trace = show_trace,
        g_tol = g_tol
    )

    result2 = infer_parameters(
        bridge,
        SSA_PGF2;
        init = init,
        a = a,
        iterations = iterations,
        show_trace = show_trace,
        g_tol = g_tol
    )

    inferred_params_equal1 = result1.inferred_params
    inferred_params_equal2 = result2.inferred_params

    soff_all = convert_LMA_toggle(
        [inferred_params_equal1; inferred_params_equal2]
    )

    inferred_params = [
        inferred_params_equal1[1];
        soff_all[1];
        inferred_params_equal1[3:end];
        inferred_params_equal2[1];
        soff_all[2];
        inferred_params_equal2[3:end];
    ]

    inferred_PGF1 = compute_full_pgf(
        bridge,
        inferred_params_equal1
    )

    inferred_PGF2 = compute_full_pgf(
        bridge,
        inferred_params_equal2
    )

    mse1 = mean((inferred_PGF1 .- SSA_PGF1) .^ 2)
    mse2 = mean((inferred_PGF2 .- SSA_PGF2) .^ 2)

    return (
        inferred_params = inferred_params,
        inferred_params_equal1 = inferred_params_equal1,
        inferred_params_equal2 = inferred_params_equal2,
        inferred_PGF1 = inferred_PGF1,
        inferred_PGF2 = inferred_PGF2,
        mse1 = mse1,
        mse2 = mse2,
        result1 = result1,
        result2 = result2
    )
end