struct CaptureRateQuadrature{I}
    β1::Vector{Float64}
    β2::Vector{Float64}

    β1_nodes::Vector{Float64}
    β2_nodes::Vector{Float64}

    β1_min::Float64
    β1_max::Float64
    β2_min::Float64
    β2_max::Float64

    density_values::Matrix{Float64}
    reference_weights::Vector{Float64}
    weighted_density::Matrix{Float64}
    scale::Float64

    interpolation::I
end


function read_capture_rates(path::AbstractString)
    data = readdlm(path)

    if size(data, 2) < 2
        error("Capture-rate file should contain at least two columns: β1 and β2.")
    end

    β1 = Float64.(data[:, 1])
    β2 = Float64.(data[:, 2])

    return β1, β2
end


function build_capture_rate_quadrature(
    β1::AbstractVector,
    β2::AbstractVector,
    n::Int = 7
)
    if length(β1) != length(β2)
        error("β1 and β2 should have the same length.")
    end

    β1 = Float64.(β1)
    β2 = Float64.(β2)

    interval_X, weights = gausslegendre(n)

    density_joint = kde((β1, β2))

    interp_joint = Interpolations.LinearInterpolation(
        (density_joint.x, density_joint.y),
        density_joint.density;
        extrapolation_bc = Interpolations.Flat()
    )

    β1_min = minimum(β1)
    β1_max = maximum(β1)

    β2_min = minimum(β2)
    β2_max = maximum(β2)

    β1_nodes =
        (β1_max - β1_min) / 2 .* interval_X .+
        (β1_max + β1_min) / 2

    β2_nodes =
        (β2_max - β2_min) / 2 .* interval_X .+
        (β2_max + β2_min) / 2

    density_values = [
        interp_joint(β1_nodes[i], β2_nodes[j])
        for i in 1:n, j in 1:n
    ]

    weighted_density = density_values .* (weights * weights')

    scale =
        ((β1_max - β1_min) / 2) *
        ((β2_max - β2_min) / 2)

    return CaptureRateQuadrature(
        β1,
        β2,
        Float64.(β1_nodes),
        Float64.(β2_nodes),
        Float64(β1_min),
        Float64(β1_max),
        Float64(β2_min),
        Float64(β2_max),
        Float64.(density_values),
        Float64.(weights),
        Float64.(weighted_density),
        Float64(scale),
        interp_joint
    )
end


function G_tele_delay_cp(σon, σoff, ρ, τ, β, z)
    ρ_eff = ρ * β

    u1 = z - 1
    r = 1 - ρ_eff * u1 + σoff + σon

    θ = sqrt(
        Complex(
            (ρ_eff * u1 - σoff - σon)^2 +
            4 * ρ_eff * σon * u1
        )
    )

    uz = (r + θ - 1) / 2
    uf = (r - θ - 1) / 2

    G1 =
        (uz * exp(-uf * τ) - uf * exp(-uz * τ)) / θ +
        ρ_eff * u1 * σon *
        (exp(-uf * τ) - exp(-uz * τ)) /
        (θ * (σoff + σon))

    return real(G1)
end


function compute_full_pgf_capture(
    bridge::AGENTModel,
    capture::CaptureRateQuadrature,
    ps::AbstractVector
)
    if bridge.dim != 3
        error("Capture-rate inference requires a 3D AGENT model.")
    end

    if length(ps) != 6
        error(
            "compute_full_pgf_capture expects six parameters: " *
            "[σ_on, σ_off, ρ, d_m, λ, d_p]."
        )
    end

    σon, σoff, ρ, dm, λ, dp = ps

    mdl = bridge.network

    nβ1 = length(capture.β1_nodes)
    nβ2 = length(capture.β2_nodes)
    L1 = length(bridge.z1)

    Gβ1 = G_tele_delay_cp.(
        σon,
        σoff,
        ρ,
        bridge.τ,
        reshape(capture.β1_nodes, 1, nβ1),
        reshape(bridge.z1, L1, 1)
    )

    G_tiled = repeat(Gβ1, 1, nβ2)

    dm_row = fill(dm, 1, nβ1 * nβ2)
    dp_row = fill(dp, 1, nβ1 * nβ2)

    λβ2_row = repeat(
        reshape(λ .* capture.β2_nodes, 1, nβ2);
        inner = (1, nβ1)
    )

    X = vcat(
        G_tiled,
        dm_row,
        λβ2_row,
        dp_row
    )

    Y = mdl(X)

    wcol = vec(capture.weighted_density)

    out = Y * wcol

    return vec(out .* capture.scale)
end


function int_dist_capture(
    bridge::AGENTModel,
    capture::CaptureRateQuadrature,
    ps::AbstractVector,
    SSA_PGF::AbstractVector,
    a::Real
)
    if a <= 0
        error("The divergence parameter a must be positive. You used a = $a.")
    end

    inferred_PGF = compute_full_pgf_capture(
        bridge,
        capture,
        ps
    )

    if length(inferred_PGF) != length(SSA_PGF)
        error(
            "Length mismatch: inferred PGF has length $(length(inferred_PGF)), " *
            "but SSA_PGF has length $(length(SSA_PGF))."
        )
    end

    if length(bridge.W) != length(SSA_PGF)
        error(
            "Length mismatch: quadrature weight vector has length $(length(bridge.W)), " *
            "but SSA_PGF has length $(length(SSA_PGF))."
        )
    end

    dist =
        inferred_PGF .^ (1 + a) .-
        inferred_PGF .^ a .* SSA_PGF .* (1 + 1 / a) .+
        SSA_PGF / a

    return sum(bridge.W .* dist)
end


function infer_parameters_capture(
    bridge::AGENTModel,
    capture::CaptureRateQuadrature,
    SSA_PGF::AbstractVector;
    init = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    a::Real = 1.0,
    iterations::Int = 2000,
    show_trace::Bool = false,
    g_tol::Real = 1e-20
)
    if bridge.dim != 3
        error("Capture-rate inference requires a 3D AGENT model.")
    end

    init_ps = log.(Float64.(init))

    objective = log_ps -> int_dist_capture(
        bridge,
        capture,
        exp.(log_ps),
        SSA_PGF,
        a
    )

    timed_result = @timed Optim.optimize(
        objective,
        init_ps,
        Optim.Options(
            show_trace = show_trace,
            g_tol = g_tol,
            iterations = iterations
        )
    )

    opt = timed_result.value
    elapsed_time = timed_result.time

    inferred_params = exp.(Optim.minimizer(opt))

    inferred_PGF = compute_full_pgf_capture(
        bridge,
        capture,
        inferred_params
    )

    mse = mean((inferred_PGF .- SSA_PGF) .^ 2)

    return (
        inferred_params = inferred_params,
        inferred_PGF = inferred_PGF,
        mse = mse,
        opt = opt,
        time = elapsed_time
    )
end