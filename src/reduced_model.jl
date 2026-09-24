function G_tele_delay(σon, σoff, ρ, τ, z)
    u1 = z - 1

    r = 1 - ρ * u1 + σoff + σon

    θ = sqrt(
        Complex(
            (ρ * u1 - σoff - σon)^2 +
            4 * ρ * σon * u1
        )
    )

    uz = (r + θ - 1) / 2
    uf = (r - θ - 1) / 2

    G1 =
        (uz * exp(-uf * τ) - uf * exp(-uz * τ)) / θ +
        ρ * u1 * σon *
        (exp(-uf * τ) - exp(-uz * τ)) /
        (θ * (σoff + σon))

    return real(G1)
end