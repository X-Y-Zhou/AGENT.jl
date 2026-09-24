function _gauss_nodes_weights(n::Int; lower::Real = 0.0, upper::Real = 1.0)
    interval_X, weights = gausslegendre(n)

    x = ((upper - lower) .* interval_X .+ upper .+ lower) ./ 2
    w = weights .* (upper - lower) ./ 2

    return Float64.(x), Float64.(w)
end


function gauss_grid_2d(n::Int; lower::Real = 0.0, upper::Real = 1.0)
    x, w = _gauss_nodes_weights(n; lower = lower, upper = upper)

    z1 = copy(x)
    z2 = copy(x)

    W = vec(w * w')

    return z1, z2, Float64.(W)
end


function gauss_grid_3d(n::Int; lower::Real = 0.0, upper::Real = 1.0)
    x, w = _gauss_nodes_weights(n; lower = lower, upper = upper)

    z1 = copy(x)
    z2 = copy(x)
    z3 = copy(x)

    W = vcat([vec(w * w') .* w[k] for k in eachindex(w)]...)

    return z1, z2, z3, Float64.(W)
end



