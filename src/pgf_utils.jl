function read_counts2d(path::AbstractString)
    counts = readdlm(path)

    U_sample = Int.(counts[:, 1])
    S_sample = Int.(counts[:, 2])

    return U_sample, S_sample
end


function read_counts3d(path::AbstractString)
    counts = readdlm(path)

    U_sample = Int.(counts[:, 1])
    S_sample = Int.(counts[:, 2])
    P_sample = Int.(counts[:, 3])

    return U_sample, S_sample, P_sample
end


function counts_to_joint_prob2d(
    U_counts::AbstractVector{<:Integer},
    S_counts::AbstractVector{<:Integer}
)
    @assert length(U_counts) == length(S_counts)

    U_max = maximum(U_counts)
    S_max = maximum(S_counts)

    joint_prob_matrix = zeros(Float64, U_max + 1, S_max + 1)

    for i in eachindex(U_counts)
        u = U_counts[i]
        s = S_counts[i]

        joint_prob_matrix[u + 1, s + 1] += 1
    end

    joint_prob_matrix ./= length(U_counts)

    return joint_prob_matrix
end


function counts_to_joint_prob3d(
    U_counts::AbstractVector{<:Integer},
    S_counts::AbstractVector{<:Integer},
    P_counts::AbstractVector{<:Integer}
)
    @assert length(U_counts) == length(S_counts) == length(P_counts)

    U_max = maximum(U_counts)
    S_max = maximum(S_counts)
    P_max = maximum(P_counts)

    joint_prob_matrix = zeros(Float64, U_max + 1, S_max + 1, P_max + 1)

    for i in eachindex(U_counts)
        u = U_counts[i]
        s = S_counts[i]
        p = P_counts[i]

        joint_prob_matrix[u + 1, s + 1, p + 1] += 1
    end

    joint_prob_matrix ./= length(U_counts)

    return joint_prob_matrix
end


function hist_gf1d(hist_data::AbstractVector, z)
    N = length(hist_data)

    if z isa Number
        return sum(hist_data[i] * z^(i - 1) for i in 1:N)
    else
        return [
            sum(hist_data[i] * z_value^(i - 1) for i in 1:N)
            for z_value in z
        ]
    end
end


function hist_gf2d(
    hist_data::AbstractMatrix,
    z1::AbstractVector,
    z2::AbstractVector
)
    # Preserve the original two-species summation order.
    z1_vec = [z1.^i for i in 0:size(hist_data, 1)-1]
    z2_vec = [z2.^i for i in 0:size(hist_data, 2)-1]
    return sum((z1_vec*z2_vec') .* hist_data)
end


function hist_gf3d(
    hist_data::AbstractArray{<:Real, 3},
    z1::AbstractVector,
    z2::AbstractVector,
    z3::AbstractVector
)
    Nx = size(hist_data, 1)
    Ny = size(hist_data, 2)
    Nz = size(hist_data, 3)

    G = zeros(Float64, length(z1), length(z2), length(z3))

    for i in 1:Nx
        for j in 1:Ny
            for k in 1:Nz
                p = hist_data[i, j, k]

                if p != 0
                    G .+=
                        p .*
                        reshape(z1 .^ (i - 1), :, 1, 1) .*
                        reshape(z2 .^ (j - 1), 1, :, 1) .*
                        reshape(z3 .^ (k - 1), 1, 1, :)
                end
            end
        end
    end

    return G
end


function counts_to_pgf2d(
    U_counts::AbstractVector{<:Integer},
    S_counts::AbstractVector{<:Integer},
    z1::AbstractVector,
    z2::AbstractVector
)
    joint_prob_matrix = counts_to_joint_prob2d(U_counts, S_counts)

    SSA_PGF = vec(hist_gf2d(joint_prob_matrix, z1, z2)')

    return SSA_PGF
end


function counts_to_pgf3d(
    U_counts::AbstractVector{<:Integer},
    S_counts::AbstractVector{<:Integer},
    P_counts::AbstractVector{<:Integer},
    z1::AbstractVector,
    z2::AbstractVector,
    z3::AbstractVector
)
    joint_prob_matrix = counts_to_joint_prob3d(
        U_counts,
        S_counts,
        P_counts
    )

    SSA_PGF = vec(hist_gf3d(joint_prob_matrix, z1, z2, z3))

    return SSA_PGF
end
