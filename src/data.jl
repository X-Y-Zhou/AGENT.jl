"""Read Matrix <id> blocks without renormalizing or reordering their values."""
function read_distributions(path::AbstractString)
    matrices=Matrix{Float64}[];rows=Vector{Float64}[];ids=Int[]
    for line in eachline(path)
        isempty(strip(line)) && continue
        if startswith(line,"Matrix")
            push!(ids,parse(Int,split(line)[2]))
            if !isempty(rows)
                push!(matrices,permutedims(reduce(hcat,rows)));empty!(rows)
            end
        else
            push!(rows,parse.(Float64,split(line,',')))
        end
    end
    isempty(rows) || push!(matrices,permutedims(reduce(hcat,rows)))
    ids==collect(1:length(matrices)) || error("Missing, duplicate, or unordered matrix IDs")
    isempty(matrices) && error("No distributions found")
    all(p->all(isfinite,p)&&minimum(p)>=0&&abs(sum(p)-1)<1e-12,matrices) || error("Invalid empirical distributions")
    matrices
end
function distributions_to_pgfs2d(matrices,model::AGENTModel)
    model.dim==2 || throw(ArgumentError("Requires a 2D model"))
    hcat([vec(hist_gf2d(p,model.z1,model.z2)') for p in matrices]...)
end
function read_pgf_matrix(path::AbstractString,dim::Int;groups=nothing)
    dim in (2,3) || throw(ArgumentError("dim must be 2 or 3"))
    # Existing experiment .csv files are tab-delimited, with no header.
    data=reshape(readdlm(path),7^dim,:)
    all(isfinite,data) || error("Nonfinite observed PGF")
    groups===nothing || size(data,2)==groups || throw(DimensionMismatch("PGF group count"))
    data
end
