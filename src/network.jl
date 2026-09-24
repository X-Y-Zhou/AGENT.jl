"""Frozen network with standardization already folded into the first layer."""
struct AGENTModel{M,T}
    dim::Int
    z1::Vector{Float64}
    z2::Vector{Float64}
    z3::Union{Nothing,Vector{Float64}}
    W::Vector{Float64}
    network::M
    params::Vector{T}
    τ::Float64
    n_extra_params::Int
end
function load_trained_params(path::AbstractString; column::Symbol=:params)
    df=CSV.read(path,DataFrame)
    column in propertynames(df) || error("Missing parameter column $column")
    p=Float64.(df[!,column])
    all(isfinite,p) || error("Nonfinite network weights")
    p
end
function _build(dim; n=7, hidden_channels=dim==2 ? 80 : 160, params,
                precision=dim==2 ? Float64 : Float32, τ=1.0)
    n==7 || throw(ArgumentError("Released models use exactly seven nodes per axis"))
    precision in (Float32,Float64) || throw(ArgumentError("Unsupported precision"))
    nin=dim==2 ? 8 : 10; nout=n^dim
    expected=nin*hidden_channels+hidden_channels+nout*hidden_channels+nout
    length(params)==expected || throw(DimensionMismatch("Expected $expected network parameters"))
    # Same Flux reconstruction and precision conversion as the frozen loaders.
    core=Chain(Dense(nin,hidden_channels,tanh),Dense(hidden_channels,nout))
    precision==Float64 && (core=Flux.f64(core))
    _,re=Flux.destructure(core)
    rebuilt=re(params)
    net=Chain(rebuilt.layers...,x->Flux.sigmoid.(x))
    @assert eltype(net[1].weight)==precision
    if dim==2
        z1,z2,W=gauss_grid_2d(n); z3=nothing
    else
        z1,z2,z3,W=gauss_grid_3d(n)
    end
    AGENTModel(dim,z1,z2,z3,W,net,precision.(params),Float64(τ),dim==2 ? 1 : 3)
end
"""Build the 8→80→49 released model; supplied weights must have folded standardization."""
function build_agent_model2d(n::Int=7; hidden_channels::Int=80, params=nothing,
                             τ::Real=1.0, precision=Float64)
    weights=params===nothing ? load_trained_params(joinpath(@__DIR__,"../examples/parameters_trained/params_trained2d.txt")) : params
    _build(2;n=n,hidden_channels=hidden_channels,params=weights,precision=precision,τ=τ)
end

"""Build the 10→160→343 released model, preserving Float32 inference by default."""
function build_agent_model3d(n::Int=7; hidden_channels::Int=160, params=nothing,
                             τ::Real=1.0, precision=Float32)
    weights=params===nothing ? load_trained_params(joinpath(@__DIR__,"../examples/parameters_trained/params_trained3d.txt")) : params
    _build(3;n=n,hidden_channels=hidden_channels,params=weights,precision=precision,τ=τ)
end
"""Load bundled 2D/3D weights, or an explicitly supplied directory with model_config.toml."""
function load_agent_model(dim::Int; directory=nothing)
    dim in (2,3) || throw(ArgumentError("dim must be 2 or 3"))
    if directory === nothing
        return dim == 2 ? build_agent_model2d() : build_agent_model3d()
    end
    cf=TOML.parsefile(joinpath(directory,"model_config.toml"))
    cf["hidden"]=="tanh" && cf["output"]=="sigmoid" && !cf["residual"] && cf["standardization_folded"] ||
        error("This loader requires tanh, direct sigmoid, no residual, folded standardization")
    expected_order=dim==2 ? "z2 fastest, then z1" : "z1 fastest, then z2 then z3"
    cf["output_order"]==expected_order || error("Unexpected PGF ordering")
    precision=Dict("Float32"=>Float32,"Float64"=>Float64)[cf["inference_precision"]]
    file=dim==2 ? "params_trained.txt" : "params_trained3d.txt"
    _build(dim;hidden_channels=cf["width"],params=load_trained_params(joinpath(directory,file)),precision=precision)
end
function compute_full_pgf(model::AGENTModel,ps::AbstractVector)
    length(ps)==3+model.n_extra_params || throw(DimensionMismatch("Kinetic parameter count"))
    on,off,rho=ps[1:3]
    input=vcat(G_tele_delay.(on,off,rho,model.τ,model.z1),ps[4:end])
    vec(model.network(input))
end
