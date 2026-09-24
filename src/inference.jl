function int_dist(model::AGENTModel,ps::AbstractVector,target::AbstractVector,a::Real=1.0)
    a>0 || throw(ArgumentError("a must be positive"))
    length(target)==length(model.W) || throw(DimensionMismatch("PGF target length"))
    pred=compute_full_pgf(model,ps)
    # Preserve the reference objective, including its target-only constant.
    sum(model.W .* (pred.^(1+a) .- pred.^a .* target .* (1+1/a) .+ target/a))
end
function infer_parameters(model::AGENTModel,target::AbstractVector;
                          init=ones(3+model.n_extra_params),a::Real=1.0,
                          iterations::Int=model.dim==2 ? 1000 : 2000,
                          show_trace::Bool=false,g_tol::Real=model.dim==2 ? 1e-11 : 1e-20)
    length(init)==3+model.n_extra_params || throw(DimensionMismatch("Initial parameters"))
    all(x->isfinite(x)&&x>0,init) || throw(ArgumentError("Initial parameters must be positive and finite"))
    length(target)==length(model.W) || throw(DimensionMismatch("PGF target length"))
    start=time_ns()
    opt=Optim.optimize(q->int_dist(model,exp.(q),target,a),log.(Float64.(init)),
        Optim.Options(show_trace=show_trace,g_tol=g_tol,iterations=iterations))
    seconds=(time_ns()-start)/1e9
    (inferred_params=exp.(Optim.minimizer(opt)),opt=opt,time=seconds)
end
