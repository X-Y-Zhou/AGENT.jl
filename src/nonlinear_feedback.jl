"""Original nonlinear LMA: use the returned root without rejection or retry."""
function infer_nonlinear_feedback_parameters(model::AGENTModel,target::AbstractVector;
        init=ones(6),iterations::Int=2000,g_tol::Real=1e-20,show_trace::Bool=false)
    model.dim==3 || throw(ArgumentError("Requires a 3D model"))
    length(target)==343 || throw(DimensionMismatch("PGF target length"))
    length(init)==6 && all(x->isfinite(x)&&x>0,init) || throw(ArgumentError("Initial parameters"))
    calls=0;not_converged=0;nonpositive=0;nonfinite=0;exceptions=0;max_residual=0.0
    function predict(ps)
        son,sof,rho,dm,lambda,dp=ps
        function equations!(F,x)
            g,m,p,mg,pg,soff=x
            F[1]=son*(1-g)-soff*g
            F[2]=rho*g-dm*m
            F[3]=lambda*m-dp*p
            F[4]=rho*g+son*m-(son+soff+dm)*mg
            F[5]=lambda*mg+son*p-(son+soff+dp)*pg
            F[6]=soff-sof*pg/g
        end
        calls+=1
        result=try
            nlsolve(equations!,[1.,1,1,1,1,1])
        catch
            exceptions+=1;rethrow()
        end
        equivalent_soff=result.zero[6]
        not_converged+=!NLsolve.converged(result);nonpositive+=(equivalent_soff<=0)
        nonfinite+=!isfinite(equivalent_soff);max_residual=max(max_residual,result.residual_norm)
        compute_full_pgf(model,[son,equivalent_soff,rho,dm,lambda,dp])
    end
    function objective(q)
        ps=exp.(q);a=1.0
        # Two calls intentionally retained, including diagnostic call counts.
        sum(model.W.*(predict(ps).^(1+a).-predict(ps).^a.*target.*(1+1/a).+target/a))
    end
    start=time_ns()
    opt=Optim.optimize(objective,log.(Float64.(init)),Optim.Options(show_trace=show_trace,g_tol=g_tol,iterations=iterations))
    seconds=(time_ns()-start)/1e9
    (inferred_params=exp.(Optim.minimizer(opt)),opt=opt,time=seconds,
     inner_calls=calls,inner_not_converged=not_converged,inner_nonpositive_soff=nonpositive,
     inner_nonfinite_soff=nonfinite,inner_exceptions=exceptions,inner_max_residual=max_residual)
end
