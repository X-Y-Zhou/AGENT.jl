module AGENT
using Optim, FastGaussQuadrature, Flux, CSV, DataFrames, DelimitedFiles
using Statistics, KernelDensity, Interpolations, TOML, Random, NLsolve
include("reduced_model.jl")
include("quadrature.jl")
include("pgf_utils.jl")
include("network.jl")
include("inference.jl")
include("feedback.jl")
include("toggle.jl")
include("capture_rate.jl")
include("nonlinear_feedback.jl")
include("data.jl")
export AGENTModel, load_agent_model, load_trained_params, build_agent_model2d, build_agent_model3d
export G_tele_delay, gauss_grid_2d, gauss_grid_3d, compute_full_pgf, int_dist, infer_parameters
export read_counts2d, read_counts3d, counts_to_joint_prob2d, counts_to_joint_prob3d
export hist_gf1d, hist_gf2d, hist_gf3d, counts_to_pgf2d, counts_to_pgf3d
export read_distributions, distributions_to_pgfs2d, read_pgf_matrix
export convert_LMA_feedback, infer_feedback_parameters, infer_nonlinear_feedback_parameters
export read_counts_toggle, counts_to_toggle_pgfs, convert_LMA_toggle, infer_toggle_parameters
export CaptureRateQuadrature, read_capture_rates, build_capture_rate_quadrature
export G_tele_delay_cp, compute_full_pgf_capture, int_dist_capture, infer_parameters_capture
end
