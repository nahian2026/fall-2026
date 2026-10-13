# PS5 script file - Shashi
# ECON 6343, Problem Set 5
#
# Runs the full problem set: static logit (Q1-2) and dynamic estimation (Q3).
# Requires PS5_Shashi_source.jl and create_grids.jl in the same folder.
using Random, LinearAlgebra, Statistics, Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM

include(joinpath(@__DIR__, "PS5_Shashi_source.jl"))

main()
