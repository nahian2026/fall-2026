#--------------------------------------------------
# Problem Set 6 - Script
# ECON 6343: Econometrics III
# Rust (1987) Bus Engine Replacement Model - CCP Estimation
# 
#--------------------------------------------------

using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, DataFramesMeta, CSV

include(joinpath(@__DIR__, "PS6_Shashi_source.jl"))

@time results = main();

#--------------------------------------------------
# Writeup
#--------------------------------------------------
# CCP estimates (β = 0.9), s.e. in parentheses:
#   θ0 (Intercept) =  1.944  (0.033)
#   θ1 (Odometer)  = -0.149  (0.006)
#   θ2 (Branded)   =  1.125  (0.040)
# Higher mileage lowers the payoff of running, so replacement
# becomes more likely; branded buses are less likely to be replaced.
# Own logit with offset (Q3d) matches GLM exactly.
# Total run time: about 37 seconds.
#--------------------------------------------------
