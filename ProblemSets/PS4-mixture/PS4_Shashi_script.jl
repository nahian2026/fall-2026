#---------------------------------------------------
# ECON 6343: Econometrics III — Problem Set 4
# Shashi
# Script file: loads packages, reads in the source code, and runs everything.
#---------------------------------------------------
using ForwardDiff, Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables, Distributions

# Include quadrature function (make sure lgwt.jl is in your working directory)
include("lgwt.jl")
include("PS4_Shashi_source.jl")

# call our functions
allwrap()

# New interpretation of γ̂: 1.3074772796390102
# Now, coefficient estimate says that if we increase expected log wage
# by 1 unit, then our utility for that occupation will be expected to
# increase by 1.307 units.
