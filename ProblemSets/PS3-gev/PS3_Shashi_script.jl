# PS3 - ECON 6343 - Shashi
using Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables
include("PS3_Shashi_source.jl")

# Question 4: call main function (prints MNL and nested logit estimates)
allwrap()

#---------------------------------------------------
# Question 2: Interpretation of gamma hat (MNL)
#---------------------------------------------------
# gamma hat = -0.09 (approx.)
# gamma is the change in latent utility from a 1-unit increase in expected
# log wage (i.e., roughly a 100% increase in the expected wage) in an occupation,
# relative to the base occupation (Other). Because gamma is common across
# occupations, it tells us how sensitive occupational choice is to relative wages.
# The negative sign is counterintuitive (higher expected wages should make an
# occupation more attractive), which suggests the model is misspecified,
# e.g. omitted non-wage job characteristics that are correlated with wages.
