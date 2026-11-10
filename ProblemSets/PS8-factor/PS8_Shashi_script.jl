#==================================================================================
# Problem Set 8 - SCRIPT FILE
# ECON 6343: Econometrics III, Prof. Tyler Ransom
# Author: Shashi
#
# Loads packages, reads in the source code, and runs all questions.
# Needs lgwt.jl and PS8_Shashi_source.jl in the same folder.
# To save the output: julia PS8_Shashi_script.jl | Tee-Object PS8_Shashi_output.txt
==================================================================================#

# If needed, install packages once:
# using Pkg; Pkg.add(["MultivariateStats", "LineSearches", "Optim", "HTTP", "GLM",
#     "DataFrames", "DataFramesMeta", "CSV", "Distributions", "FreqTables", "ForwardDiff"])

using Random, LinearAlgebra, Statistics, Distributions
using Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM
using MultivariateStats, FreqTables, ForwardDiff, LineSearches

# Set working directory
cd(@__DIR__)

# Read in quadrature function and source code
include("lgwt.jl")
include("PS8_Shashi_source.jl")

# Use the local nlsy.csv if it is in this folder, otherwise download it
datafile = isfile("nlsy.csv") ? "nlsy.csv" :
    "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS8-factor/nlsy.csv"

# Run everything
main(datafile)
