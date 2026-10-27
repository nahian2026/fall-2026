################################################################################
# Problem Set 7 - Script
# ECON 6343: Econometrics III
################################################################################

using Random, LinearAlgebra, Statistics, Distributions, Optim, DataFrames, CSV, HTTP, GLM, FreqTables, ADTypes, ForwardDiff

cd(@__DIR__)
include("PS7_Shashi_source.jl")

# Questions 1, 2, 3, 5
@time main()

################################################################################
# Question 4: SMM.jl example (Lecture 9, slide #21)
################################################################################

import Pkg
if Base.find_package("SMM") === nothing
    try
        Pkg.add("SMM")
    catch
        try
            Pkg.add(url="https://github.com/floswald/SMM.jl")
        catch
            println("Could not install SMM.jl; using fallback for Question 4.")
        end
    end
end

smm_loaded = try
    @eval using SMM
    true
catch
    println("SMM.jl failed to load; using fallback for Question 4.")
    false
end

function smm_example()
    MA = SMM.parallelNormal()  # may take a few minutes
    dc = SMM.history(MA.chains[1])
    dc = dc[dc[!, :accepted] .== true, :]
    println(describe(dc))
    return dc
end

if smm_loaded
    smm_example()
else
    smm_normal_fallback()
end


# Findings
# Q1: GMM with an identity weighting matrix reproduces the closed-form OLS estimates.
# Q2: Just-identified GMM matches MLE; over-identified GMM (NLLS) is close but not identical, and its objective is not globally concave, so results can depend on starting values.
# Q3: Both simulation methods (inverse CDF and Gumbel) recover parameters close to the true β.
# Q4: SMM.jl would not install on Julia 1.12; a direct SMM fallback recovers means close to (-1, 1).
# Q5: SMM estimates stay close to MLE, since the simulated objective is a step function and LBFGS makes little progress.