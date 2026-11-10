#==================================================================================
# Problem Set 8 - UNIT TESTS (Question 7)
# ECON 6343: Econometrics III, Prof. Tyler Ransom
# Author: Shashi
#
# Reads in the source code first, then tests each function.
==================================================================================#

using Random, LinearAlgebra, Statistics, Distributions
using Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM
using MultivariateStats, FreqTables, ForwardDiff, LineSearches
using Test

cd(@__DIR__)
include("lgwt.jl")
include("PS8_Shashi_source.jl")

datafile = isfile("nlsy.csv") ? "nlsy.csv" :
    "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS8-factor/nlsy.csv"

# Small simulated dataset from the factor model with known parameters
function simulate_factor_data(N::Int; seed::Int=1234)
    Random.seed!(seed)
    J = 6
    black    = Float64.(rand(N) .< 0.25)
    hispanic = Float64.((rand(N) .< 0.2) .* (1 .- black))
    female   = Float64.(rand(N) .< 0.5)
    schoolt  = round.(rand(N) .* 6, digits=1)
    gradHS   = Float64.(rand(N) .< 0.85)
    grad4yr  = Float64.((rand(N) .< 0.3) .* gradHS)
    ξ = randn(N)
    Xfac = [black hispanic female ones(N)]
    X    = [black hispanic female schoolt gradHS grad4yr ones(N)]
    α = repeat([-0.5, -0.3, -0.1, 0.2], 1, J)
    β = [-0.15, -0.05, -0.15, 0.01, 0.1, 0.25, 2.0]
    γ = [0.8, 0.5, 0.8, 0.6, 0.7, 0.7]
    δ = 0.1
    σ = [0.5, 0.8, 0.5, 0.8, 0.6, 0.6, 0.5]
    M = Xfac*α .+ ξ*γ' .+ randn(N, J) .* σ[1:J]'
    y = X*β .+ δ .* ξ .+ σ[end] .* randn(N)
    df = DataFrame(black=black, hispanic=hispanic, female=female, schoolt=schoolt,
                   gradHS=gradHS, grad4yr=grad4yr, logwage=y,
                   asvabAR=M[:,1], asvabCS=M[:,2], asvabMK=M[:,3],
                   asvabNO=M[:,4], asvabPC=M[:,5], asvabWK=M[:,6])
    θtrue = vcat(vec(α), β, γ, δ, σ)
    return df, θtrue
end

df = load_data(datafile)

@testset "PS8 Unit Tests" begin

    @testset "Q1: load_data and base regression" begin
        @test df isa DataFrame
        @test nrow(df) == 2438
        @test all(c -> c in names(df), ["logwage", "black", "hispanic", "female", "schoolt",
                                         "gradHS", "grad4yr", String.(ASVAB_COLS)...])
        est = estimate_base_regression(df)
        @test length(coef(est)) == 7               # intercept + 6 regressors
        # OLS coefficients equal the normal-equation solution
        X = [ones(nrow(df)) Matrix{Float64}(df[:, [:black, :hispanic, :female, :schoolt, :gradHS, :grad4yr]])]
        @test coef(est) ≈ X \ df.logwage
    end

    @testset "Q2: ASVAB correlations" begin
        cdf = compute_asvab_correlations(df)
        C = Matrix{Float64}(cdf[:, 2:end])
        @test size(C) == (6, 6)
        @test C ≈ C'                              # symmetric
        @test all(diag(C) .≈ 1.0)                 # ones on the diagonal
        @test all(-1 .<= C .<= 1)
        @test cdf.test == String.(ASVAB_COLS)
    end

    @testset "Q3: full regression" begin
        est = estimate_full_regression(df)
        @test length(coef(est)) == 13             # intercept + 6 + 6 ASVABs
        @test r2(est) >= r2(estimate_base_regression(df))   # adding regressors can't lower R²
    end

    @testset "Q4/Q5: PCA and factor-analysis regressions" begin
        ncol_before = ncol(df)
        estp = estimate_pca_regression(df)
        estf = estimate_factor_regression(df)
        @test length(coef(estp)) == 8
        @test length(coef(estf)) == 8
        @test ncol(df) == ncol_before             # df not modified by the functions
        @test all(isfinite, coef(estp)) && all(isfinite, coef(estf))
        # Works even if a PCA column is already in df (ASVABs are selected by name)
        df2 = copy(df); df2.asvabPCA = randn(nrow(df2))
        @test coef(estimate_factor_regression(df2)) ≈ coef(estf)
    end

    @testset "lgwt quadrature" begin
        x, w = lgwt(25, -5, 5)
        @test length(x) == 25 && length(w) == 25
        @test sum(w) ≈ 10.0                                   # integral of 1 on [-5,5]
        @test sum(w .* pdf.(Normal(), x)) ≈ 1.0 atol=1e-5     # N(0,1) density
        @test sum(w .* x.^2 .* pdf.(Normal(), x)) ≈ 1.0 atol=1e-4  # variance
    end

    @testset "Q6: prepare_factor_matrices" begin
        X, y, Xfac, asvabs = prepare_factor_matrices(df)
        N = nrow(df)
        @test size(X) == (N, 7)
        @test size(Xfac) == (N, 4)
        @test size(asvabs) == (N, 6)
        @test length(y) == N
        @test all(X[:, end] .== 1) && all(Xfac[:, end] .== 1)   # constant at the end
    end

    @testset "Q6: starting values and factor_model" begin
        X, y, Xfac, asvabs = prepare_factor_matrices(df)
        θ0 = get_start_vals(df)
        @test length(θ0) == 4*6 + 7 + 6 + 1 + 7               # 45 parameters
        nll = factor_model(θ0, X, Xfac, asvabs, y, 25)
        @test nll isa Real && isfinite(nll)
        # σ enters as sqrt(σ^2): flipping the sign of σ's doesn't change the likelihood
        θneg = copy(θ0); θneg[end-6:end] .*= -1
        @test factor_model(θneg, X, Xfac, asvabs, y, 25) ≈ nll
        # Sign symmetry of the factor: flipping all loadings (γ, δ) leaves likelihood unchanged
        θflip = copy(θ0); θflip[32:38] .*= -1
        @test factor_model(θflip, X, Xfac, asvabs, y, 25) ≈ nll rtol=1e-6
        # With zero loadings the model = independent normal regressions (closed form)
        θ0z = copy(θ0); θ0z[32:38] .= 0.0
        α = reshape(θ0z[1:24], 4, 6); β = θ0z[25:31]; σ = θ0z[39:45]
        ll = sum(logpdf.(Normal.(0, σ[j]), asvabs[:, j] .- Xfac*α[:, j]) for j in 1:6) .+
             logpdf.(Normal(0, σ[7]), y .- X*β)
        @test factor_model(θ0z, X, Xfac, asvabs, y, 25) ≈ -sum(ll) rtol=1e-5
        # Automatic differentiation works (needed for Newton)
        g = ForwardDiff.gradient(th -> factor_model(th, X, Xfac, asvabs, y, 9), θ0)
        @test length(g) == 45 && all(isfinite, g)
    end

    @testset "Q6: estimation converges and recovers parameters (simulated data)" begin
        dfsim, θtrue = simulate_factor_data(1000)
        θ0 = get_start_vals(dfsim)
        θ̂, se, loglike = run_estimation(dfsim, θ0; show_trace=false)
        @test length(θ̂) == length(θtrue) == length(se)
        @test all(isfinite, se) && all(se .> 0)
        @test isfinite(loglike)
        X, y, Xfac, asvabs = prepare_factor_matrices(dfsim)
        @test loglike >= -factor_model(θ0, X, Xfac, asvabs, y, 25)   # improved on start
        # Fix sign normalization (γ, δ only identified up to a common sign)
        s = sign(sum(θ̂[32:37]))
        @test maximum(abs.(θ̂[25:31] .- θtrue[25:31])) < 0.15         # β
        @test maximum(abs.(s .* θ̂[32:37] .- θtrue[32:37])) < 0.15     # γ
        @test abs(s * θ̂[38] - θtrue[38]) < 0.1                        # δ
        @test maximum(abs.(abs.(θ̂[39:45]) .- θtrue[39:45])) < 0.1     # σ

        res = format_results(θ̂, se, loglike, asvabs)
        @test res isa DataFrame
        @test nrow(res) == 46                                         # 45 params + loglike row
        @test names(res) == ["equation", "variable", "coefficient", "std_error"]
    end
end
