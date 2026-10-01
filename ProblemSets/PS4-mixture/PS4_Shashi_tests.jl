#---------------------------------------------------
# ECON 6343: Econometrics III — Problem Set 4
# Shashi
# Tests file: reads in the source code, then runs simple unit tests.
#---------------------------------------------------
using Test, ForwardDiff, Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables, Distributions

include("lgwt.jl")
include("PS4_Shashi_source.jl")

# small fake data set (no download needed): N = 16 obs, K = 3, J = 8
Random.seed!(42)
N, K, J = 16, 3, 8
X = randn(N, K)
Z = randn(N, J)
y = repeat(1:J, 2)            # every choice appears, so length(unique(y)) = 8

@testset "PS4 unit tests" begin

    # Test 1: multinomial logit — with all parameters = 0 every choice has prob 1/J,
    # so the negative log-likelihood must equal N*log(J)
    θ0 = zeros(K*(J-1) + 1)
    @test mlogit_with_Z(θ0, X, Z, y) ≈ N * log(J)

    # Test 2: quadrature practice — ∫φ(x)dx ≈ 1 and ∫xφ(x)dx ≈ 0
    integral_density, expectation = practice_quadrature()
    @test isapprox(integral_density, 1.0; atol = 0.01)
    @test isapprox(expectation, 0.0; atol = 1e-8)

    # Test 3: mixed logit (Monte Carlo) — with sigma_gamma = 0 every draw is gamma = mu,
    # so it must collapse to the plain multinomial logit
    θ = [0.1 * randn(K*(J-1)); 0.5]
    @test mixed_logit_mc([θ; 0.0], X, Z, y, 50) ≈ mlogit_with_Z(θ, X, Z, y)

end
