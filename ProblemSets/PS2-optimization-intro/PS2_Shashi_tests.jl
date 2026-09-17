# unit tests
using Test, Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, CSV, FreqTables
include("PS2_Shashi_source.jl")

X = [ones(10) collect(1.0:10.0)]

@testset "PS2 tests" begin
    # 1. OLS: SSR is zero when y is exactly X*beta
    beta = [1.0, 2.0]
    @test ols(beta, X, X*beta) ≈ 0.0 atol=1e-10

    # 2. Logit: at beta = 0, p = 0.5, so negative log-likelihood = N*log(2)
    y_bin = [0,1,0,1,0,1,0,1,0,1]
    @test logit_like(zeros(2), X, y_bin) ≈ 10*log(2)

    # 3. Multinomial logit: at alpha = 0, p = 1/J, so negative log-likelihood = N*log(J)
    y_multi = [1,2,3,1,2,3,1,2,3,3]   # J = 3
    @test mlogit(zeros(2*2), X, y_multi) ≈ 10*log(3)
end
