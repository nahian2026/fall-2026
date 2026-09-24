# PS3 - ECON 6343 - Shashi : simple unit tests
using Test, Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables
include("PS3_Shashi_source.jl")

# small fake data set (all 8 choices appear)
Random.seed!(1234)
N = 200
X = [rand(N) rand(N) rand(N)]
Z = randn(N, 8)
y = repeat(1:8, 25)
nest = [[1, 2, 3], [4, 5, 6, 7]]

@testset "PS3 tests" begin

    @testset "load_data" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS3-gev/nlsw88w.csv"
        df, Xd, Zd, yd = load_data(url)
        @test size(Xd, 2) == 3
        @test size(Zd, 2) == 8
        @test length(yd) == size(Xd, 1)
    end

    @testset "mlogit_with_Z" begin
        # all parameters zero -> every choice has prob 1/8
        @test mlogit_with_Z(zeros(22), X, Z, y) ≈ N * log(8)
        @test mlogit_with_Z(randn(22), X, Z, y) > 0
    end

    @testset "nested_logit_with_Z" begin
        # betas = 0, gamma = 0, lambdas = 1 -> prob 1/8
        @test nested_logit_with_Z([zeros(6); 1.0; 1.0; 0.0], X, Z, y, nest) ≈ N * log(8)
        # lambdas = 1 -> nested logit collapses to MNL with nest-level betas
        bWC, bBC, g = randn(3), randn(3), 0.3
        theta_mnl = [repeat(bWC, 3); repeat(bBC, 4); g]
        @test nested_logit_with_Z([bWC; bBC; 1.0; 1.0; g], X, Z, y, nest) ≈
              mlogit_with_Z(theta_mnl, X, Z, y)
    end

    @testset "optimizers" begin
        @test length(optimize_mlogit(X, Z, y)) == 22
        @test length(optimize_nested_logit(X, Z, y, nest)) == 9
    end
end
