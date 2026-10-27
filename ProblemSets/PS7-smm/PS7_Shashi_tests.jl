################################################################################
# Problem Set 7 - Unit Tests
# ECON 6343: Econometrics III
################################################################################

using Test, Random, LinearAlgebra, Statistics, Distributions, Optim, DataFrames, CSV, HTTP, GLM, FreqTables, ADTypes, ForwardDiff
cd(@__DIR__)
include("PS7_Shashi_source.jl")

@testset "PS7 Tests" begin

    @testset "load_data" begin
        println("Testing: load_data")
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
        df, X, y = load_data(url)
        @test size(X, 2) == 4
        @test size(X, 1) == length(y) == nrow(df)
        @test all(X[:, 1] .== 1)
        @test y ≈ log.(df.wage)
    end

    @testset "prepare_occupation_data" begin
        println("Testing: prepare_occupation_data")
        df = DataFrame(occupation = [1, 2, missing, 8, 13, 7, 3],
                       race       = [1, 2, 1, 1, 2, 1, 3],
                       age        = [30, 35, 40, 45, 38, 41, 36],
                       collgrad   = [0, 1, 0, 1, 1, 0, 0])
        df2, X, y = prepare_occupation_data(df)
        @test nrow(df2) == 6
        @test maximum(y) == 7
        @test size(X) == (6, 4)
        @test df2.white == [true, false, true, false, true, false]
    end

    @testset "ols_gmm" begin
        println("Testing: ols_gmm")
        Random.seed!(1)
        N = 500
        X = [ones(N) randn(N) rand(N)]
        β = [1.0, 2.0, -0.5]
        y = X*β
        @test ols_gmm(β, X, y) ≈ 0 atol=1e-10
        @test ols_gmm(β .+ 1, X, y) > 0

        y = X*β .+ 0.1 .* randn(N)
        res = optimize(b -> ols_gmm(b, X, y), zeros(3), LBFGS(), Optim.Options(g_tol=1e-8, iterations=1_000); autodiff = AutoForwardDiff())
        @test res.minimizer ≈ X \ y atol=1e-4
    end

    @testset "mlogit_mle" begin
        println("Testing: mlogit_mle")
        Random.seed!(2)
        N, J = 200, 4
        X = [ones(N) randn(N) randn(N)]
        y = rand(1:J, N)
        α0 = zeros(3*(J-1))
        @test mlogit_mle(α0, X, y) ≈ N*log(J)
        @test mlogit_mle(randn(9), X, y) > 0
    end

    @testset "mlogit_gmm (just-identified) matches MLE" begin
        println("Testing: mlogit_gmm (just-identified) matches MLE")
        Random.seed!(3)
        y, X = sim_logit_w_gumbel(2_000, 4)
        opts = Optim.Options(g_tol=1e-6, iterations=1_000)
        mle = optimize(a -> mlogit_mle(a, X, y), zeros(9), LBFGS(), opts; autodiff = AutoForwardDiff())
        gmm = optimize(a -> mlogit_gmm(a, X, y), mle.minimizer, LBFGS(), opts; autodiff = AutoForwardDiff())
        @test mlogit_gmm(mle.minimizer, X, y) ≈ 0 atol=1e-6
        @test gmm.minimizer ≈ mle.minimizer atol=1e-3
    end

    @testset "mlogit_gmm_overid" begin
        println("Testing: mlogit_gmm_overid")
        Random.seed!(4)
        N, J = 100, 3
        X = [ones(N) randn(N)]
        y = rand(1:J, N)
        α0 = zeros(2*(J-1))
        bigY = [y[i] == j for i in 1:N, j in 1:J]
        @test mlogit_gmm_overid(α0, X, y) ≈ N * sum((bigY .- 1/J).^2)
        @test mlogit_gmm_overid(randn(4), X, y) ≥ 0

        α1 = randn(4)
        G = zeros(4)
        mlogit_gmm_overid_grad!(G, α1, X, y)
        h = 1e-6
        Gfd = [(mlogit_gmm_overid(α1 .+ h .* (1:4 .== k), X, y) -
                mlogit_gmm_overid(α1 .- h .* (1:4 .== k), X, y)) / (2h) for k in 1:4]
        @test G ≈ Gfd rtol=1e-4
    end

    @testset "sim_logit" begin
        println("Testing: sim_logit")
        Random.seed!(5)
        y, X = sim_logit(10_000, 4)
        @test length(y) == 10_000
        @test size(X) == (10_000, 3)
        @test all(in(1:4), y)
        res = optimize(a -> mlogit_mle(a, X, y), zeros(9), LBFGS(), Optim.Options(g_tol=1e-6, iterations=1_000); autodiff = AutoForwardDiff())
        @test reshape(res.minimizer, 3, 3) ≈ true_beta(4)[:, 1:3] atol=0.2
    end

    @testset "sim_logit_w_gumbel" begin
        println("Testing: sim_logit_w_gumbel")
        Random.seed!(6)
        y, X = sim_logit_w_gumbel(10_000, 4)
        @test length(y) == 10_000
        @test all(in(1:4), y)
        res = optimize(a -> mlogit_mle(a, X, y), zeros(9), LBFGS(), Optim.Options(g_tol=1e-6, iterations=1_000); autodiff = AutoForwardDiff())
        @test reshape(res.minimizer, 3, 3) ≈ true_beta(4)[:, 1:3] atol=0.2

        y5, X5 = sim_logit_w_gumbel(1_000, 5)
        @test all(in(1:5), y5)
    end

    @testset "mlogit_smm_overid" begin
        println("Testing: mlogit_smm_overid")
        Random.seed!(7)
        y, X = sim_logit_w_gumbel(2_000, 4)
        αtrue = vec(true_beta(4)[:, 1:3])
        o1 = mlogit_smm_overid(αtrue, X, y, 50)
        o2 = mlogit_smm_overid(αtrue, X, y, 50)
        @test o1 == o2
        @test o1 ≥ 0
        @test o1 < mlogit_smm_overid(αtrue .+ 2, X, y, 50)
    end

    @testset "smm_normal_fallback" begin
        println("Testing: smm_normal_fallback")
        res = smm_normal_fallback(1_000, 50)
        @test res.minimizer ≈ [-1.0, 1.0] atol=0.15
        @test res.minimum < 1e-6
    end
end
