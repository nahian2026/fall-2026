# PS5 unit tests - Shashi
# ECON 6343, Problem Set 5
#
# Reads in the source code first, then runs unit tests on each function.
# Requires PS5_Shashi_source.jl and create_grids.jl in the same folder.

using Test
using Random, LinearAlgebra, Statistics, Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM
include(joinpath(@__DIR__, "PS5_Shashi_source.jl"))

Random.seed!(1234)

# Helper: build a small synthetic data tuple with the same structure as load_dynamic_data()
function make_test_data(; N=200, T=3, β=0.9, θ=[2.0, -0.15, 1.0])
    zval, zbin, xval, xbin, xtran = create_grids()
    Zstate = rand(1:zbin, N)
    Xstate = rand(1:40, N, T)               # mileage bins 0 to ~5 (10,000s of miles)
    X      = xval[Xstate]
    B      = rand(0:1, N)
    flow   = θ[1] .+ θ[2] .* X .+ θ[3] .* B
    Y      = Int.(rand(N, T) .< 1 ./ (1 .+ exp.(-flow)))
    return (Y=Y, X=X, B=B, Xstate=Xstate, Zstate=Zstate, N=N, T=T,
            xval=xval, xbin=xbin, zbin=zbin, xtran=xtran, β=β)
end

@testset "PS5 unit tests" begin

    #---------------------------------------------------------------------------
    # 1. Transition matrix is a Markov matrix
    #---------------------------------------------------------------------------
    @testset "create_grids: xtran is a Markov transition matrix" begin
        zval, zbin, xval, xbin, xtran = create_grids()

        # Grids match the problem set description
        @test zbin == length(0.25:0.01:1.25) == 101
        @test xbin == length(0:0.125:25) == 201
        @test size(xtran) == (zbin*xbin, xbin)      # square block for each z

        # (i) every entry is a probability (non-negative, up to rounding)
        @test all(xtran .>= -1e-12)
        @test all(xtran .<= 1 + 1e-12)

        # (ii) every row sums to one
        @test all(isapprox.(sum(xtran, dims=2), 1.0; atol=1e-10))

        # (iii) each z-block is itself a square Markov matrix
        for z in 1:zbin
            P = xtran[(z-1)*xbin+1:z*xbin, :]
            @test size(P, 1) == size(P, 2)
            @test all(isapprox.(sum(P, dims=2), 1.0; atol=1e-10))
        end

        # (iv) mileage never decreases: no mass below the diagonal
        for z in 1:zbin, x in 2:xbin
            @test all(abs.(xtran[x + (z-1)*xbin, 1:x-1]) .<= 1e-12)
        end

        # (v) Chapman-Kolmogorov: a two-step transition is still Markov
        P = xtran[1:xbin, :]
        @test all(isapprox.(sum(P*P, dims=2), 1.0; atol=1e-10))
    end

    #---------------------------------------------------------------------------
    # 2. Static logit estimation
    #---------------------------------------------------------------------------
    @testset "estimate_static_model" begin
        n = 20_000
        θtrue = [1.5, -0.2, 0.8]
        Odometer = 10 .* rand(n)
        Branded  = rand(0:1, n)
        p = 1 ./ (1 .+ exp.(-(θtrue[1] .+ θtrue[2] .* Odometer .+ θtrue[3] .* Branded)))
        df_test = DataFrame(Y = Float64.(rand(n) .< p), Odometer = Odometer, Branded = Branded)

        res = estimate_static_model(df_test)
        @test !isnothing(res)
        @test length(coef(res)) == 3
        @test isapprox(coef(res), θtrue; atol=0.15)   # recovers true parameters
    end

    #---------------------------------------------------------------------------
    # 3. Future value (backward recursion)
    #---------------------------------------------------------------------------
    @testset "compute_future_value!" begin
        d = make_test_data()
        θ = [2.0, -0.15, 1.0]
        FV = zeros(d.zbin*d.xbin, 2, d.T+1)
        compute_future_value!(FV, θ, d)

        @test size(FV) == (d.zbin*d.xbin, 2, d.T+1)
        @test all(FV[:, :, d.T+1] .== 0)       # terminal condition
        @test all(isfinite.(FV))
        @test all(FV[:, :, 1:d.T] .> 0)        # β·log(e^v0 + e^v1) > 0 here

        # In period T the future is zero, so FV = β·log(1 + exp(flow utility))
        for (z, x, b) in [(1, 1, 0), (50, 10, 1), (101, 201, 0)]
            row = x + (z-1)*d.xbin
            @test FV[row, b+1, d.T] ≈ d.β * log(1 + exp(θ[1] + θ[2]*d.xval[x] + θ[3]*b))
        end

        # With β = 0 there is no future value at all
        d0 = merge(d, (β = 0.0,))
        FV0 = zeros(d.zbin*d.xbin, 2, d.T+1)
        compute_future_value!(FV0, θ, d0)
        @test all(FV0 .== 0)
    end

    #---------------------------------------------------------------------------
    # 4. Dynamic log likelihood
    #---------------------------------------------------------------------------
    @testset "log_likelihood_dynamic" begin
        d = make_test_data()
        θ = [2.0, -0.15, 1.0]
        negll = log_likelihood_dynamic(θ, d)
        @test isfinite(negll)
        @test negll > 0                        # returns NEGATIVE log likelihood

        # With β = 0 the dynamic likelihood equals the static logit likelihood
        d0 = merge(d, (β = 0.0,))
        v = θ[1] .+ θ[2] .* d.X .+ θ[3] .* d.B
        ll_static = sum(d.Y .* v .- log.(1 .+ exp.(v)))
        @test log_likelihood_dynamic(θ, d0) ≈ -ll_static
    end

    #---------------------------------------------------------------------------
    # 5. Optimization wrapper (β = 0 so it is fast and must match GLM)
    #---------------------------------------------------------------------------
    @testset "estimate_dynamic_model (β = 0 matches GLM)" begin
        d0 = make_test_data(N=500, T=3, β=0.0)
        df_test = DataFrame(Y = vec(d0.Y), Odometer = vec(d0.X), Branded = repeat(d0.B, d0.T))
        θ_glm = coef(estimate_static_model(df_test))

        result = estimate_dynamic_model(d0, θ_start=zeros(3))
        @test Optim.converged(result)
        @test isapprox(result.minimizer, θ_glm; atol=1e-3)
    end

    #---------------------------------------------------------------------------
    # 6. Data loading (downloads the course data)
    #---------------------------------------------------------------------------
    @testset "data loading" begin
        df_long = load_static_data()
        @test nrow(df_long) == 1000*20
        @test all(in.([:bus_id, :time, :Y, :Odometer, :Branded], Ref(propertynames(df_long))))

        d = load_dynamic_data()
        @test size(d.Y) == (d.N, d.T) == (1000, 20)
        @test size(d.X) == size(d.Xstate) == (d.N, d.T)
        @test length(d.Zstate) == length(d.B) == d.N
        @test all(in.(d.Y, Ref((0, 1))))
        @test all(1 .<= d.Xstate .<= d.xbin)
        @test all(1 .<= d.Zstate .<= d.zbin)
        @test d.β == 0.9
    end
end
