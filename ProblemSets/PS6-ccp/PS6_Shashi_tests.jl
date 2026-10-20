#--------------------------------------------------
# Problem Set 6 - Unit Tests
# ECON 6343: Econometrics III

#--------------------------------------------------

using Test, Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, DataFramesMeta, CSV

include(joinpath(@__DIR__, "PS6_Shashi_source.jl"))

# Simulated wide bus data
function fake_wide_bus_data(N::Int; rng = MersenneTwister(1234))
    T  = 20
    df = DataFrame()
    for t in 1:T; df[!, Symbol("Y$t")]   = rand(rng, 0:1, N);        end
    for t in 1:T; df[!, Symbol("Odo$t")] = 10t .+ (1:N) .* 0.5;      end
    df.RouteUsage = collect(range(0.3, 1.2, length = N))
    df.Branded    = [isodd(i) ? 1 : 0 for i in 1:N]
    for t in 1:T; df[!, Symbol("Xst$t")] = fill(t, N);               end
    df.Zst = collect(1:N)
    return df
end

logistic(z) = 1 / (1 + exp(-z))

@testset "PS6 Shashi unit tests" begin

    #------------------------------------------------------------------
    @testset "create_grids" begin
        zval, zbin, xval, xbin, xtran = create_grids()
        @test zbin == length(zval) == 101
        @test xbin == length(xval) == 201
        @test size(xtran) == (xbin * zbin, xbin)
        @test all(xtran .>= -1e-12)
        @test all(isapprox.(sum(xtran, dims = 2), 1.0; atol = 1e-8))
        @test all(xtran[1:xbin, :][i, j] == 0 for i in 1:xbin, j in 1:xbin if j < i)
    end

    #------------------------------------------------------------------
    @testset "Q1: reshape_bus_data" begin
        N  = 4
        dfw = fake_wide_bus_data(N)
        df_long, Xstate, Zstate, Branded = reshape_bus_data(dfw)

        @test nrow(df_long) == N * 20
        @test all(c -> c in names(df_long),
                  ["bus_id", "time", "Y", "Odometer", "RouteUsage", "Branded"])
        @test df_long.bus_id == repeat(1:N, inner = 20)
        @test df_long.time   == repeat(1.0:20.0, outer = N)
        for i in 1:N, t in (1, 7, 20)
            row = df_long[(df_long.bus_id .== i) .& (df_long.time .== t), :]
            @test row.Y[1]          == dfw[i, Symbol("Y$t")]
            @test row.Odometer[1]   == dfw[i, Symbol("Odo$t")]
            @test row.RouteUsage[1] == dfw.RouteUsage[i]
            @test row.Branded[1]    == dfw.Branded[i]
        end
        @test size(Xstate) == (N, 20)
        @test Xstate[2, 5] == dfw.Xst5[2]
        @test Zstate == dfw.Zst
        @test Branded == dfw.Branded
    end

    #------------------------------------------------------------------
    @testset "Q3: construct_state_space" begin
        xval = [0.0, 1.0, 2.0]; zval = [0.5, 0.9]
        xbin, zbin = 3, 2
        xtran = zeros(xbin * zbin, xbin)
        sdf = construct_state_space(xbin, zbin, xval, zval, xtran)

        @test nrow(sdf) == size(xtran, 1)
        @test names(sdf) == ["Odometer", "RouteUsage", "Branded", "time"]
        @test sdf.Odometer   == [0, 1, 2, 0, 1, 2]
        @test sdf.RouteUsage == [0.5, 0.5, 0.5, 0.9, 0.9, 0.9]
        @test all(sdf.Branded .== 0) && all(sdf.time .== 0)
    end

    #------------------------------------------------------------------
    @testset "Q3: compute_future_values" begin
        rng = MersenneTwister(42)
        n   = 3000
        sim = DataFrame(Odometer = 10 .* rand(rng, n), RouteUsage = 0.25 .+ rand(rng, n),
                        Branded = rand(rng, 0:1, n), time = rand(rng, 1:20, n))
        sim.Y = Int.(rand(rng, n) .< logistic.(2 .- 0.2 .* sim.Odometer .+ 0.5 .* sim.Branded))
        simple_logit = glm(@formula(Y ~ Odometer + RouteUsage + Branded + time),
                           sim, Binomial(), LogitLink())

        xval = collect(0:0.5:5); zval = [0.4, 0.8, 1.2]
        xbin, zbin = length(xval), length(zval)
        xtran = zeros(xbin * zbin, xbin)
        sdf = construct_state_space(xbin, zbin, xval, zval, xtran)
        sdf_before = copy(sdf)

        T, β = 20, 0.9
        FV = compute_future_values(sdf, simple_logit, xtran, T, β)

        @test size(FV) == (xbin * zbin, 2, T + 1)
        @test all(FV[:, :, 1] .== 0)
        @test all(FV[:, :, T+1] .== 0)
        @test all(FV .>= 0)
        @test all(isfinite, FV)
        @test sdf == sdf_before

        t, b = 7, 1
        check = copy(sdf); check.time .= t; check.Branded .= b
        p0 = 1 .- Float64.(predict(simple_logit, check))
        @test FV[:, b+1, t] ≈ -β .* log.(p0)
    end

    #------------------------------------------------------------------
    @testset "Q3: compute_fvt1" begin
        rng   = MersenneTwister(7)
        xbin, zbin, T = 3, 2, 2
        xtran = rand(rng, xbin * zbin, xbin)
        xtran ./= sum(xtran, dims = 2)
        FV    = rand(rng, xbin * zbin, 2, T + 1)

        N       = 2
        Xstate  = [1 2; 3 1]
        Zstate  = [1, 2]
        B       = [0, 1]
        df_long = DataFrame(bus_id = repeat(1:N, inner = T), time = repeat(1:T, outer = N))

        fvt1 = compute_fvt1(df_long, FV, xtran, Xstate, Zstate, xbin, B)
        @test length(fvt1) == N * T

        expected = Float64[]
        for i in 1:N, t in 1:T
            row0 = (Zstate[i] - 1) * xbin + 1
            row1 = row0 + Xstate[i, t] - 1
            push!(expected, sum((xtran[row1, k] - xtran[row0, k]) * FV[row0+k-1, B[i]+1, t+1]
                                for k in 1:xbin))
        end
        @test fvt1 ≈ expected

        @test fvt1[1] == 0.0
        @test fvt1[4] == 0.0
    end

    #------------------------------------------------------------------
    @testset "Q3: estimate_structural_params and own MLE" begin
        rng = MersenneTwister(2026)
        n   = 20_000
        θ   = [1.5, -0.2, 0.8]
        df  = DataFrame(Odometer = 15 .* rand(rng, n), Branded = rand(rng, 0:1, n))
        off = 0.5 .* randn(rng, n)
        X   = hcat(ones(n), df.Odometer, df.Branded)
        df.Y = Int.(rand(rng, n) .< logistic.(X * θ .+ off))

        m = estimate_structural_params(df, off)
        @test length(coef(m)) == 3
        @test isapprox(coef(m), θ; atol = 0.15)
        @test !("fv" in names(df))

        θ_own, se_own, res = estimate_logit_offset(X, Float64.(df.Y), off)
        @test Optim.converged(res)
        @test isapprox(θ_own, coef(m); atol = 1e-4)
        @test isapprox(se_own, stderror(m); rtol = 1e-3)

        m_no = glm(@formula(Y ~ Odometer + Branded), df, Binomial(), LogitLink())
        @test !isapprox(coef(m_no), coef(m); atol = 1e-3)
    end

    #------------------------------------------------------------------
    @testset "Full pipeline on real bus data (needs internet)" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS5-ddc/busdata.csv"
        online = try
            HTTP.head(url); true
        catch
            false
        end

        if !online
            @info "No internet connection: skipping real-data tests"
            @test_skip false
        else
            df_long, Xstate, Zstate, Branded = load_and_reshape_data(url)
            N = size(Xstate, 1)
            @test nrow(df_long) == N * 20
            @test all(in((0, 1)), df_long.Y)

            flex = estimate_flexible_logit(df_long)
            @test length(coef(flex)) == 2^7
            phat = predict(flex)
            @test all(0 .<= phat .<= 1)

            zval, zbin, xval, xbin, xtran = create_grids()
            sdf  = construct_state_space(xbin, zbin, xval, zval, xtran)
            FV   = compute_future_values(sdf, flex, xtran, 20, 0.9)
            fvt1 = compute_fvt1(df_long, FV, xtran, Xstate, Zstate, xbin, Branded)
            @test length(fvt1) == nrow(df_long)
            @test all(isfinite, fvt1)

            est = estimate_structural_params(df_long, fvt1)
            θ̂ = coef(est)
            @test θ̂[2] < 0
            @test θ̂[3] > 0
        end
    end
end
