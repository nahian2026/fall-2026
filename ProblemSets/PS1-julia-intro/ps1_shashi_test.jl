# ==============================================================================
# ps1_nas_tests.jl
# ==============================================================================

using Test
using JLD, Random, LinearAlgebra, Statistics, CSV, DataFrames, FreqTables, Distributions

include("ps1_shashi_source.jl")

Random.seed!(1234)

@testset "Problem Set 1 Unit Tests" begin

    @testset "Question 1 Tests" begin
        A, B, C, D = q1()

        # Output dimensions
        @test size(A) == (10, 7)
        @test size(B) == (10, 7)
        @test size(C) == (5, 7)
        @test size(D) == (10, 7)

        # Range and zero-masking checks
        @test all(A .>= -5.0)
        @test all(A .<= 10.0)
        @test all(D[A .> 0] .== 0.0)
        @test all(D[A .<= 0] .== A[A .<= 0])

        # File creation checks
        @test isfile("matrixpractice.jld")
        @test isfile("firstmatrix.jld")
        @test isfile("Cmatrix.csv")
        @test isfile("Dmatrix.dat")
    end

    @testset "Question 2 Tests" begin
        A_dummy = rand(10, 7)
        B_dummy = rand(10, 7)
        C_dummy = rand(5, 7)
        D_dummy = rand(10, 7)

        X, β, Y = q2(A_dummy, B_dummy, C_dummy, D_dummy)

        # Output dimensions
        @test size(X) == (15_169, 6, 5)
        @test size(β) == (6, 5)
        @test size(Y) == (15_169, 5)

        # Variable structure
        @test all(X[:, 1, :] .== 1.0)
        @test all(val -> val == 0.0 || val == 1.0, X[:, 2, :])

        # Known beta formulas for t = 1
        @test β[1, 1] == 1.0
        @test β[2, 1] == 0.0
        @test β[3, 1] == -1.0
        @test β[5, 1] == 1.0

        # OLS estimation recovery with standard-error-aware tolerance
        for t in 1:size(X, 3)
            b_hat = X[:, :, t] \ Y[:, t]
            max_err = maximum(abs.(b_hat .- β[:, t]))
            @test max_err < 1.5
        end
    end

end