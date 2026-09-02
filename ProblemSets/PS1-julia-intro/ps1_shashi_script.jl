using JLD, Random, LinearAlgebra, Statistics, CSV, DataFrames, FreqTables, Distributions
Random.seed!(1234)
include("ps1_shashi_source.jl")
A,B,C,D = q1()
X, β, Y = q2(A, B, C, D)