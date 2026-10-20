#--------------------------------------------------
# Problem Set 6 - Source
# ECON 6343: Econometrics III
# Rust (1987) Bus Engine Replacement Model - CCP Estimation
# 
#--------------------------------------------------


# ================================================
# State grids (from PS5)
# ================================================

"""
    create_grids()

Build the discretized state space and mileage transition matrix used in PS5.

# Returns
- `zval`  : grid for route usage (x2)
- `zbin`  : number of route-usage bins
- `xval`  : grid for mileage (x1)
- `xbin`  : number of mileage bins
- `xtran` : (xbin*zbin) × xbin Markov transition matrix for mileage.
            Rows (z-1)*xbin+1 : z*xbin belong to route-usage bin z.
"""
function create_grids()

    function xgrid(theta, xval)
        N       = length(xval)
        xub     = vcat(xval[2:N], Inf)
        xtran1  = zeros(N, N)
        xtran1c = zeros(N, N)
        lcdf    = zeros(N)
        for i = 1:length(xval)
            xtran1[:, i]   = (xub[i] .>= xval) .* (1 .- exp.(-theta * (xub[i] .- xval)) .- lcdf)
            lcdf         .+= xtran1[:, i]
            xtran1c[:, i] .+= lcdf
        end
        return xtran1, xtran1c
    end

    zval   = collect([.25:.01:1.25]...)
    zbin   = length(zval)
    xval   = collect([0:.125:25]...)
    xbin   = length(xval)
    tbin   = xbin * zbin
    xtran  = zeros(tbin, xbin)
    xtranc = zeros(xbin, xbin, xbin)
    for z = 1:zbin
        xtran[(z-1)*xbin+1:z*xbin, :], xtranc[:, :, z] = xgrid(zval[z], xval)
    end

    return zval, zbin, xval, xbin, xtran
end


# ================================================
# QUESTION 1: Data Loading and Reshaping
# ================================================

"""
    reshape_bus_data(df::DataFrame)

Reshape the wide bus data (one row per bus) into a long panel
(one row per bus-period).

# Returns
- `df_long` : long panel with columns bus_id, time, Y, Odometer, RouteUsage, Branded
              sorted by (bus_id, time)
- `Xstate`  : N × T matrix of mileage-bin indices
- `Zstate`  : length-N vector of route-usage-bin indices
- `Branded` : length-N vector of the Branded dummy
"""
function reshape_bus_data(df::DataFrame)
    N = size(df, 1)
    T = 20

    # Create bus id variable
    df = @transform(df, :bus_id = 1:N)

    # Decision variable (Y1-Y20)
    ycols = [Symbol("Y$t") for t in 1:T]
    dfy = select(df, :bus_id, ycols, :RouteUsage, :Branded)
    dfy_long = DataFrames.stack(dfy, Not([:bus_id, :RouteUsage, :Branded]))
    rename!(dfy_long, :value => :Y)
    dfy_long = @transform(dfy_long, :time = kron(collect(1:T), ones(N)))
    select!(dfy_long, Not(:variable))

    # Odometer variable (Odo1-Odo20)
    xcols = [Symbol("Odo$t") for t in 1:T]
    dfx = select(df, :bus_id, xcols)
    dfx_long = DataFrames.stack(dfx, Not([:bus_id]))
    rename!(dfx_long, :value => :Odometer)
    dfx_long = @transform(dfx_long, :time = kron(collect(1:T), ones(N)))
    select!(dfx_long, Not(:variable))

    # Join and sort by bus and time
    df_long = leftjoin(dfy_long, dfx_long, on = [:bus_id, :time])
    sort!(df_long, [:bus_id, :time])
    disallowmissing!(df_long)

    # State variables in wide format
    Xstate  = Matrix(df[:, [Symbol("Xst$t") for t in 1:T]])
    Zstate  = Vector(df[:, :Zst])
    Branded = Vector(df[:, :Branded])

    return df_long, Xstate, Zstate, Branded
end


"""
    load_and_reshape_data(url::String)

Load the bus data and reshape to long panel format.
"""
function load_and_reshape_data(url::String)
    df = CSV.read(HTTP.get(url).body, DataFrame)
    return reshape_bus_data(df)
end


# ================================================
# QUESTION 2: Flexible Logit Estimation
# ================================================

"""
    estimate_flexible_logit(df::DataFrame)

Flexible logit of Y on a fully interacted set of Odometer, Odometer²,
RouteUsage, RouteUsage², Branded, time, time².
"""
function estimate_flexible_logit(df::DataFrame)
    flex_logit = glm(@formula(Y ~ Odometer * Odometer^2 * RouteUsage * RouteUsage^2 *
                                  Branded * time * time^2),
                     df, Binomial(), LogitLink())
    return flex_logit
end


# ================================================
# QUESTION 3: CCP-Based Estimation
# ================================================

"""
    construct_state_space(xbin, zbin, xval, zval, xtran)

Data frame with one row per grid point (x1, x2), in the same order as the
rows of `xtran`.
"""
function construct_state_space(xbin::Int, zbin::Int, xval::AbstractVector,
                               zval::AbstractVector, xtran::AbstractMatrix)
    state_df = DataFrame(
        Odometer   = kron(ones(zbin), xval),
        RouteUsage = kron(zval, ones(xbin)),
        Branded    = zeros(size(xtran, 1)),
        time       = zeros(size(xtran, 1))
    )
    return state_df
end


"""
    compute_future_values(state_df, flex_logit, xtran, T, β)

Evaluate −β·log p0 at every state (x1, x2, b, t) with the flexible-logit CCPs.

# Returns
- `FV::Array{Float64,3}` of size (rows of xtran) × 2 × (T+1).
  FV[:, b+1, t] holds −β log p0_t(x, b) for t = 2..T.
"""
function compute_future_values(state_df::DataFrame, flex_logit, xtran::AbstractMatrix,
                               T::Int, β::Float64)
    sdf = copy(state_df)
    FV  = zeros(size(xtran, 1), 2, T + 1)

    for t in 2:T
        for b in 0:1
            # Set current period and brand
            sdf.time    .= t
            sdf.Branded .= b
            # Probability of replacement
            p0 = 1 .- predict_in_chunks(flex_logit, sdf)
            FV[:, b+1, t] .= -β .* log.(max.(p0, eps()))
        end
        GC.gc()
    end

    return FV
end


"""
    predict_in_chunks(model, df; chunk = 1000)

Predicted probabilities from a fitted GLM, computed in blocks of rows.
"""
function predict_in_chunks(model, df::DataFrame; chunk::Int = 1000)
    n   = nrow(df)
    out = Vector{Float64}(undef, n)
    for s in 1:chunk:n
        e = min(s + chunk - 1, n)
        out[s:e] .= predict(model, df[s:e, :])
    end
    return out
end


"""
    compute_fvt1(df_long, FV, xtran, Xstate, Zstate, xbin, B)

Map the future-value array onto the observed data:

    FVT1[i,t] = (xtran[row1,:] − xtran[row0,:])' * FV[row0:row0+xbin-1, B[i]+1, t+1]

# Returns
- vector of length N*T in long panel format
"""
function compute_fvt1(df_long::DataFrame, FV::Array{Float64,3}, xtran::AbstractMatrix,
                      Xstate::AbstractMatrix, Zstate::AbstractVector, xbin::Int,
                      B::AbstractVector)
    N = size(Xstate, 1)
    T = size(Xstate, 2)
    @assert N == length(unique(df_long.bus_id)) "Xstate rows must equal number of buses"
    @assert nrow(df_long) == N * T "df_long must have N*T rows"

    FVT1 = zeros(N, T)
    for i in 1:N
        row0 = (Zstate[i] - 1) * xbin + 1
        for t in 1:T
            row1 = row0 + Xstate[i, t] - 1
            FVT1[i, t] = (xtran[row1, :] .- xtran[row0, :]) ⋅ FV[row0:row0+xbin-1, B[i]+1, t+1]
        end
    end

    # Convert to long format
    fvt1_long = FVT1'[:]
    return fvt1_long
end


"""
    estimate_structural_params(df_long, fvt1)

Estimate θ by a logit of Y on Odometer and Branded with the future-value term
entered as an offset (coefficient restricted to 1).
"""
function estimate_structural_params(df_long::DataFrame, fvt1::AbstractVector)
    df_long = @transform(df_long, :fv = fvt1)

    theta_hat = glm(@formula(Y ~ Odometer + Branded),
                    df_long,
                    Binomial(),
                    LogitLink(),
                    offset = df_long.fv)
    return theta_hat
end


# ================================================
# QUESTION 3(d): Binary logit with offset
# ================================================

log1pexp_stable(z) = z > 0 ? z + log1p(exp(-z)) : log1p(exp(z))

"""
    logit_offset_negloglike(θ, X, y, offset)

Negative log likelihood of a binary logit with index X*θ + offset.
"""
function logit_offset_negloglike(θ, X, y, offset)
    xb = X * θ .+ offset
    return -sum(y .* xb .- log1pexp_stable.(xb))
end


"""
    estimate_logit_offset(X, y, offset)

MLE of a binary logit with an offset whose coefficient is fixed at 1.
Returns (θ̂, standard errors, Optim result).
"""
function estimate_logit_offset(X::AbstractMatrix, y::AbstractVector, offset::AbstractVector)
    X      = Matrix{Float64}(X)
    y      = Vector{Float64}(y)
    offset = Vector{Float64}(offset)

    f(θ) = logit_offset_negloglike(θ, X, y, offset)
    function g!(G, θ)
        p = 1 ./ (1 .+ exp.(-(X * θ .+ offset)))
        G .= -X' * (y .- p)
        return G
    end

    θ0  = zeros(size(X, 2))
    res = optimize(f, g!, θ0, LBFGS(), Optim.Options(g_tol = 1e-8, iterations = 1_000))
    θ̂   = Optim.minimizer(res)

    # Standard errors
    p  = 1 ./ (1 .+ exp.(-(X * θ̂ .+ offset)))
    H  = X' * (X .* (p .* (1 .- p)))
    se = sqrt.(diag(inv(H)))

    return θ̂, se, res
end


# ================================================
# MAIN WRAPPER FUNCTION
# ================================================

"""
    main()

Run the CCP estimation and print results.
"""
function main()
    println("="^60)
    println("Rust (1987) Bus Engine Replacement Model - CCP Estimation")
    println("="^60)

    β = 0.9   # discount factor
    T = 20    # number of periods

    # Step 1: Load and reshape data
    println("\nStep 1: Loading and reshaping data...")
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS5-ddc/busdata.csv"
    df_long, Xstate, Zstate, Branded = load_and_reshape_data(url)
    println("Long panel has $(nrow(df_long)) rows ($(size(Xstate,1)) buses × $(size(Xstate,2)) periods)")

    # Step 2: Flexible logit
    println("\nStep 2: Estimating flexible logit...")
    flex_logit = estimate_flexible_logit(df_long)
    println("Flexible logit estimated with $(length(coef(flex_logit))) coefficients.")

    # Step 3a: State transition matrices
    println("\nStep 3a: Constructing state transition matrices...")
    zval, zbin, xval, xbin, xtran = create_grids()

    # Step 3b: State space
    println("\nStep 3b: Constructing state space...")
    state_df = construct_state_space(xbin, zbin, xval, zval, xtran)

    # Step 3c: Future values at every state
    println("\nStep 3c: Computing future values...")
    FV = compute_future_values(state_df, flex_logit, xtran, T, β)

    # Step 3d: Map future values to the data
    println("\nStep 3d: Mapping future values to data...")
    fvt1 = compute_fvt1(df_long, FV, xtran, Xstate, Zstate, xbin, Branded)

    # Step 3e: Structural parameters via GLM with offset
    println("\nStep 3e: Estimating structural parameters...")
    theta_hat_ccp_glm = estimate_structural_params(df_long, fvt1)

    println("\n" * "="^60)
    println("RESULTS: structural parameters (GLM with offset)")
    println("="^60)
    println(DataFrame(coeftable(theta_hat_ccp_glm)))

    # Question 3(d): Binary logit with offset
    X = hcat(ones(nrow(df_long)), df_long.Odometer, df_long.Branded)
    θ_own, se_own, _ = estimate_logit_offset(X, Float64.(df_long.Y), fvt1)
    println("\nRESULTS: structural parameters (own logit with offset)")
    println(DataFrame(term = ["θ0 (Intercept)", "θ1 Odometer", "θ2 Branded"],
                      estimate = θ_own, std_error = se_own))

    return (df_long = df_long, flex_logit = flex_logit, FV = FV, fvt1 = fvt1,
            theta_hat_ccp_glm = theta_hat_ccp_glm, theta_own = θ_own)
end
