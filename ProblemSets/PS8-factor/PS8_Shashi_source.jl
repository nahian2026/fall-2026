#==================================================================================
# Problem Set 8 - SOURCE FILE (functions only)
# ECON 6343: Econometrics III, Prof. Tyler Ransom
# Author: Shashi
==================================================================================#

ASVAB_COLS = [:asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK]

#==================================================================================
# Question 1: Load data and estimate base regression
==================================================================================#

"""
    load_data(url::String)

Load the NLSY dataset from the given URL and return as a DataFrame.
If `url` is instead a path to a local file (e.g. "nlsy.csv"), the local file is read.
"""
function load_data(url::String)
    if isfile(url)
        return CSV.read(url, DataFrame)
    end
    return CSV.read(HTTP.get(url).body, DataFrame)
end

"""
    estimate_base_regression(df::DataFrame)

Estimate the baseline wage regression without ASVAB scores.
Model: logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr
"""
function estimate_base_regression(df::DataFrame)
    return lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr), df)
end

#==================================================================================
# Question 2: Compute correlations among ASVAB scores
==================================================================================#

"""
    compute_asvab_correlations(df::DataFrame)

Compute the correlation matrix for the six ASVAB test scores.
Returns a DataFrame (6x6) whose columns (and the `test` column) name the ASVAB tests.
"""
function compute_asvab_correlations(df::DataFrame)
    # Extract the six ASVAB columns (by name) as a matrix
    asvabs = df[:, ASVAB_COLS]
    # Compute correlation matrix using cor()
    correlation = cor(Matrix{Float64}(asvabs))
    # Return as a formatted DataFrame
    cordf = DataFrame(
        cor1 = correlation[:,1],
        cor2 = correlation[:,2],
        cor3 = correlation[:,3],
        cor4 = correlation[:,4],
        cor5 = correlation[:,5],
        cor6 = correlation[:,6]
    )
    rename!(cordf, String.(ASVAB_COLS))
    insertcols!(cordf, 1, :test => String.(ASVAB_COLS))
    return cordf
end

#==================================================================================
# Question 3: Regression with all ASVAB scores
==================================================================================#

"""
    estimate_full_regression(df::DataFrame)

Estimate wage regression including all six ASVAB scores.
"""
function estimate_full_regression(df::DataFrame)
    return lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + asvabAR + asvabCS + asvabMK + asvabNO + asvabPC + asvabWK), df)
end

#==================================================================================
# Question 4: PCA regression
==================================================================================#

"""
    estimate_pca_regression(df::DataFrame)

Perform PCA on ASVAB scores and include first principal component in regression.
Note: MultivariateStats wants a J×N matrix (features × observations), hence the transpose.
"""
function estimate_pca_regression(df::DataFrame)
    # Extract ASVAB matrix and transpose to J×N
    asvabs = df[:, ASVAB_COLS]
    asvabMat = Matrix{Float64}(asvabs)'

    # Fit PCA model with maxoutdim=1
    M = fit(PCA, asvabMat; maxoutdim=1)

    # Transform data to get principal component (1×N)
    asvabPCA = MultivariateStats.transform(M, asvabMat)

    # Add PC to dataframe (reshape 1×N to a vector)
    df = @transform(df, :asvabPCA = asvabPCA[:])

    println("Correlation between grad4yr and asvabPCA:")
    println(cor(Matrix{Float64}(df[:, [:grad4yr, :asvabPCA]])))

    # Run regression with PC included
    return lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + asvabPCA), df)
end

#==================================================================================
# Question 5: Factor Analysis regression
==================================================================================#

"""
    estimate_factor_regression(df::DataFrame)

Perform Factor Analysis on ASVAB scores and include first factor in regression.
ASVAB columns are selected BY NAME so a PCA column (if present) is never included.
"""
function estimate_factor_regression(df::DataFrame)
    # Extract ASVAB matrix and transpose to J×N
    asvabs = df[:, ASVAB_COLS]
    asvabMat = Matrix{Float64}(asvabs)'

    # Fit Factor model with maxoutdim=1
    M = fit(FactorAnalysis, asvabMat; maxoutdim=1)

    # Transform data to get the factor scores (1×N)
    asvabFactor = MultivariateStats.transform(M, asvabMat)

    # Add factor to dataframe
    df = @transform(df, :asvabFactor = asvabFactor[:])

    println("Correlation between grad4yr and asvabFactor:")
    println(cor(Matrix{Float64}(df[:, [:grad4yr, :asvabFactor]])))

    # Run regression with factor included
    return lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + asvabFactor), df)
end

#==================================================================================
# Question 6: Full factor model with MLE
==================================================================================#

"""
    prepare_factor_matrices(df::DataFrame)

Returns
- `X`: wage covariates (N×7: black, hispanic, female, schoolt, gradHS, grad4yr, constant)
- `y`: log wage (N)
- `Xfac`: measurement covariates (N×4: black, hispanic, female, constant)
- `asvabs`: the 6 ASVAB scores (N×6)
"""
function prepare_factor_matrices(df::DataFrame)
    # X matrix for wage equation (constant at end)
    X = [Matrix{Float64}(df[:, [:black, :hispanic, :female, :schoolt, :gradHS, :grad4yr]]) ones(nrow(df))]

    # y (logwage)
    y = Vector{Float64}(df.logwage)

    # Xfac matrix for measurement equations (constant at end)
    Xfac = [Matrix{Float64}(df[:, [:black, :hispanic, :female]]) ones(nrow(df))]

    # asvabs matrix
    asvabs = Matrix{Float64}(df[:, ASVAB_COLS])

    return X, y, Xfac, asvabs
end

"""
    factor_model(θ, X, Xfac, Meas, y, R)

Negative log-likelihood of the factor model.

θ ordering: [vec(α) (L×J) ; β (K) ; γ (J) ; δ (1) ; σ (J+1)]

  asvab_j = Xfac*α_j + γ_j*ξ + ε_j,   ε_j ~ N(0, σ_j²)
  logwage = X*β     + δ*ξ   + ε,     ε   ~ N(0, σ_{J+1}²)
  ξ ~ N(0,1), integrated out with Gauss-Legendre quadrature on [-5,5].
"""
function factor_model(θ::Vector{T}, X::Matrix, Xfac::Matrix, Meas::Matrix,
                     y::Vector, R::Integer) where T<:Real

    # Get dimensions
    K = size(X, 2)      # Number of covariates in wage equation
    L = size(Xfac, 2)   # Number of covariates in measurement equations
    J = size(Meas, 2)   # Number of ASVAB tests
    N = length(y)       # Number of observations

    # Unpack parameters from θ vector
    α = reshape(θ[1:J*L], L, J)            # L×J
    β = θ[J*L+1:J*L+K]                     # K
    γ = θ[J*L+K+1:J*L+K+J]                 # J
    δ = θ[J*L+K+J+1]                       # scalar
    σ = θ[end-J:end]                       # J+1

    # Quadrature nodes (ξ) and weights (ω)
    ξ, ω = lgwt(R, -5, 5)

    # Initialize likelihood storage
    like  = zeros(T, N)
    Mlike = zeros(T, N, J)

    # Linear predictions (do not depend on ξ)
    Xα = Xfac*α     # N×J
    Xβ = X*β        # N

    # Loop over quadrature points
    for r in 1:R
        # Measurement equations
        for j = 1:J
            Mres = Meas[:,j] .- Xα[:,j] .- γ[j] * ξ[r]
            sdj  = sqrt(σ[j]^2)
            Mlike[:,j] = (1 ./ sdj) .* pdf.(Normal(0, 1), Mres ./ sdj)
        end

        # Wage equation
        Yres  = y .- Xβ .- δ * ξ[r]
        sdy   = sqrt(σ[end]^2)
        Ylike = (1 ./ sdy) .* pdf.(Normal(0, 1), Yres ./ sdy)

        # Update likelihood: weight × ∏_j Mlike × Ylike × φ(ξ_r)
        like .+= ω[r] .* vec(prod(Mlike; dims = 2)) .* Ylike .* pdf(Normal(0, 1), ξ[r])
    end

    # Return negative sum of log-likelihoods
    return -sum(log.(like))
end

"""
    get_start_vals(df::DataFrame)

Starting values: OLS for α (each ASVAB on Xfac) and β (wage on X),
0.5 for γ and δ (all-zero loadings are a stationary point), 0.5 for σ's.
"""
function get_start_vals(df::DataFrame)
    X, y, Xfac, asvabs = prepare_factor_matrices(df)
    J = size(asvabs, 2)
    α0 = vec(Xfac \ asvabs)        # L×J, vectorized column by column
    β0 = X \ y
    γ0 = 0.5 .* ones(J)
    δ0 = 0.5
    σ0 = 0.5 .* ones(J + 1)
    return vcat(α0, β0, γ0, δ0, σ0)
end

"""
    run_estimation(df::DataFrame, start_vals::Vector; R=25, show_trace=true)

MLE via Newton with BackTracking line search. Returns (θ̂, se, loglike).
R = 25 quadrature points.
"""
function run_estimation(df::DataFrame, start_vals::Vector; R::Integer=25, show_trace::Bool=true)
    # Prepare data matrices
    X, y, Xfac, asvabs = prepare_factor_matrices(df)

    # Create TwiceDifferentiable objective
    td = TwiceDifferentiable(th -> factor_model(th, X, Xfac, asvabs, y, R),
                start_vals, autodiff = Optim.ADTypes.AutoForwardDiff())

    # Optimize using Newton method
    result = optimize(td, start_vals, Newton(linesearch = BackTracking()),
                      Optim.Options(g_tol = 1e-5, iterations = 100_000,
                                    show_trace = show_trace, show_every = 1))

    # Compute Hessian and standard errors
    θ̂ = Optim.minimizer(result)
    H  = Optim.hessian!(td, θ̂)
    se = sqrt.(abs.(diag(inv(H))))

    # Return estimates, standard errors, and log-likelihood
    return θ̂, se, -Optim.minimum(result)
end

"""
    format_results(θ::Vector, se::Vector, loglike::Real, asvabs::Matrix)

DataFrame with columns: equation, variable, coefficient, std_error
(last row reports the log-likelihood).
"""
function format_results(θ::Vector, se::Vector, loglike::Real, asvabs::Matrix)
    J = size(asvabs, 2)
    asvab_names = String.(ASVAB_COLS[1:J])
    xfac_names  = ["black", "hispanic", "female", "constant"]
    x_names     = ["black", "hispanic", "female", "schoolt", "gradHS", "grad4yr", "constant"]

    equation = String[]
    variable = String[]
    # α (vectorized column by column: all L coefficients for test 1, then test 2, ...)
    for j in 1:J, v in xfac_names
        push!(equation, asvab_names[j]); push!(variable, v)
    end
    # β
    for v in x_names
        push!(equation, "logwage"); push!(variable, v)
    end
    # γ
    for j in 1:J
        push!(equation, asvab_names[j]); push!(variable, "factor loading (gamma)")
    end
    # δ
    push!(equation, "logwage"); push!(variable, "factor loading (delta)")
    # σ
    for j in 1:J
        push!(equation, asvab_names[j]); push!(variable, "sigma")
    end
    push!(equation, "logwage"); push!(variable, "sigma")

    # σ enters the likelihood as sqrt(σ^2), so report its absolute value
    coef = copy(θ)
    coef[end-J:end] .= abs.(coef[end-J:end])

    out = DataFrame(equation = equation, variable = variable,
                    coefficient = round.(coef, digits=4), std_error = round.(se, digits=4))
    push!(out, ("(model)", "log-likelihood", round(loglike, digits=4), NaN))
    return out
end

#==================================================================================
# Main execution function
==================================================================================#

"""
    main(url)

Run all questions (1-6) in order and print the results and written answers.
"""
function main(url::String = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS8-factor/nlsy.csv")

    println("="^80)
    println("Problem Set 8: Factor Models and Dimension Reduction  --  Shashi")
    println("="^80)

    # Load data
    println("\nLoading data...")
    df = load_data(url)
    println("Data loaded successfully. Dimensions: ", size(df))

    # Question 1
    println("\n" * "="^80)
    println("Question 1: Base Regression (without ASVAB)")
    println("="^80)
    est = estimate_base_regression(df)
    show(stdout, MIME("text/plain"), coeftable(est)); println()
    println("R² = ", round(r2(est), digits=4))

    # Question 2
    println("\n" * "="^80)
    println("Question 2: ASVAB Correlations")
    println("="^80)
    cordfout = compute_asvab_correlations(df)
    println(cordfout)
    println("""
    Answer Q2: The ASVAB scores are strongly positively correlated (roughly 0.4 to 0.8).
    The highest pairs are AR–MK (~0.80), PC–WK (~0.75), MK–PC and AR–PC (~0.71-0.72).
    All six tests seem to be picking up a common underlying ability.""")

    # Question 3
    println("\n" * "="^80)
    println("Question 3: Full Regression (with all ASVAB)")
    println("="^80)
    estfull = estimate_full_regression(df)
    show(stdout, MIME("text/plain"), coeftable(estfull)); println()
    println("R² = ", round(r2(estfull), digits=4))
    println("""
    Answer Q3: Yes, including them directly is problematic. Because the six scores are
    highly correlated (Q2), the regression suffers from multicollinearity: the individual
    ASVAB coefficients are imprecise, some are insignificant and one even has the
    'wrong' (negative) sign (asvabPC), so they cannot be interpreted separately.
    Adding them also shrinks the black and education coefficients relative to Q1,
    since ability was an omitted variable correlated with those regressors.""")

    # Question 4
    println("\n" * "="^80)
    println("Question 4: PCA Regression")
    println("="^80)
    estpca = estimate_pca_regression(df)
    show(stdout, MIME("text/plain"), coeftable(estpca)); println()
    println("""
    Answer Q4: The first principal component summarizes the six ASVAB scores in one
    regressor. Its coefficient is negative (about -0.05), but the sign of a PC is
    arbitrary: here the PC is negatively correlated with grad4yr (about -0.49), so it is
    oriented so that higher values mean LOWER test scores. Flipping the sign, higher
    ability raises wages, which matches Q3. The black and grad4yr coefficients shrink
    relative to Q1 (about -0.17 to -0.09 and 0.30 to 0.24), as in Q3, but now ability
    is controlled for with a single, precisely estimated regressor.""")

    # Question 5
    println("\n" * "="^80)
    println("Question 5: Factor Analysis Regression")
    println("="^80)
    estfac = estimate_factor_regression(df)
    show(stdout, MIME("text/plain"), coeftable(estfac)); println()
    println("""
    Answer Q5: The results are very close to the PCA regression. The factor is positively
    correlated with grad4yr (about 0.49), so its coefficient is positive (about 0.11):
    higher ability means higher wages. The other coefficients are almost the same as
    in Q4. The factor is on a different scale than the PC, so the size of its
    coefficient is not directly comparable.""")

    # Question 6
    println("\n" * "="^80)
    println("Question 6: Full Factor Model (MLE)")
    println("="^80)

    # Starting values: OLS for α, β; 0.5 for γ, δ, σ
    start_vals = get_start_vals(df)

    println("\nEstimating full factor model (Newton, Gauss-Legendre quadrature, R = 25)...")
    θ̂, se, loglike = run_estimation(df, start_vals)

    # Format and display results
    X, y, Xfac, asvabs = prepare_factor_matrices(df)
    results = format_results(θ̂, se, loglike, asvabs)
    show(stdout, results; allrows = true)
    println()
    CSV.write("PS8_Shashi_factor_results.csv", results)
    println("\nFactor-model results also saved to PS8_Shashi_factor_results.csv")
    println("""
    Answer Q6: Ability (ξ) is treated as a latent factor that drives both the ASVAB
    scores (loadings γ_j, all positive, about 0.55 to 0.82) and log wages (δ about 0.10).
    So a one standard deviation increase in ability raises wages by about 10%, holding
    schooling fixed. Unlike Q3-Q5, the model does not plug in a noisy estimated proxy for
    ability, so measurement error in the test scores does not bias the wage equation.
    The grad4yr (about 0.24) and gradHS (about 0.11) coefficients are close to Q4-Q5.
    The black coefficient (about -0.19) is close to Q1, not Q4-Q5 (about -0.09), because
    the model assumes ξ is independent of race: the racial gap in test scores goes into
    the α's, not into ability. Note the sign of (γ, δ) is only identified jointly:
    flipping all of them gives the same likelihood.""")

    println("\n" * "="^80)
    println("Analysis complete!")
    println("="^80)
    return nothing
end
