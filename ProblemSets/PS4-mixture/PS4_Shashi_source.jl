#---------------------------------------------------
# ECON 6343: Econometrics III — Problem Set 4
# Multinomial and Mixed Logit Estimation
# Shashi
#
# Source file: all functions live here. The script and tests files include it.
#---------------------------------------------------

# NOTE: packages and lgwt.jl are loaded in the script/tests file before this file is included.

#---------------------------------------------------
# Data Loading Function
#---------------------------------------------------
function load_data()
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS4-mixture/nlsw88t.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    X = [df.age df.white df.collgrad]
    Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,
             df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
    y = df.occ_code
    return df, X, Z, y
end

#---------------------------------------------------
# Question 1: Multinomial Logit with Alternative-Specific Covariates
#---------------------------------------------------
function mlogit_with_Z(θ, X, Z, y)
    # θ = [α1, α2, ..., α21, γ]
    # α has K*(J-1) = 3*7 = 21 elements
    # γ is the coefficient on Z
    α = θ[1:end-1]  # first 21 elements
    γ = θ[end]      # last element

    K = size(X, 2)          # number of covariates in X (3)
    J = length(unique(y))   # number of choices (8)
    N = length(y)           # number of observations

    # Create choice indicator matrix
    bigY = zeros(N, J)
    for j = 1:J
        bigY[:, j] = y .== j
    end

    # Reshape α into K x (J-1) matrix, add zeros for normalized choice J
    bigα = [reshape(α, K, J-1) zeros(K)]

    # Initialize numerator/denominator (T allows ForwardDiff dual numbers)
    T = promote_type(eltype(X), eltype(θ))
    num = zeros(T, N, J)
    dem = zeros(T, N)

    # numerator for each choice j: exp(X_i*β_j + γ*(Z_ij - Z_iJ))
    for j = 1:J
        num[:, j] = exp.(X * bigα[:, j] .+ γ .* (Z[:, j] .- Z[:, J]))
    end

    # denominator (sum of numerators)
    dem = sum(num, dims=2)

    # probabilities
    P = num ./ dem

    # negative log-likelihood
    loglike = -sum(bigY .* log.(P))

    return loglike
end

#---------------------------------------------------
# Question 3a: Quadrature Practice
#---------------------------------------------------
function practice_quadrature()
    println("=== Question 3a: Quadrature Practice ===")

    # Define standard normal distribution
    d = Normal(0, 1)

    # Get quadrature nodes and weights for 7 grid points
    nodes, weights = lgwt(7, -4, 4)

    # Verify integral of density equals 1
    integral_density = sum(weights .* pdf.(d, nodes))
    println("∫φ(x)dx = ", integral_density, " (should be ≈ 1)")

    # Verify expectation equals 0
    expectation = sum(weights .* nodes .* pdf.(d, nodes))
    println("∫xφ(x)dx = ", expectation, " (should be ≈ 0)")

    return integral_density, expectation
end

#---------------------------------------------------
# Question 3b: More Quadrature Practice
#---------------------------------------------------

function variance_quadrature()
    println("\n=== Question 3b: Variance using Quadrature ===")
    
    # Define N(0,2) distribution
    d = Normal(0, 2)
    σ = 2
    
    # Use quadrature to compute ∫x²f(x)dx with 7 points
    nodes7, weights7 = lgwt(7, -5*σ, 5*σ)
    variance_7pts = sum(weights7 .* (nodes7.^2) .* pdf.(d, nodes7))
    
    # Use quadrature to compute ∫x²f(x)dx with 10 points
    nodes10, weights10 = lgwt(10, -5*σ, 5*σ)  
    variance_10pts = sum(weights10 .* (nodes10.^2) .* pdf.(d, nodes10))
    
    println("Variance with 7 quadrature points: ", variance_7pts)
    println("Variance with 10 quadrature points: ", variance_10pts)
    println("True variance: $(σ^2)")
    
    # Comment on approximation quality:
    # With 7 points over [-10,10] the approximation is poor (≈ 3.27 vs. 4).
    # With 10 points it is very close (≈ 4.04). More points -> better approximation.
    println("Comment: 7 points is off (≈3.27); 10 points is close to the truth (≈4.04).")

    return variance_7pts, variance_10pts
end

#---------------------------------------------------
# Question 3c: Monte Carlo Practice  
#---------------------------------------------------

function practice_monte_carlo()
    println("\n=== Question 3c: Monte Carlo Integration ===")
    
    Random.seed!(1234)
    σ = 2
    d = Normal(0, σ)
    a, b = -5*σ, 5*σ
    
    # Monte Carlo integration function
    function mc_integrate(f, a, b, D)
        # ∫f(x)dx ≈ (b-a) * (1/D) * Σf(X_i) where X_i ~ U[a,b]
        draws = rand(D) * (b - a) .+ a  # uniform draws on [a,b]
        return (b - a) * mean(f.(draws))
    end
    
    results = Dict{Int, Tuple{Float64, Float64, Float64}}()

    # Test with different numbers of draws
    for D in [1_000, 1_000_000]
        println("\nWith D = $D draws:")
        
        # Variance: ∫x²f(x)dx  
        variance_mc = mc_integrate(x -> x^2 * pdf(d, x), a, b, D)
        println("MC Variance: ", variance_mc, " (true: $(σ^2))")
        
        # Mean: ∫xf(x)dx
        mean_mc = mc_integrate(x -> x * pdf(d, x), a, b, D)  
        println("MC Mean: ", mean_mc, " (true: 0)")
        
        # Density integral: ∫f(x)dx
        density_mc = mc_integrate(x -> pdf(d, x), a, b, D)
        println("MC Density integral: ", density_mc, " (true: 1)")

        results[D] = (variance_mc, mean_mc, density_mc)
    end

    # Comment: with D = 1,000 the answers are noisy (e.g. mean off by ~0.1);
    # with D = 1,000,000 they are all very close to 4, 0 and 1.
    println("\nComment: D = 1,000 is noisy; D = 1,000,000 is very close to the true values.")

    return results
end

#---------------------------------------------------
# Question 4: Mixed Logit with Quadrature (DO NOT RUN!)
#---------------------------------------------------

function mixed_logit_quad(theta, X, Z, y, nodes, weights)
    # Extract parameters
    # theta = [alpha1, ..., alpha21, mu_gamma, sigma_gamma]
    K = size(X, 2)
    J = length(unique(y))
    N = length(y)
    
    alpha = theta[1:(K*(J-1))]  # coefficients on X
    mu_gamma = theta[end-1]     # mean of gamma distribution
    sigma_gamma = theta[end]    # std dev of gamma distribution
    
    # Create choice indicator matrix
    bigY = zeros(N, J)
    for j = 1:J
        bigY[:, j] = y .== j
    end
    
    # Reshape alpha 
    bigAlpha = [reshape(alpha, K, J-1) zeros(K)]
    
    # Initialize integrated probabilities
    T = promote_type(eltype(X), eltype(theta))
    P_integrated = zeros(T, N, J)
    
    # For each quadrature point r:
    # 1. Transform node: gamma_r = mu_gamma + sigma_gamma * nodes[r]
    # 2. Compute choice probabilities for this gamma_r (like regular logit)
    # 3. Weight by quadrature weight and normal density
    # 4. Add to integrated probabilities
    for r in eachindex(nodes)
        gamma_r = mu_gamma + sigma_gamma * nodes[r]
        
        # Compute probabilities for this gamma_r
        num_r = zeros(T, N, J)
        for j = 1:J
            num_r[:,j] = exp.(X * bigAlpha[:,j] .+ gamma_r .* (Z[:,j] .- Z[:,J]))
        end
        dem_r = sum(num_r, dims=2)
        P_r = num_r ./ dem_r
        
        # Weight and add to integrated probabilities
        density_weight = weights[r] * pdf(Normal(0,1), nodes[r])
        P_integrated .+= P_r * density_weight
    end
    
    # Compute negative log-likelihood  
    loglike = -sum(bigY .* log.(P_integrated))
    
    return loglike
end

#---------------------------------------------------
# Question 5: Mixed Logit with Monte Carlo (DO NOT RUN!)
#---------------------------------------------------

function mixed_logit_mc(theta, X, Z, y, D)
    # Extract parameters (same as quadrature version)
    K = size(X, 2)
    J = length(unique(y))
    N = length(y)
    
    alpha = theta[1:(K*(J-1))]
    mu_gamma = theta[end-1]
    sigma_gamma = theta[end]
    
    # Create choice indicator matrix
    bigY = zeros(N, J)
    for j = 1:J
        bigY[:, j] = y .== j
    end
    
    bigAlpha = [reshape(alpha, K, J-1) zeros(K)]
    
    T = promote_type(eltype(X), eltype(theta))
    P_integrated = zeros(T, N, J)
    
    # For d = 1 to D:
    # 1. Draw gamma_d from N(mu_gamma, sigma_gamma^2)
    # 2. Compute choice probabilities for this draw
    # 3. Add to running average (P_integrated += P_d / D)
    
    # Fix the seed so every likelihood evaluation uses the same draws.
    # gamma_d = mu + sigma*randn() is the same as rand(Normal(mu, sigma)),
    # but it also works with ForwardDiff (automatic differentiation).
    Random.seed!(1234)
    
    for d = 1:D
        gamma_d = mu_gamma + sigma_gamma * randn()
        
        # Compute probabilities for this draw (same as regular logit)
        num_d = zeros(T, N, J)  
        for j = 1:J
            num_d[:,j] = exp.(X * bigAlpha[:,j] .+ gamma_d .* (Z[:,j] .- Z[:,J]))
        end
        dem_d = sum(num_d, dims=2)
        P_d = num_d ./ dem_d
        
        # Add to running average
        P_integrated .+= P_d / D
    end
    
    # Compute negative log-likelihood
    loglike = -sum(bigY .* log.(P_integrated))
    
    return loglike
end

#---------------------------------------------------
# Optimization Functions
#---------------------------------------------------

function optimize_mlogit(X, Z, y)
    K = size(X, 2)
    J = length(unique(y))
    
    # Starting values: K*(J-1) alphas + 1 gamma
    # Using multinomial logit estimates (PS3 Q1-style) to speed things up;
    # replace with your own PS3 Q1 estimates if they differ.
    # Random alternative: startvals = [2*rand(K*(J-1)).-1; 0.1]
    startvals = [ .0403744; .2439942; -1.57132;
                  .0433254; .1468556; -2.959103;
                  .1020574; .7473086; -4.12005;
                  .0375628; .6884899; -3.65577;
                  .0204543; -.3584007; -4.376929;
                  .1074636; -.5263738; -6.199197;
                  .1168824; -.2870554; -5.322248;
                  1.307477]
    
    # initialize the twice differentiable object (automatic differentiation)
    td = TwiceDifferentiable(theta -> mlogit_with_Z(theta, X, Z, y),
                             startvals, autodiff = Optim.ADTypes.AutoForwardDiff())
    
    result = optimize(td, startvals, LBFGS(), 
                     Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true))
        
    # evaluate the Hessian at the estimates
    H  = Optim.hessian!(td, result.minimizer)
    result_se = sqrt.(diag(inv(H)))
    return result.minimizer, result_se
end

function optimize_mixed_logit_quad(X, Z, y, theta_mle)
    K = size(X, 2)  
    J = length(unique(y))
    
    # Get quadrature nodes and weights
    nodes, weights = lgwt(7, -4, 4)
    
    # Starting values: K*(J-1) alphas + mu_gamma + sigma_gamma
    # use multinomial logit estimates (Q1) as starting values: [alphas; mu_gamma = gamma_hat; sigma_gamma = 1]
    startvals = [theta_mle; 1.0]
    
    # Set up optimization (DON'T ACTUALLY RUN - TOO SLOW!)
    # result = optimize(theta -> mixed_logit_quad(theta, X, Z, y, nodes, weights),
    #                  startvals, LBFGS(),
    #                  Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true);
    #                  autodiff = Optim.ADTypes.AutoForwardDiff())
    
    println("Mixed logit quadrature optimization setup complete (not executed)")
    return startvals  # Return starting values instead of running
end

function optimize_mixed_logit_mc(X, Z, y, theta_mle)
    K = size(X, 2)
    J = length(unique(y))
    
    D = 1000  # Number of Monte Carlo draws
    
    # Starting values: same as quadrature version (Q1 estimates)
    startvals = [theta_mle; 1.0]
    
    # Set up optimization (DON'T ACTUALLY RUN - TOO SLOW!)
    # result = optimize(theta -> mixed_logit_mc(theta, X, Z, y, D),
    #                  startvals, LBFGS(),
    #                  Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true);
    #                  autodiff = Optim.ADTypes.AutoForwardDiff())
    
    println("Mixed logit Monte Carlo optimization setup complete (not executed)")
    return startvals  # Return starting values instead of running
end

#---------------------------------------------------
# Question 6: Main Function
#---------------------------------------------------

function allwrap()
    println("=== Problem Set 4: Multinomial and Mixed Logit ===")
    
    # Load data
    df, X, Z, y = load_data()
    
    println("Data loaded successfully!")
    println("Sample size: ", size(X, 1))
    println("Number of covariates in X: ", size(X, 2))
    println("Number of alternatives: ", length(unique(y)))
    
    # Question 1: Estimate multinomial logit
    println("\n=== QUESTION 1: MULTINOMIAL LOGIT RESULTS ===")
    theta_hat_mle, theta_hat_se = optimize_mlogit(X, Z, y)
    println("Estimates: ", theta_hat_mle)
    println("Std. errors: ", theta_hat_se)
    alpha_hat = theta_hat_mle[1:end-1]
    gamma_hat = theta_hat_mle[end]
    println("γ̂ = ", gamma_hat, "  (s.e. = ", theta_hat_se[end], ")")
    
    # Question 2: Interpret gamma
    println("\n=== QUESTION 2: INTERPRETATION ===")
    println("In PS3, γ̂ was negative (≈ -0.09), implying a higher expected wage in an")
    println("occupation made it LESS likely to be chosen, which is counterintuitive.")
    println("Now γ̂ = $(round(gamma_hat, digits=3)) is positive: a higher expected log wage in an")
    println("occupation raises the probability of choosing it. So yes, it makes more sense now.")
    
    # Question 3: Practice with quadrature and Monte Carlo
    practice_quadrature()
    variance_quadrature() 
    practice_monte_carlo()
    
    # Question 4: Mixed logit with quadrature (setup only)
    println("\n=== QUESTION 4: MIXED LOGIT QUADRATURE (SETUP) ===")
    optimize_mixed_logit_quad(X, Z, y, theta_hat_mle)
    
    # Question 5: Mixed logit with Monte Carlo (setup only)  
    println("\n=== QUESTION 5: MIXED LOGIT MONTE CARLO (SETUP) ===")
    optimize_mixed_logit_mc(X, Z, y, theta_hat_mle)
    
    println("\n=== ALL ANALYSES COMPLETE ===")
end
