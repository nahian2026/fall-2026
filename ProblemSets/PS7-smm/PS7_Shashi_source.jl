################################################################################
# Problem Set 7 - Source
# ECON 6343: Econometrics III
# GMM and SMM Estimation
################################################################################

################################################################################
# Data Loading and Preparation Functions
################################################################################

function load_data(url)
    df = CSV.read(HTTP.get(url).body, DataFrame)
    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad]
    y = log.(df.wage)
    return df, X, y
end

function prepare_occupation_data(df)
    df = dropmissing(df, :occupation)
    df[df.occupation.==8 ,:occupation] .= 7
    df[df.occupation.==9 ,:occupation] .= 7
    df[df.occupation.==10,:occupation] .= 7
    df[df.occupation.==11,:occupation] .= 7
    df[df.occupation.==12,:occupation] .= 7
    df[df.occupation.==13,:occupation] .= 7
    df.white = df.race .== 1
    X = [ones(size(df,1),1) df.age df.white df.collgrad]
    y = df.occupation
    return df, X, y
end

################################################################################
# Question 1: OLS via GMM
################################################################################

function ols_gmm(β, X, y)
    ŷ = X*β
    g = y .- ŷ
    J = g'*I*g
    return J
end

################################################################################
# Question 2: Multinomial Logit via MLE and GMM
################################################################################

function mlogit_mle(α, X, y)
    K = size(X, 2)
    J = length(α) ÷ K + 1
    N = length(y)

    bigY = zeros(N, J)
    for j = 1:J
        bigY[:,j] = y .== j
    end

    bigα = [reshape(α, K, J-1) zeros(K)]

    P = exp.(X*bigα) ./ sum.(eachrow(exp.(X*bigα)))

    loglike = -sum(bigY .* log.(P))

    return loglike
end

function mlogit_gmm(α, X, y)
    K = size(X, 2)
    J = length(α) ÷ K + 1
    N = length(y)

    bigY = zeros(N, J)
    for j = 1:J
        bigY[:,j] = y .== j
    end

    bigα = [reshape(α, K, J-1) zeros(K)]

    P = exp.(X*bigα) ./ sum.(eachrow(exp.(X*bigα)))

    g = zeros(eltype(α), (J-1)*K)
    for j = 1:(J-1)
        for k = 1:K
            g[(j-1)*K + k] = mean((bigY[:,j] .- P[:,j]) .* X[:,k])
        end
    end

    obj = N * g'g

    return obj
end

function mlogit_gmm_overid(α, X, y)
    K = size(X, 2)
    J = length(α) ÷ K + 1
    N = length(y)

    bigY = zeros(N, J)
    for j = 1:J
        bigY[:,j] = y .== j
    end

    bigα = [reshape(α, K, J-1) zeros(K)]
    P = exp.(X*bigα) ./ sum.(eachrow(exp.(X*bigα)))

    g = bigY[:] .- P[:]

    obj = N * g'g

    return obj
end

function mlogit_gmm_overid_grad!(G, α, X, y)
    K = size(X, 2)
    J = length(α) ÷ K + 1
    N = length(y)

    bigY = zeros(N, J)
    for j = 1:J
        bigY[:,j] = y .== j
    end

    bigα = [reshape(α, K, J-1) zeros(K)]
    P = exp.(X*bigα) ./ sum.(eachrow(exp.(X*bigα)))

    R = bigY .- P
    S = R .* P .- P .* sum(R .* P; dims=2)
    G .= vec(-2N .* (X' * S[:, 1:J-1]))

    return G
end

################################################################################
# Question 3: Simulate Data from Multinomial Logit
################################################################################

function true_beta(J=4)
    if J == 4
        β = hcat([1, -1, 0.5], [-2, 0.5, 0.3], [0, -0.5, 2], zeros(3))
    else
        β = hcat(rand(Uniform(-2, 2), 3, J-1), zeros(3))
    end
    return β
end

function sim_logit(N=100_000, J=4)
    X = hcat(ones(N), randn(N), 2 .+ 2 .* randn(N))

    β = true_beta(J)

    P = exp.(X*β) ./ sum.(eachrow(exp.(X*β)))

    draw = rand(N)

    Y = zeros(N)
    for j = 1:J
        Y += vec(sum(P[:,j:J]; dims=2) .> draw)
    end

    return Y, X
end

function sim_logit_w_gumbel(N=100_000, J=4)
    X = hcat(ones(N), randn(N), 2 .+ 2 .* randn(N))
    β = true_beta(J)

    ε = rand(Gumbel(0,1), N, J)

    Y = argmax.(eachrow(X*β .+ ε))

    return Y, X
end

################################################################################
# Question 5: Multinomial Logit via SMM
################################################################################

function mlogit_smm_overid(α, X, y, D)
    K = size(X, 2)
    J = length(α) ÷ K + 1
    N = length(y)

    bigY = zeros(N, J)
    for j = 1:J
        bigY[:,j] = y .== j
    end

    bigỸ = zeros(N, J)

    bigα = [reshape(α, K, J-1) zeros(K)]

    U = X*bigα
    Random.seed!(1234)
    for d = 1:D
        ε = rand(Gumbel(0,1), N, J)
        ỹ = argmax.(eachrow(U .+ ε))
        for j = 1:J
            bigỸ[:,j] .+= (ỹ .== j) * (1/D)
        end
    end

    g = bigY[:] .- bigỸ[:]

    obj = N * g'g

    return obj
end

################################################################################
# Question 4: Fallback for the SMM.jl example
################################################################################

# SMM.jl would not install on Julia 1.12, so this fallback (written by Claude)
# estimates the slide #21 model (Y1 = β01 + ε1, Y2 = β02 + ε2, ε ~ MVN(0, I),
# true β = (-1, 1)) by SMM directly. Estimates come out close to (-1, 1).
function smm_normal_fallback(N=1_000, D=100)
    Random.seed!(1234)
    Y = [-1.0 1.0] .+ randn(N, 2)
    μ_data = vec(mean(Y; dims=1))

    function obj(b)
        Random.seed!(42)
        μ_sim = zeros(2)
        for d = 1:D
            Ỹ = b' .+ randn(N, 2)
            μ_sim .+= vec(mean(Ỹ; dims=1)) ./ D
        end
        g = μ_data .- μ_sim
        return g'g
    end

    res = optimize(obj, zeros(2), NelderMead())
    println("SMM estimates of (β01, β02): ", round.(res.minimizer, digits=3), "  (true: [-1, 1])")
    return res
end

################################################################################
# Main Function - Question 6
################################################################################

function main()
    println("="^80)
    println("Problem Set 7: GMM and SMM Estimation")
    println("="^80)

    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df, X_wage, y_wage = load_data(url)
    df, X, y = prepare_occupation_data(df)

    #--------------------------------------------------------------------------
    # Question 1: OLS via GMM
    #--------------------------------------------------------------------------
    println("\n" * "="^80)
    println("Question 1: OLS Estimation via GMM")
    println("="^80)

    β_hat_gmm = optimize(b -> ols_gmm(b, X_wage, y_wage),
                         rand(size(X_wage,2)),
                         LBFGS(),
                         Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=false);
                         autodiff = AutoForwardDiff())

    β_ols = X_wage \ y_wage

    println("GMM estimates: ", β_hat_gmm.minimizer)
    println("OLS estimates: ", β_ols)

    #--------------------------------------------------------------------------
    # Question 2: Multinomial Logit via MLE and GMM
    #--------------------------------------------------------------------------
    println("\n" * "="^80)
    println("Question 2: Multinomial Logit via MLE and GMM")
    println("="^80)

    for j = 1:7
        tempname = Symbol(string("occ", j))
        df[:, tempname] = df.occupation .== j
    end

    svals = zeros(size(X,2), 7)
    for j = 1:7
        tempname = Symbol(string("occ", j))
        f = term(tempname) ~ term(:age) + term(:white) + term(:collgrad)
        svals[:,j] = coef(lm(f, df))
    end
    svals = svals[:,1:6] .- svals[:,7]
    svals = svals[:]

    # (a) MLE
    println("Estimating MLE...")
    α_hat_mle = optimize(a -> mlogit_mle(a, X, y),
                         svals,
                         LBFGS(),
                         Optim.Options(g_tol=1e-5, iterations=100_000, show_trace=false);
                         autodiff = AutoForwardDiff())

    # just-identified GMM (should match MLE)
    println("Estimating just-identified GMM...")
    α_hat_gmm_justid = optimize(a -> mlogit_gmm(a, X, y),
                                α_hat_mle.minimizer,
                                LBFGS(),
                                Optim.Options(g_tol=1e-5, iterations=100_000, show_trace=false);
                                autodiff = AutoForwardDiff())

    # (b) over-identified GMM, MLE starting values
    println("Estimating over-identified GMM (MLE start)...")
    α_hat_gmm_mle_start = optimize(a -> mlogit_gmm_overid(a, X, y),
                                   (G, a) -> mlogit_gmm_overid_grad!(G, a, X, y),
                                   α_hat_mle.minimizer,
                                   LBFGS(),
                                   Optim.Options(g_tol=1e-5, iterations=100_000, show_trace=false))

    # (c) over-identified GMM, random starting values
    println("Estimating over-identified GMM (random start)...")
    α_hat_gmm_random_start = optimize(a -> mlogit_gmm_overid(a, X, y),
                                      (G, a) -> mlogit_gmm_overid_grad!(G, a, X, y),
                                      rand(length(svals)),
                                      LBFGS(),
                                      Optim.Options(g_tol=1e-5, iterations=1_000, show_trace=false))

    println("MLE estimates:               ", round.(α_hat_mle.minimizer, digits=4))
    println("Just-identified GMM:         ", round.(α_hat_gmm_justid.minimizer, digits=4))
    println("GMM (MLE start):             ", round.(α_hat_gmm_mle_start.minimizer, digits=4))
    println("GMM (random start):          ", round.(α_hat_gmm_random_start.minimizer, digits=4))
    println("Objective (MLE start):       ", α_hat_gmm_mle_start.minimum)
    println("Objective (random start):    ", α_hat_gmm_random_start.minimum)
    # If the two starts give different estimates/objective values, the objective
    # is not globally concave (convex for minimization) and has local optima.

    #--------------------------------------------------------------------------
    # Question 3: Simulate and Recover Parameters
    #--------------------------------------------------------------------------
    println("\n" * "="^80)
    println("Question 3: Simulate Data and Recover Parameters")
    println("="^80)

    Random.seed!(1234)
    β_true = true_beta(4)

    ySim, XSim = sim_logit(100_000, 4)
    α_hat_sim = optimize(a -> mlogit_mle(a, XSim, ySim),
                         rand(9),
                         LBFGS(),
                         Optim.Options(g_tol=1e-5, iterations=100_000, show_trace=false);
                         autodiff = AutoForwardDiff())

    println("True β used in simulation:")
    display(β_true[:, 1:3])
    println("Estimated β from simulated data (inverse CDF):")
    display(reshape(α_hat_sim.minimizer, 3, 3))

    ySim_gumbel, XSim_gumbel = sim_logit_w_gumbel(100_000, 4)
    α_hat_sim_gumbel = optimize(a -> mlogit_mle(a, XSim_gumbel, ySim_gumbel),
                                rand(9),
                                LBFGS(),
                                Optim.Options(g_tol=1e-5, iterations=100_000, show_trace=false);
                                autodiff = AutoForwardDiff())

    println("Estimated β from simulated data (Gumbel):")
    display(reshape(α_hat_sim_gumbel.minimizer, 3, 3))

    #--------------------------------------------------------------------------
    # Question 5: Multinomial Logit via SMM
    #--------------------------------------------------------------------------
    println("\n" * "="^80)
    println("Question 5: Multinomial Logit via SMM")
    println("="^80)

    println("Estimating SMM (stops after 100 iterations or 5 minutes)...")
    α_hat_smm = optimize(th -> mlogit_smm_overid(th, X, y, 100),
                         α_hat_mle.minimizer,
                         LBFGS(),
                         Optim.Options(g_tol=1e-6, iterations=100, time_limit=300, show_trace=false))

    println("MLE estimates: ", round.(α_hat_mle.minimizer, digits=4))
    println("GMM estimates: ", round.(α_hat_gmm_mle_start.minimizer, digits=4))
    println("SMM estimates: ", round.(α_hat_smm.minimizer, digits=4))

    println("\n" * "="^80)
    println("Estimation Complete!")
    println("="^80)

    return β_hat_gmm, α_hat_mle, α_hat_gmm_mle_start, α_hat_gmm_random_start, α_hat_sim, α_hat_smm
end
