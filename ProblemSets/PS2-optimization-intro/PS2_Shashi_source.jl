

function q1()
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 1
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # identically parameterize the function as:
    f(x) = -x[1]^4-10x[1]^3-2x[1]^2-3x[1]-2
    minusf(x) = x[1]^4+10x[1]^3+2x[1]^2+3x[1]+2 #fliping it cause Optim only does min
    startval = rand(1)   # random starting value, to guess you need a straring value
    result = optimize(minusf, startval, LBFGS()) #LBFGS is the algorithm
    println("optimization summary:", result) #for extra detail
    println("argmin (minimizer) is ",Optim.minimizer(result)[1])
    println("min is ",Optim.minimum(result))
    println("max is ",-Optim.minimum(result))
    return nothing
end 
#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 2
#:::::::::::::::::::::::::::::::::::::::::::::::::::
function ols(beta, X, y)
    ssr = (y.-X*beta)'*(y.-X*beta)
    return ssr
end
function q2()

    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    @show describe(df)
    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    y = df.married.==1
    beta_hat_ols = optimize(b -> ols(b, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
    println(beta_hat_ols.minimizer)

    bols = inv(X'*X)*X'*y
    @show bols
    df.white = df.race.==1
    bols_lm = lm(@formula(married ~ age + white + collgrad), df)
    
    # standard errors
    sigma2 = sum((y.-X*bols).^2)/(size(X,1)-size(X,2))
    vcov_bols =sigma2*inv(X'*X)
    @show [bols sqrt.(diag(vcov_bols))] # print out
    return nothing
end

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 3-4
#:::::::::::::::::::::::::::::::::::::::::::::::::::
function logit_like(beta, X, y)
    p = exp.(X*beta) ./ (1 .+ exp.(X*beta))
    neg_log_like = -sum(y.*log.(p) .+ (1 .-y).*log.(1 .-p))
    return neg_log_like
end
function q3_q4()

    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    @show describe(df)
    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    y = df.married.==1
    beta_hat_logit = optimize(b -> logit_like(b, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
    println(beta_hat_logit.minimizer)

    
    df.white = df.race.==1
    blogit_glm = glm(@formula(married ~ age + white + collgrad), df,Binomial(), LogitLink())
    @show blogit_glm
    return nothing
end

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 5
#:::::::::::::::::::::::::::::::::::::::::::::::::::

function mlogit(alpha, X, y)
    N = size(X,1) #no. of obs
    K = size(X, 2) # no. of covariates
    J = maximum(y) # no. of choice alternatives
    # unslice the vector
    alpha_mat = [reshape(alpha, K, J-1) zeros(K,1)] #add a column of zeros for the base category
    # made d matrix
    d = zeros(N, J)
    for j = 1:J
        d[:,j] .= y .== j 
    end    
    #make p matrixp
    num = zeros(N, J)
    for j = 1:J
        num[:,j] .= exp.(X*alpha_mat[:, j])
    end 
    dem = sum(num, dims=2) # CORRECTED: sum over alternatives for each person (row sums)
    p = num ./ dem
    #log-likelihood expression
    loglike = -sum(d .* log.(p))
    return loglike
end


function q5()

    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    @show describe(df)
    freqtable(df, :occupation) # note small number of obs in some occupations
    df = dropmissing(df, :occupation)
    df[df.occupation.==8 ,:occupation] .= 7
    df[df.occupation.==9 ,:occupation] .= 7
    df[df.occupation.==10,:occupation] .= 7
    df[df.occupation.==11,:occupation] .= 7
    df[df.occupation.==12,:occupation] .= 7
    df[df.occupation.==13,:occupation] .= 7
freqtable(df, :occupation) # problem solved
    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    y = df.occupation


    startval = rand(size(X,2)*(maximum(y)-1))
    alpha_hat_logit = optimize(b -> mlogit(b, X, y), startval, LBFGS(), Optim.Options(g_tol=1e-5, iterations=100_000, show_trace=true))
    println(alpha_hat_logit.minimizer)
    return nothing
end
