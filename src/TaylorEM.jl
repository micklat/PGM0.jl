module TaylorEM
using Distributions
using PDMats
using PyPlot
using NumericExtensions
using Debug
typealias F64 Float64

function rand_soft_assignment(n_modes, n_samples)
    start_assignment = rand(1:n_modes, n_samples)
    q = [float64(start_assignment[j]==i) for i in 1:n_modes, j in 1:n_samples]
    assert(sum(q)==n_samples)
    assert(size(q)==(n_modes, n_samples))
    return q
end

normed1(x) = x ./ sum(x)

function mode_posterior(mixture, x)
    prior = mixture.probs
    components = mixture.components
    normed1([(prior[j]*pdf(components[j], x))::F64
             for j in 1:length(components)])
end

mog_stat_index(n_modes,n_dims,i,j) = n_modes + (i-1)*n_dims + j

function pack_mog_covs(cov_ii, cov_ij, cov_jj, n_modes, n_dims)
    n = n_modes + n_modes * n_dims
    c = zeros(F64, (n,n))

    # the first n_modes rows and columns correspond to the indicator Subsequently, every
    # n_dims rows and columns contain the product of a single indicator with each of the
    # observation's components.
    k(i,j) = mog_stat_index(n_modes,n_dims,i,j)
    
    tii = 1
    tiij = 1
    tiijj = 1
    for i1 in 1:n_modes
        for i2 in 1:i1
            c[i1,i2] = c[i2,i1] = cov_ii[tii]
            tii += 1
            for j1 in 1:n_dims
                (c[k(i1,j1),i2] =
                 c[i2, k(i1,j1)] =
                 c[k(i2,j1), i1] =
                 c[i1, k(i2,j1)] = cov_ij[tiij])
                tiij += 1
                for j2 in 1:j1
                    (c[k(i1,j1), k(i2,j2)] =
                     c[k(i2,j2), k(i1,j1)] =
                     c[k(i1,j2), k(i2,j1)] =
                     c[k(i2,j1), k(i1,j2)] = cov_jj[tiijj])
                    tiijj += 1
                end
            end
        end
    end
    return c
end

immutable MogESS
    inds :: Array{F64,1}
    x_inds :: Array{F64,2}
end

function mog_expectation(mixture::MixtureModel, x::Array)
    n_dims, n_samples = size(x)
    n_modes = length(mixture.components)
    e_inds = zeros(F64, (n_modes,))
    e_inds_xs = zeros(F64, (n_dims, n_modes))
    for m=1:n_samples
        xm = x[:,m]
        e_inds += qm = mode_posterior(mixture, xm)
        for i1=1:n_modes
            e_inds_xs[:,i1] += qm[i1]*xm
        end
    end
    MogESS(e_inds ./ n_samples, e_inds_xs ./ n_samples)
end

function mog_expectation_and_cov(mixture::MixtureModel, x::Array)
    n_dims, n_samples = size(x)
    n_modes = length(mixture.components)
    nii = sum(1:n_modes) # for each n_modes>=i1>=i2>=1
    nij = nii * n_dims # for each i1>=i2, 1<=j1<=n_dims
    njj = nii * sum(1:n_dims) # for each i1>=i2, j1>=j2
    n_covs = nii + nij + njj
    cov_ii = zeros(F64, (nii,))
    cov_ij = zeros(F64, (nij,))
    cov_jj = zeros(F64, (njj,))
    e_inds = zeros(F64, (n_modes,))
    e_inds_xs = zeros(F64, (n_dims, n_modes))
    for m=1:n_samples
        xm = x[:,m]
        e_inds += qm = mode_posterior(mixture, xm)
        tii = 1
        tij = 1
        tjj = 1
        for i1=1:n_modes
            e_inds_xs[:,i1] += qm[i1]*xm
            for i2=1:i1
                c_ii = qm[i1] * (int(i1==i2) - qm[i2])
                cov_ii[tii] += c_ii
                tii += 1
                for j1=1:n_dims
                    ciij = c_ii * xm[j1]
                    cov_ij[tij] += ciij
                    tij += 1
                    for j2=1:j1
                        cov_jj[tjj] += ciij * xm[j2]
                        tjj += 1
                    end
                end
            end
        end
    end
    cov_t = pack_mog_covs(cov_ii, cov_ij, cov_jj, n_modes, n_dims)
    return (MogESS(e_inds ./ n_samples, e_inds_xs ./ n_samples),
            cov_t ./ n_samples)
end

function initial_mog_model(x, n_modes, Σ)
    TGaussian = GenericMvNormal{typeof(Σ)}
    n_dims, n_samples = size(x)
    components = Array(TGaussian, n_modes)
    for i=1:n_modes
        μ = x[:,rand(1:n_samples)]
        components[i] = TGaussian(μ, Σ)
    end
    prior = repmat([1/n_modes], n_modes)
    return MixtureModel(components, prior)
end

function eta_from(mixture, Σ)
    invΣ = inv(Σ)
    n_dims = dim(mixture)
    n_modes = length(mixture.probs)
    k(i,j) = mog_stat_index(n_modes,n_dims,i,j)
    eta = zeros(F64, (n_modes + n_modes*n_dims,))
    for i in 1:n_modes
        μ_i = mixture.components[i].μ
        invΣμ_i = invΣ * μ_i
        eta[i] = log(mixture.probs[i]) - 0.5 * (μ_i' * invΣμ_i)[1]
        for j in 1:n_dims
            eta[k(i,j)] = invΣμ_i[j]
        end
    end
    eta
end

#@debug begin
function project_mog_ess(et, eta, eta_during_inference, covt, n_modes, n_dims, m_iter)
    Δη = eta - eta_during_inference
    Δτ = covt * Δη
    Δx_inds = reshape(Δτ[n_modes+1:end], (n_dims,n_modes))
    Δinds = Δτ[1:n_modes]
    decreasing_modes = Δinds .< 0
    if any(decreasing_modes)
        ## # we want: 0 <= new_inds[i] = inds[i] + α * Δinds[i]
        ## # -inds[i] <= α * Δinds[i], and since Δinds[i]<0, this implies
        ## # -inds[i] / Δinds[u] >= α, hence we take a minimum over the LHS:
        # α = minimum(-et.inds[decreasing_modes] ./ Δinds[decreasing_modes])
        # prevent any mode from losing more than 50% of its mass
        # 0.5 <= (inds[i] + α*Δinds[i]) / inds[i]
        # 0.5*inds[i] <= inds[i] + α*Δinds[i]
        # -inds[i] <= 2*α*Δinds[i]
        # -0.5*inds[i]/Δinds[i] >= α
        # hence we take the minimum of the LHS
        α = min(1.0, minimum(-0.5 .* et.inds[decreasing_modes] / Δinds[decreasing_modes]))
    else
        α = 1.0
    end
    #α *= 0.8^(m_iter-1)
    #print(α, ' ')
    projected_et = MogESS(et.inds + α*Δinds, et.x_inds + α*Δx_inds)
end
#end # @debug

MAX_M_ITER = 2

function mog_accelerated_em(x::Array{F64,2}, n_modes, Σ = nothing)
    n_dims, n_samples = size(x)
    if Σ === nothing; Σ = ScalMat(n_dims, 1.0); end
    TGaussian = GenericMvNormal{typeof(Σ)}
    mixture = initial_mog_model(x, n_modes, Σ)
    eta = eta_from(mixture, Σ)
    while true
        produce(mixture)
    
        # E step. Returns a mean of per-sample moments.
        et, covt = mog_expectation_and_cov(mixture, x)
        assert(issym(covt))
        
        # M step
        eta_during_inference = eta
        m_iter = 1
        projected_et = et
        #println(et)
        while true
            prior = projected_et.inds
            components = [TGaussian(projected_et.x_inds[:,mode] ./ prior[mode], Σ)
                          for mode in 1:n_modes]                          
            mixture = MixtureModel(components, prior)
            #println(m_iter, ": ", mixture.probs, ", ", [c.μ for c in mixture.components])
            eta = eta_from(mixture, Σ)
            
            if m_iter==MAX_M_ITER; break; end           
            m_iter+=1
            projected_et = project_mog_ess(et, eta, eta_during_inference, covt,
                              n_modes, n_dims, m_iter)
        end
    end
end

function mog_batch_em(x::Array{F64,2}, n_modes, Σ = nothing)
    n_dims, n_samples = size(x)
    if Σ === nothing; Σ = ScalMat(n_dims, 1.0); end
    TGaussian = GenericMvNormal{typeof(Σ)}
    mixture = initial_mog_model(x, n_modes, Σ)
    eta = eta_from(mixture, Σ)
    while true
        produce(mixture)
    
        # E step. Returns a mean of per-sample moments.
        et = mog_expectation(mixture, x)
        
        # M step
        prior = et.inds
        components = [TGaussian(et.x_inds[:,mode] ./ prior[mode], Σ) for mode in 1:n_modes]
        mixture = MixtureModel(components, prior)
    end
end

function test1(n=100, em_method=mog_accelerated_em, chart=false; scale = 0.4)
    # first, generate n samples from a mixture
    μs = [2.0 1;
          -11 3;
          -4 9]' * scale
    dim = size(μs,1)
    n_modes = size(μs, 2)
    Σ = PDMats.ScalMat(dim,1.0)
    mode_prior = [0.2, 0.1, 0.7]
    components = [IsoNormal(μs[:,i],1.0) for i in 1:n_modes]
    mixture = MixtureModel(components, mode_prior)
    x = rand(mixture, n)
    if chart
        plot(x[1,:], x[2,:], "k.")
        hold(true)
    end
    # trajectories[i] : n_estimates x dim    
    trajectories = {Array(F64, (0,dim)) for _ in 1:n_modes}
    last_est = nothing
    for est in @task em_method(x,n_modes)
        last_est = est
        d = 0.0
        for i in 1:n_modes
            trajectories[i] = [trajectories[i]; est.components[i].μ']
        end
        if (size(trajectories[1],1)>=2) &&
            (maximum([norm(trajectories[i][end,:]-trajectories[i][end-1,:],1)
                      for i in 1:n_modes])<0.005) ||
                          size(trajectories[1],1)>100
            break
        end
    end
    
    println(size(trajectories[1]))
    if chart
        for i in 1:n_modes
            plot(trajectories[i][:,1], trajectories[i][:,2], "*-")
        end
    end
    return mean([Distributions.logpdf(last_est, x[:,m]) for m in 1:n])
end

# this settings shows that the 2nd order algorithm is faster, with commit 6a7bd6
#function compare_methods(n=3000, seed=0, repetitions=20, chart=false)

function compare_methods(n=3000, repetitions=20; seed=0, chart=false)
    if chart
        PyPlot.figure(1)
        PyPlot.clf()
    end
    srand(seed)
    tic()
    mll = mean([test1(n, mog_accelerated_em, chart) for _ in 1:repetitions])
    @printf "2nd order batch: dt=%fs, mll=%f\n" toq() mll
    if chart; PyPlot.title("Taylor batch"); end

    if chart
        PyPlot.figure(2)
        PyPlot.clf()
    end
    srand(seed)
    tic()
    mll = mean([test1(n, mog_batch_em, chart) for _ in 1:repetitions])
    @printf "1st order batch: dt=%fs, mll=%f\n" toq() mll
    if chart; PyPlot.title("vanilla batch"); end
end

end # TaylorEM

