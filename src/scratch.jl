# this is a work area, something I can reload quickly without reloading the whole package.

module Scratch
import PGM0.structures: DiscreteBN, n_vars, n_states
import Iterators: columns

immutable Chunk
    observed::Vector{Int}
    observations::Matrix{Int} # rows correspond to variables, columns
                              # to examples.
end

type DiscreteBN_ESS # Expected sufficient statistics for discrete bayesian networks
    # Counts are the sufficient statistics of a DiscreteBN. For every factor
    # F, and for every assignment x to F's domain, we need a value that is the expectation
    # of the indicator that is 1 when the variables in the domain have the joint
    # assignment x. Thus, the ESS has the same shape as the BN's factors, which also hold
    # one scalar per assignment.
    structure :: BNStructure
    counts :: Vector{Array{Float64}}
    total_weight :: Float64
end

function update!(ss::DiscreteBN_ESS, x::Vector{Int}, model::DiscreteBN)
    ss.total_weight += 1
    for (i,factor) in enumerate(model.all_factors)
        idx = x[domain(factor)]
        ss.counts[i][idx...] += 1.0
    end
end

function suffstats(get_samples, model::DiscreteBN, chunks)
    counts = [zeros(size(factor.logp)) for factor in model.factors]
    ss = DiscreteBN_ESS(model.structure, counts, 0.)
    for chunk in chunks
        for i in 1:size(chunk,2)
            y = chunk.observations[:,i]
            for x in get_samples(model, chunk.observed, y)
                update!(ss, x, model)
            end
        end
    end
    ss
end

function default_sampler(model::DiscreteBN, observed::Vector{Int},
                         observatiosn::Vector{Int};
                         n_samples = 10, burn_in=100, interval = 5)
    clamps = Vars.Clamps(env(model), observed, observations)
    take(Task(Gibbs.sample(model, clamps, burn_in, interval)), n_samples)
end

function accelerated_em(structure::DiscreteMRF,
                        chunks::AbstractVector{Chunk};
                        max_iters = 2,
                        sampler = default_sampler)
    estimate = initial_mrf_estimate(structure, chunks)
    
    while true
        produce(estimate)

        # E step, returns a mean of per-sample moments
        et, covt = suffstats(sampler, estimate, chunks)
        assert(issym(covt))

        # M steps
        eta_during_inference = eta
        m_iter = 1
        projected_et = et
        while true
            estimate = fit_mle(structure, projected_et)
            if m_iter == max_iters; break; end
            m_iter += 1
            projected_et = project_ess(et, eta-eta_during_inference, covt)
        end
    end
end

end

