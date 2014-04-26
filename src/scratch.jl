# this is a work area, something I can reload quickly without reloading the whole package.

module Scratch
import PGM0.MRF: DiscreteMRF, n_vars, n_states
import Iterators: columns

immutable Chunk
    observed::Vector{Int}
    observations::Matrix{Int} # rows correspond to variables, columns
                              # to examples.
end

type Model
    bn :: 
end

type MRF_ESS
    # Counts are the sufficient statistics of a DiscreteMRF. For every factor F, and for
    # every assignment x to F's domain, we need a value that is the expectation of the
    # indicator that is 1 when the variables in the domain have the joint assignment
    # x. Thus, the ESS has the same shape as the MRF's factors, which also hold one scalar
    # per assignment.
    structure :: DiscreteMRF
    expected_counts :: Vector{Array{Float64}}
    total_weight :: Float64
end

function suffstats(model, chunks)
    ss = MRF_ESS(model., 
    for chunk in chunks
        for i in 1:size(chunk,2)
            
        end
    end
end

function accelerated_em(structure::DiscreteMRF,
                        chunks::AbstractVector{Chunk};
                        max_iters = 2)
    estimate = initial_mrf_estimate(structure, chunks)
    
    while true
        produce(estimate)

        # E step, returns a mean of per-sample moments
        et, covt = suffstats(estimate, chunks)
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

