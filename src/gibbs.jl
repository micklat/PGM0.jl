module Gibbs

import PGM0: DiscreteMRF, PairwiseMRF

function transition!(m::DiscreteMRF, x::Vector)
    i = rand(1:model.n_vars)
    x[i] = sample1(m,x,i)
end

function sample1(m::PairwiseMRF, x::Vector, i::Int)
    log_measure = get_factor(model,i)
    for j in model.neighbours[i]
        theta = pairwise_factor(model,i,j)
        log_measure = log_measure + theta[:,x[j]]
    end
    a_posterior = exp(log_measure - log_sum_exp(log_measure))
    return StatsBase.wsample(a_posterior)
end

end
