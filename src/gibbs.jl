module Gibbs

import PGM0: DiscreteMRF
import Iterators: skip, everynth

function transition!(model::DiscreteMRF, x::Vector)
    i = rand(1:n_vars(model))
    x[i] = sample1(model,x,i)
end

function a_posterior(model::DiscreteMRF, x::Vector, i::Int)
    log_measure = zeros(i_n_states)
    i_n_states = n_states(model,i)
    for factor in model.factors[i]
        str = strides(factor.logp)
        offset = 1
        i_stride = 0
        for j in 1:length(factor.domain)
            v = factor.domain[j]
            if v==i
                i_stride = str[j]
                continue
            end
            offset += str[j] * (x[v]-1)
        end
        indices = offset + i_stride * (0:i_n_states-1)
        log_measure = log_measure .+ factor.logp[indices]
    end
    exp(log_measure - log_sum_exp(log_measure))
end

function sample1(model::DiscreteMRF, x::Vector, i::Int)
    post = a_posterior(model, x, i)
    return StatsBase.sample(StatsBase.WeightVec(post, 1.0))
end

unif_random_assignment(model::DiscreteMRF) = [rand(1:n_states(model,i)) for i in 1:n_vars(model)]

function produce_chain!(model::DiscreteMRF, x::Vector=nothing)
    if x === nothing; x = unif_random_assignment(model); end
    nv = n_vars(model)
    while true
        for i in 1:nv
            transition!(model,x)
            produce(x)
        end
    end
end

function sample(model::DiscreteMRF, burn_in::Int, interval::Int;
                start::Vector=nothing)
    chain = Task(produce_chain!(model, start))
    nv = n_vars(model)
    everynth(drop(chain, burn_in*nv), interval*nv)
end

end
