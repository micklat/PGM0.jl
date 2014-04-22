module Gibbs

import PGM0.MRF: DiscreteMRF
import PGM0.Vars: Clamps
import Iterators: skip, everynth

uniform_from{T}(a::Vector{T}) = a[rand(1:length(a))]

function transition!(model::DiscreteMRF, x::Vector, clamps::Clamps)
    i = uniform_from(clamps.unclamped)
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

function unif_random_assignment(model::DiscreteMRF, clamps::Clamps)
    x = zeros(Int, n_vars(model))
    for i in clamps.unclamped; x[i] = rand(1:n_states(model,i)); end
    for c in clamps.clamped; x[c.var_id] = c.val end
end

function produce_chain!(model::DiscreteMRF, clamps::Clamps, start::Vector=nothing)
    x = start
    if x === nothing; x = unif_random_assignment(model, clamps); end
    while true
        for i in clamps.unclamped
            x[i] = sample1(model, x, i)
            produce(x)
        end
    end
end

function sample(model::DiscreteMRF, clamps::Clamps, burn_in::Int, interval::Int;
                start::Vector=nothing)
    chain = Task(produce_chain!(model, clamps, start))
    n_unclamped = length(clamps.unclamped)
    everynth(drop(chain, burn_in*n_unclamped), interval*n_unclamped)
end

end
