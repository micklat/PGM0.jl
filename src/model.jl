type DiscreteMRF
    n_vars :: Int
    n_states :: Vector{Int}
    factors :: Dict
    neighbours :: Vector{Vector{Int}}
end

function pairwise_factor(model::DiscreteMRF, i::Int, j::Int)
    if i>j
        return model.factors[(j,i)]'
    end
    return model.factors[(i,j)]
end

function get_factor(model::DiscreteMRF, i::Int)
    model.factors[(i,)]
end

typealias PairwiseMRF DiscreteMRF # TODO: remove this assumption

