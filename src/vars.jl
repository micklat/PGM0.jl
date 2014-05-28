module Vars

export Named, Discrete, AnyVar, Env

abstract Variable

type Discrete <: Variable
    n_states :: Int
end

type Named{T<:Variable}
    var::T
    name::String
    id::Int
end

typealias Env Vector{Named} # a mapping from ids to Named vars

function set_ids!(env::Env)
    for i in 1:length(env); env[i].id = i; end
    env
end

type Clamp{T}
    var_id::Int
    val::T
end

type Clamps
    clamped::Vector{Clamp}
    unclamped::Vector{Int}

    function Clamps(env::Env, clamped::Vector{Clamp})
        new(clamped, setdiff([v.id for v in env], [c.var_id for c in clamped]))
    end
end

Clamps{T}(env::Env, ids::Vector{Int}, vals::Vector{T}) = Clamps(env, map(Clamp{T}, ids, vals))

end # Vars
