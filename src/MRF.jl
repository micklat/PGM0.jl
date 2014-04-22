module Vars

export Named, Discrete, AnyVar, Env

abstract AnyVar

type Discrete <: AnyVar
    n_states :: Int
end

type Named
    var::AnyVar
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

module Factors
using PGM0.Vars
export Indexed

type Indexed{T}
    vars_env :: Vars.Env
    domain :: Vector{Int}
    logp :: Array{T} # conditional log-probability or log-density

    function Indexed(env::Vars.Env,
                     dom::AbstractVector{Int}, lp::Array{T};
                     sorted=false, check_n_states=true)
        n_dims = length(dom)
        assert(ndims(lp) == n_dims)
        if !sorted
            order = sort(1:n_dims, by=i->dom[i])
            dom = dom[order]
            lp = permutedims(lp, order)
        end
        ndom = [env[i] for i in dom]
        assert(all([dom[i-1] < dom[i] for i in 2:n_dims]),
            "mis-ordered domain or a duplicate variable")
        if check_n_states
            assert(all([size(lp,i)==env[dom[i]].var.n_states
                        for i in 1:n_dims]),
                   "n_states mismatch between the factor's domain and its value")
        end
        new(env, dom, lp)
    end
end

function +(f1::Indexed, f2::Indexed)
    assert(f1.vars_env === f2.vars_env)
    inputs = [f1,f2]
    shapes = [Array(Int,0) for _ in inputs]
    joint_domain = sort(union(f1.domain, f2.domain))
    for (shape,input) in zip(shapes, inputs), var in joint_domain
        push!(shape, (var in input.domain) ? var.n_states : 1)
    end
    joint_logp = reshape(f1.logp, shape[1]) .+ reshape(f2.logp, shape[2])
    Indexed(f1.vars_env, joint_domain, joint_logp)
end

end # Factors

module MRF
import PGM0: Vars
import PGM0: Factors
import Iterators: distinct, chain

type DiscreteMRF
    env :: Vars.Env
    # factors[i] is the vector of factors that contain variable i
    # Note that for a factor over k variables, there are k values
    # of i s.t. the factor is in factors[i]. 
    factors :: Vector{Vector{Factors.Indexed}}
    # all_factors contains every factor just once
    all_factors :: Vector{Factors.Indexed} # collect(distinct(chain(factors...)))
end

function DiscreteMRF(vars::Vector{Vars.Named}, factors::Vector{Vector{Factors.Indexed}})
    DiscreteMRF(vars, factors, collect(distinct(chain(factors...))))
end

n_vars(m::DiscreteMRF) = length(m.vars)
n_states(m::DiscreteMRF, i::Int) = m.vars[i].var.n_states

end # MRF