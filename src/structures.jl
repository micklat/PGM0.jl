module Structures
import PGM0: Vars
import PGM0: Factors
import Iterators: distinct, chain, imap

abstract PGMStructure

type BNStructure <: PGMStructure
    env :: Vars.Env
    parents :: Vector{Vector{Int}} # parents[i] are the parents of env[i]
end

factor_domains(bn::BNStructure) = imap((i,pa)->sort([i pa]) , enumerate(bn.parents))
env(bn::BNStructure) = bn.env

type MRFStructure <: PGMStructure
    env :: Vars.Env
    domains :: Vector{Vector{Int}} # domains[i] is the (sorted) domain of the the i'th factor
end

factor_domains(mrf::MRFStructure) = mrf.domains
env(mrf::MRFStructure) = mrf.env

type PGM{Structure<:PGMStructure, F<:Factors.Factor}
    structure :: Structure
    # var_to_factors[i] is the vector of factors that contain variable i
    # Note that for a factor over k variables, there are k values
    # of i s.t. the factor is in factors[i]. 
    var_to_factors :: Vector{Vector{F}}
    # all_factors contains every factor just once
    all_factors :: Vector{F} # collect(distinct(chain(var_to_factors...)))

    function PGM(s::Structure, factors::Vector{F})
        var_to_factors = [[] for _ in 1:length(env(s))]
        for factor in factors
            for var in domain(factor)
                push!(var_to_factors[var], factor)
            end
        end
        new(s, var_to_factors, factors)
    end
end

typealias DiscretePGM{T<:PGMStructure} PGM{T, Factors.Indexed}
typealias DiscreteBN PGM{BNStructure, Factors.Indexed}
typealias DiscreteMRF PGM{MRFStructure, Factors.Indexed}
typealias HybridBN PGM{BNStructure, Factors.Factor}

function discrete_mrf(vars::Vector{Vars.Named}, factors::Vector{Factors.Indexed})
    structure = MRFStructure(vars, map(domain, factors))
    DiscreteMRF(structure, factors)
end

env(pgm::PGM) = env(pgm.structure)
n_vars(m::PGM) = length(env(m))
n_states(m::PGM, i::Integer) = env(m)[i].var.n_states # only works for discrete variables

end # Structures
