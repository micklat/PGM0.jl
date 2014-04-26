
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

type DiscretePGM{Structure<:PGMStructure}
    structure :: Structure
    # factors[i] is the vector of factors that contain variable i
    # Note that for a factor over k variables, there are k values
    # of i s.t. the factor is in factors[i]. 
    factors :: Vector{Vector{Factors.Indexed}}
    # all_factors contains every factor just once
    all_factors :: Vector{Factors.Indexed} # collect(distinct(chain(factors...)))
end

function discrete_mrf(vars::Vector{Vars.Named}, factors::Vector{Vector{Factors.Indexed}})
    distinct_factors = collect(distinct(chain(factors...)))
    domains = [factor.domain for factor in distinct_factors]
    structure = MRFStructure(vars, domains)
    mrf = DiscretePGM{MRFStructure}(structure, factors, distinct_factors)
    return mrf
end

n_vars(m::DiscretePGM) = length(env(m.structure))
n_states(m::DiscretePGM, i::Int) = let e = env(m.structure); e[i].var.n_states end

typealias DiscreteBayesianNetwork DiscretePGM{BNStructure}
typealias DiscreteMRF DiscretePGM{MRFStructure}

end # Structures
