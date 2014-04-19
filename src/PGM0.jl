module PGM0

include("utils.jl")
include("model.jl")
include("gibbs.jl")
include("TaylorEM.jl")

export DiscreteMRF, Gibbs

end # module
