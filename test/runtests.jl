include("predict.jl")
include("assume.jl")
include("observe.jl")
include("noparam.jl")
if VERSION < v"0.5" # Depends on ConjugatePriors, which is currently broken in Julia 0.5
  include("beta-binomial.jl")
end
include("resample.jl")
include("importance_sampling.jl")

include("test_clonetask.jl")

if VERSION < v"0.5" # TArray currently in Julia 0.5
  include("test_tarray.jl")
end

# include("test_tarray2.jl")
include("test_particlecontainer.jl")
