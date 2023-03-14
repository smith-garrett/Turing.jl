"""
    ABC()

Approximate Bayesian computation sampling algorithm.

Usage:

```julia
ABC(threshold)
```

Example:

```julia
# Define a simple Normal model with unknown mean and variance.
@model function gdemo(x)
    m ~ Normal(0, 1)
    x ~ Normal(m, 1)
    return m
end

sample(gdemo([0.2]), ABC(1e-3), 1000)
```
"""

struct ABC{T} <: AbstractMCMC.AbstractSampler
    threshold::T
end

function Sampler(alg::IS, model::Model, s::Selector)
    info = Dict{Symbol, Any}()
    state = ABCState(model)
    return Sampler(alg, info, s, state)
end

mutable struct ABCState{V<:VarInfo, F<:AbstractFloat} <: AbstractSamplerState
    vi::V
end
ABCState(model::Model) = ABCState(VarInfo(model))

struct Transition{T}
    θ::T
end

function DynamicPPL.assume(rng, spl::Sampler{<:ABC}, dist::Distribution, vn::VarName, vi)
    r = rand(rng, dist)
    push!(vi, vn, r, dist, spl)
    return r, 0
end

function DynamicPPL.observe(spl::Sampler{<:ABC}, dist::Distribution, value, vi)
    x_sim = rand(dist)
    return isapprox(x_sim, value, atol=spl.alg.threshold) ? 0 : -Inf
end
