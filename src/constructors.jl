"""
    determine_sampler_eltype(xs...)

Determine the element type to use for the given arguments.

Symbols are either resolved to the default float type or simply dropped
in favour of determined types from the other arguments.
"""
determine_sampler_eltype(xs...) = float(_determine_sampler_eltype(xs...))
# NOTE: We want to defer conversion to `float` until the very "end" of the
# process to allow `promote_type` to do its job properly.
# For example, in the scenario `determine_sampler_eltype(::Int64, ::Float32)`
# we want to return `Float32`, not `Float64`. The latter would occur
# if we did `float(eltype(x))` instead of just `eltype(x)`.
_determine_sampler_eltype(x) = eltype(x)
_determine_sampler_eltype(x::AbstractIntegrator) = integrator_eltype(x)
_determine_sampler_eltype(::Symbol) = DEFAULT_FLOAT_TYPE
function _determine_sampler_eltype(xs...)
    xs_not_symbol = filter(!Base.Fix2(isa, Symbol), xs)
    isempty(xs_not_symbol) && return DEFAULT_FLOAT_TYPE
    return promote_type(map(_determine_sampler_eltype, xs_not_symbol)...)
end

abstract type AbstractHMCSampler <: AbstractMCMC.AbstractSampler end

"""
    sampler_eltype(sampler)

Return the element type of the sampler.
"""
function sampler_eltype end

##############
### Custom ###
##############

"""
    HMCSampler

An `AbstractMCMC.AbstractSampler` for kernels in AdvancedHMC.jl.

# Fields

$(FIELDS)

# Notes

Note that all the fields have the prefix `initial_` to indicate
that these will not necessarily correspond to the `kernel`, `metric`,
and `adaptor` after sampling.

To access the updated fields, use the resulting [`HMCState`](@ref).
"""
struct HMCSampler{K<:AbstractMCMCKernel,M<:AbstractMetric,A<:AbstractAdaptor} <:
       AbstractHMCSampler
    "[`AbstractMCMCKernel`](@ref)."
    κ::K
    "Choice of initial metric [`AbstractMetric`](@ref). The metric type will be preserved during adaption."
    metric::M
    "[`AdvancedHMC.Adaptation.AbstractAdaptor`](@ref)."
    adaptor::A
end

sampler_eltype(sampler::HMCSampler) = eltype(sampler.metric)

############
### NUTS ###
############
"""
    NUTS(δ::Real; max_depth::Int=10, Δ_max::Real=1000, integrator = :leapfrog, metric = :diagonal)

No-U-Turn Sampler (NUTS) sampler.

# Fields

$(FIELDS)
"""
struct NUTS{T<:Real,I<:Union{Symbol,AbstractIntegrator},M<:Union{Symbol,AbstractMetric}} <:
       AbstractHMCSampler
    "Target acceptance rate for dual averaging."
    δ::T
    "Maximum doubling tree depth."
    max_depth::Int
    "Maximum divergence during doubling tree."
    Δ_max::T
    "Choice of integrator, specified either using a `Symbol` or [`AbstractIntegrator`](@ref)"
    integrator::I
    "Choice of initial metric;  `Symbol` means it is automatically initialised. The metric type will be preserved during automatic initialisation and adaption."
    metric::M
end

function NUTS(δ; max_depth = 10, Δ_max = 1000.0, integrator = :leapfrog, metric = :diagonal)
    T = determine_sampler_eltype(δ, integrator, metric)
    return NUTS(T(δ), max_depth, T(Δ_max), integrator, metric)
end

sampler_eltype(::NUTS{T}) where {T} = T

###########
### HMC ###
###########
"""
    HMC(ϵ::Real, n_leapfrog::Int)

Hamiltonian Monte Carlo sampler with static trajectory.

# Fields

$(FIELDS)
"""
struct HMC{I<:Union{Symbol,AbstractIntegrator},M<:Union{Symbol,AbstractMetric}} <:
       AbstractHMCSampler
    "Number of leapfrog steps."
    n_leapfrog::Int
    "Choice of integrator, specified either using a `Symbol` or [`AbstractIntegrator`](@ref)"
    integrator::I
    "Choice of initial metric;  `Symbol` means it is automatically initialised. The metric type will be preserved during automatic initialisation and adaption."
    metric::M
end

HMC(ϵ::Real, n_leapfrog::Int) = HMC(n_leapfrog, Leapfrog(ϵ), :diagonal)
HMC(n_leapfrog; integrator = :leapfrog, metric = :diagonal) =
    HMC(n_leapfrog, integrator, metric)

sampler_eltype(sampler::HMC) = determine_sampler_eltype(sampler.metric, sampler.integrator)

#############
### HMCDA ###
#############
"""
    HMCDA(δ::Real, λ::Real, integrator = :leapfrog, metric = :diagonal)

Hamiltonian Monte Carlo sampler with Dual Averaging algorithm.

# Fields

$(FIELDS)

# Notes

For more information, please view the following paper ([arXiv link](https://arxiv.org/abs/1111.4246)):

- Hoffman, Matthew D., and Andrew Gelman. "The No-U-turn sampler: adaptively
  setting path lengths in Hamiltonian Monte Carlo." Journal of Machine Learning
  Research 15, no. 1 (2014): 1593-1623.
"""
struct HMCDA{T<:Real,I<:Union{Symbol,AbstractIntegrator},M<:Union{Symbol,AbstractMetric}} <:
       AbstractHMCSampler
    "Target acceptance rate for dual averaging."
    δ::T
    "Target leapfrog length."
    λ::T
    "Choice of integrator, specified either using a `Symbol` or [`AbstractIntegrator`](@ref)"
    integrator::I
    "Choice of initial metric;  `Symbol` means it is automatically initialised. The metric type will be preserved during automatic initialisation and adaption."
    metric::M
end

function HMCDA(δ, λ; integrator = :leapfrog, metric = :diagonal)
    T = determine_sampler_eltype(δ, λ, integrator, metric)
    return HMCDA(T(δ), T(λ), integrator, metric)
end

sampler_eltype(::HMCDA{T}) where {T} = T
