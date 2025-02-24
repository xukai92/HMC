module AdvancedHMC

using Statistics: mean, var, middle
using LinearAlgebra:
    Symmetric, UpperTriangular, mul!, ldiv!, dot, I, diag, cholesky, UniformScaling
using StatsFuns: logaddexp, logsumexp, loghalf
import Random
using Random: AbstractRNG
using ProgressMeter: ProgressMeter
using SimpleUnPack: @unpack

using Setfield
import Setfield: ConstructionBase

using ArgCheck: @argcheck

using DocStringExtensions

using LogDensityProblems
using LogDensityProblemsAD: LogDensityProblemsAD

import AbstractMCMC
using AbstractMCMC: LogDensityModel

import StatsBase: sample

const DEFAULT_FLOAT_TYPE = typeof(float(0))

include("utilities.jl")

# Notations
# ℓπ: log density of the target distribution
# θ: position variables / model parameters
# ∂ℓπ∂θ: gradient of the log density of the target distribution w.r.t θ
# r: momentum variables
# z: phase point / a pair of θ and r

# TODO Move it back to hamiltonian.jl after the rand interface is updated
abstract type AbstractKinetic end

struct GaussianKinetic <: AbstractKinetic end

export GaussianKinetic

include("metric.jl")
export UnitEuclideanMetric, DiagEuclideanMetric, DenseEuclideanMetric

include("hamiltonian.jl")
export Hamiltonian

include("integrator.jl")
export Leapfrog, JitteredLeapfrog, TemperedLeapfrog
include("riemannian/integrator.jl")
export GeneralizedLeapfrog

include("trajectory.jl")
export Trajectory,
    HMCKernel,
    FixedNSteps,
    FixedIntegrationTime,
    ClassicNoUTurn,
    GeneralisedNoUTurn,
    StrictGeneralisedNoUTurn,
    EndPointTS,
    SliceTS,
    MultinomialTS,
    find_good_stepsize

# Useful defaults

@deprecate find_good_eps find_good_stepsize

export find_good_eps

include("adaptation/Adaptation.jl")
using .Adaptation
import .Adaptation:
    StepSizeAdaptor, MassMatrixAdaptor, StanHMCAdaptor, NesterovDualAveraging, NoAdaptation

# Helpers for initializing adaptors via AHMC structs

StepSizeAdaptor(δ::AbstractFloat, stepsize::AbstractScalarOrVec{<:AbstractFloat}) =
    NesterovDualAveraging(δ, stepsize)
StepSizeAdaptor(δ::AbstractFloat, i::AbstractIntegrator) =
    StepSizeAdaptor(δ, nom_step_size(i))

MassMatrixAdaptor(m::UnitEuclideanMetric{T}) where {T} = UnitMassMatrix{T}()
MassMatrixAdaptor(m::DiagEuclideanMetric{T}) where {T} =
    WelfordVar{T}(size(m); var = copy(m.M⁻¹))
MassMatrixAdaptor(m::DenseEuclideanMetric{T}) where {T} =
    WelfordCov{T}(size(m); cov = copy(m.M⁻¹))

MassMatrixAdaptor(::Type{TM}, sz::Dims = (2,)) where {TM<:AbstractMetric} =
    MassMatrixAdaptor(Float64, TM, sz)

MassMatrixAdaptor(::Type{T}, ::Type{TM}, sz::Dims = (2,)) where {T,TM<:AbstractMetric} =
    MassMatrixAdaptor(TM(T, sz))

# Deprecations

@deprecate StanHMCAdaptor(n_adapts, pc, ssa) initialize!(StanHMCAdaptor(pc, ssa), n_adapts)
@deprecate NesterovDualAveraging(δ::AbstractFloat, i::AbstractIntegrator) StepSizeAdaptor(
    δ,
    i,
)
@deprecate Preconditioner(args...) MassMatrixAdaptor(args...)

export StepSizeAdaptor,
    NesterovDualAveraging,
    MassMatrixAdaptor,
    UnitMassMatrix,
    WelfordVar,
    WelfordCov,
    NaiveHMCAdaptor,
    StanHMCAdaptor,
    NoAdaptation

include("diagnosis.jl")

include("sampler.jl")
export sample

include("constructors.jl")
export HMCSampler, HMC, NUTS, HMCDA

include("abstractmcmc.jl")

## Without explicit AD backend
function Hamiltonian(metric::AbstractMetric, ℓ::LogDensityModel; kwargs...)
    return Hamiltonian(metric, ℓ.logdensity; kwargs...)
end
function Hamiltonian(metric::AbstractMetric, ℓ; kwargs...)
    cap = LogDensityProblems.capabilities(ℓ)
    if cap === nothing
        throw(
            ArgumentError(
                "The log density function does not support the LogDensityProblems.jl interface",
            ),
        )
    end
    # Check if we're capable of computing gradients.
    ℓπ = if cap === LogDensityProblems.LogDensityOrder{0}()
        # In this case ℓ does not support evaluation of the gradient of the log density function
        # We use ForwardDiff to compute the gradient
        LogDensityProblemsAD.ADgradient(Val(:ForwardDiff), ℓ; kwargs...)
    else
        # In this case ℓ already supports evaluation of the gradient of the log density function
        ℓ
    end
    return Hamiltonian(
        metric,
        Base.Fix1(LogDensityProblems.logdensity, ℓπ),
        Base.Fix1(LogDensityProblems.logdensity_and_gradient, ℓπ),
    )
end

## With explicit AD specification
function Hamiltonian(
    metric::AbstractMetric,
    ℓπ::LogDensityModel,
    kind::Union{Symbol,Val,Module};
    kwargs...,
)
    return Hamiltonian(metric, ℓπ.logdensity, kind; kwargs...)
end
function Hamiltonian(metric::AbstractMetric, ℓπ, kind::Union{Symbol,Val,Module}; kwargs...)
    if LogDensityProblems.capabilities(ℓπ) === nothing
        throw(
            ArgumentError(
                "The log density function does not support the LogDensityProblems.jl interface",
            ),
        )
    end
    ℓ = LogDensityProblemsAD.ADgradient(
        kind isa Val ? kind : Val(Symbol(kind)),
        ℓπ;
        kwargs...,
    )
    return Hamiltonian(metric, ℓ)
end

### Init

struct DiffEqIntegrator{T<:AbstractScalarOrVec{<:AbstractFloat},DiffEqSolver} <:
       AbstractLeapfrog{T}
    ϵ::T
    solver::DiffEqSolver
end
export DiffEqIntegrator

if !isdefined(Base, :get_extension)
    using Requires
end
function __init__()
    # Better error message if users forgot to load OrdinaryDiffEq
    Base.Experimental.register_error_hint(MethodError) do io, exc, arg_types, kwargs
        n = length(arg_types)
        if exc.f === step &&
           (n == 3 || n == 4) &&
           arg_types[1] <: DiffEqIntegrator &&
           arg_types[2] <: Hamiltonian &&
           arg_types[3] <: PhasePoint &&
           (n == 3 || arg_types[4] === Int)

            print(io, "\\nDid you forget to load OrdinaryDiffEq?")
        end
    end
    @static if !isdefined(Base, :get_extension)
        @require OrdinaryDiffEq = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed" begin
            include("../ext/AdvancedHMCOrdinaryDiffEqExt.jl")
        end

        @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" begin
            include("../ext/AdvancedHMCCUDAExt.jl")
        end

        @require MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d" begin
            include("../ext/AdvancedHMCMCMCChainsExt.jl")
        end
    end
end

end # module
