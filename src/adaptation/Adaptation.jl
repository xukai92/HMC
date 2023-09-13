module Adaptation
export Adaptation

using LinearAlgebra: LinearAlgebra
using Statistics: Statistics
using SimpleUnPack: @unpack, @pack!

using ..AdvancedHMC: DEBUG, AbstractScalarOrVec, PhasePoint

abstract type AbstractAdaptor end
abstract type AbstractHMCAdaptorWithGradients <: AbstractAdaptor end
function getM⁻¹ end
function getϵ end
function adapt! end
function reset! end
function initialize! end
function finalize! end
export AbstractAdaptor, AbstractHMCAdaptorWithGradients, adapt!, initialize!, finalize!, reset!, getϵ, getM⁻¹

struct NoAdaptation <: AbstractAdaptor end
export NoAdaptation
include("stepsize.jl")
export StepSizeAdaptor, NesterovDualAveraging
include("massmatrix.jl")
export MassMatrixAdaptor, UnitMassMatrix, WelfordVar, WelfordCov

##
## Composite adaptors
## TODO: generalise this to a list of adaptors
##

struct NaiveHMCAdaptor{M<:MassMatrixAdaptor,Tssa<:StepSizeAdaptor} <: AbstractAdaptor
    pc::M
    ssa::Tssa
end
Base.show(io::IO, a::NaiveHMCAdaptor) =
    print(io, "NaiveHMCAdaptor(pc=$(a.pc), ssa=$(a.ssa))")

getM⁻¹(ca::NaiveHMCAdaptor) = getM⁻¹(ca.pc)
getϵ(ca::NaiveHMCAdaptor) = getϵ(ca.ssa)

# TODO: implement consensus adaptor
function adapt!(
    nca::NaiveHMCAdaptor,
    θ::AbstractVecOrMat{<:AbstractFloat},
    α::AbstractScalarOrVec{<:AbstractFloat},
)
    adapt!(nca.ssa, θ, α)
    adapt!(nca.pc, θ, α)
end
function reset!(aca::NaiveHMCAdaptor)
    reset!(aca.ssa)
    reset!(aca.pc)
end
initialize!(adaptor::NaiveHMCAdaptor, n_adapts::Int) = nothing
finalize!(aca::NaiveHMCAdaptor) = finalize!(aca.ssa)

include("stan_adaptor.jl")
export NaiveHMCAdaptor, StanHMCAdaptor

const LOWER_LIMIT::Float64 = 1e-10
const UPPER_LIMIT::Float64 = 1e10

include("nutpie_adaptor.jl")
export NutpieHMCAdaptor, ExpWeightedWelfordVar, NutpieHMCAdaptorNoSwitch, NutpieHMCAdaptorNoGradInit,NutpieHMCAdaptorNoSwitchNoGradInit


end # module
