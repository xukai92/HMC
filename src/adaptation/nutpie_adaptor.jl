####################################3
### General methods
# it will then be forwarded to `adaptor` (there it resides in `exp_variance_draw.var`)
getM⁻¹(ca::AbstractHMCAdaptorWithGradients) = getM⁻¹(ca.pc)
getϵ(ca::AbstractHMCAdaptorWithGradients) = getϵ(ca.ssa)
finalize!(adaptor::AbstractHMCAdaptorWithGradients) = finalize!(adaptor.ssa)

### Mutable states 
mutable struct NutpieHMCAdaptorState
    i::Int
    n_adapts::Int
    # The number of draws in the the early window
    early_end::Int
    # The first draw number for the final step size adaptation window
    final_step_size_window::Int

    function NutpieHMCAdaptorState(i, n_adapts, early_end, final_step_size_window)
        @assert (early_end < n_adapts) "Early_end must be less than num_tune (provided $early_end and $n_adapts)"
        return new(i, n_adapts, early_end, final_step_size_window)
    end
end
function NutpieHMCAdaptorState()
    return NutpieHMCAdaptorState(0, 1000, 300, 800)
end

function initialize!(state::NutpieHMCAdaptorState, early_window_share::Float64,
    final_step_size_window_share::Float64,
    n_adapts::Int)

    early_end = ceil(UInt64, early_window_share * n_adapts)
    step_size_window = ceil(UInt64, final_step_size_window_share * n_adapts)
    final_step_size_window = max(n_adapts - step_size_window, 0) + 1

    state.early_end = early_end
    state.n_adapts = n_adapts
    state.final_step_size_window = final_step_size_window
end

# function Base.show(io::IO, state::NutpieHMCAdaptorState)
#     print(io, "window($(state.window_start), $(state.window_end)), window_splits(" * string(join(state.window_splits, ", ")) * ")")
# end

### Nutpie's adaptation
# Acknowledgement: ...
struct NutpieHMCAdaptor{M<:MassMatrixAdaptor,Tssa<:StepSizeAdaptor} <: AbstractHMCAdaptorWithGradients
    pc::M
    ssa::Tssa
    early_window_share::Float64
    final_step_size_window_share::Float64
    mass_matrix_switch_freq::Int
    early_mass_matrix_switch_freq::Int
    state::NutpieHMCAdaptorState
end
# Base.show(io::IO, a::NutpieHMCAdaptor) =
#     print(io, "NutpieHMCAdaptor(\n    pc=$(a.pc),\n    ssa=$(a.ssa),\n    init_buffer=$(a.init_buffer), term_buffer=$(a.term_buffer), window_size=$(a.window_size),\n    state=$(a.state)\n)")

function NutpieHMCAdaptor(
    pc::ExpWeightedWelfordVar,
    ssa::StepSizeAdaptor;
    early_window_share::Float64=0.3,
    final_step_size_window_share::Float64=0.2,
    mass_matrix_switch_freq::Int=60,
    early_mass_matrix_switch_freq::Int=10
)
    return NutpieHMCAdaptor(pc, ssa, early_window_share, final_step_size_window_share, mass_matrix_switch_freq, early_mass_matrix_switch_freq, NutpieHMCAdaptorState())
end

function initialize!(adaptor::NutpieHMCAdaptor, n_adapts::Int, ∇logπ::AbstractVecOrMat{<:AbstractFloat})
    initialize!(adaptor.state, adaptor.early_window_share, adaptor.final_step_size_window_share, n_adapts)
    # !Q: Shall we initialize from the gradient?
    # Nutpie initializes the variance estimate with reciprocal of the gradient
    # Like: Some((grad).abs().recip().clamp(LOWER_LIMIT, UPPER_LIMIT))
    # TODO: point to var more dynamically
    adaptor.pc.exp_variance_draw.var = (1 ./ abs.(∇logπ)) |> x -> clamp.(x, LOWER_LIMIT, UPPER_LIMIT)
    return adaptor
end

####################################
## Special case: Skip the initiation of the mass matrix with gradient
struct NutpieHMCAdaptorNoGradInit{M<:MassMatrixAdaptor,Tssa<:StepSizeAdaptor} <: AbstractHMCAdaptorWithGradients
    pc::M
    ssa::Tssa
    early_window_share::Float64
    final_step_size_window_share::Float64
    mass_matrix_switch_freq::Int
    early_mass_matrix_switch_freq::Int
    state::NutpieHMCAdaptorState
end
# Base.show(io::IO, a::NutpieHMCAdaptor) =
#     print(io, "NutpieHMCAdaptor(\n    pc=$(a.pc),\n    ssa=$(a.ssa),\n    init_buffer=$(a.init_buffer), term_buffer=$(a.term_buffer), window_size=$(a.window_size),\n    state=$(a.state)\n)")

function NutpieHMCAdaptorNoGradInit(
    pc::ExpWeightedWelfordVar,
    ssa::StepSizeAdaptor;
    early_window_share::Float64=0.3,
    final_step_size_window_share::Float64=0.2,
    mass_matrix_switch_freq::Int=60,
    early_mass_matrix_switch_freq::Int=10
)
    return NutpieHMCAdaptorNoGradInit(pc, ssa, early_window_share, final_step_size_window_share, mass_matrix_switch_freq, early_mass_matrix_switch_freq, NutpieHMCAdaptorState())
end
function initialize!(adaptor::NutpieHMCAdaptorNoGradInit, n_adapts::Int, ∇logπ::AbstractVecOrMat{<:AbstractFloat})
    initialize!(adaptor.state, adaptor.early_window_share, adaptor.final_step_size_window_share, n_adapts)
    return adaptor
end
####################################
## Special case: No switching, use StanHMCAdaptor-like strategy (but keep var+gradients)
struct NutpieHMCAdaptorNoSwitch{M<:MassMatrixAdaptor,Tssa<:StepSizeAdaptor} <: AbstractHMCAdaptorWithGradients
    pc::M
    ssa::Tssa
    init_buffer::Int
    term_buffer::Int
    window_size::Int
    state::StanHMCAdaptorState
end

function NutpieHMCAdaptorNoSwitch(
    pc::ExpWeightedWelfordVar,
    ssa::StepSizeAdaptor;
    init_buffer::Int = 75,
    term_buffer::Int = 50,
    window_size::Int = 25,
)
    return NutpieHMCAdaptorNoSwitch(
        pc,
        ssa,
        init_buffer,
        term_buffer,
        window_size,
        StanHMCAdaptorState(),
    )
end

function initialize!(adaptor::NutpieHMCAdaptorNoSwitch, n_adapts::Int,∇logπ::AbstractVecOrMat{<:AbstractFloat})
    initialize!(
        adaptor.state,
        adaptor.init_buffer,
        adaptor.term_buffer,
        adaptor.window_size,
        n_adapts,
    )
    adaptor.pc.exp_variance_draw.var = (1 ./ abs.(∇logπ)) |> x -> clamp.(x, LOWER_LIMIT, UPPER_LIMIT)
    return adaptor
end

############################################
## Special case: No switching, use StanHMCAdaptor-like strategy (but keep var+gradients)
## Both switching and grad init disabled
struct NutpieHMCAdaptorNoSwitchNoGradInit{M<:MassMatrixAdaptor,Tssa<:StepSizeAdaptor} <: AbstractHMCAdaptorWithGradients
    pc::M
    ssa::Tssa
    init_buffer::Int
    term_buffer::Int
    window_size::Int
    state::StanHMCAdaptorState
end

function NutpieHMCAdaptorNoSwitchNoGradInit(
    pc::ExpWeightedWelfordVar,
    ssa::StepSizeAdaptor;
    init_buffer::Int = 75,
    term_buffer::Int = 50,
    window_size::Int = 25,
)
    return NutpieHMCAdaptorNoSwitchNoGradInit(
        pc,
        ssa,
        init_buffer,
        term_buffer,
        window_size,
        StanHMCAdaptorState(),
    )
end

function initialize!(adaptor::NutpieHMCAdaptorNoSwitchNoGradInit, n_adapts::Int,∇logπ::AbstractVecOrMat{<:AbstractFloat})
    initialize!(
        adaptor.state,
        adaptor.init_buffer,
        adaptor.term_buffer,
        adaptor.window_size,
        n_adapts,
    )
    return adaptor
end


#####################################
# Adaptation: main case ala Nutpie
#
# Changes vs Rust implementation
# - step_size_adapt is at the top
# - several checks are handled in sampler (finalize adaptation, does not adapt during normal sampling)
# - switch and push/update are handled separately to mimic the StanHMCAdaptor
#
# Missing: 
# - collector checks on divergences or terminating at idx=0 // finite and None esimtaor
# - adapt! estimator only if collected stuff is good (divergences)
# - init for mass_matrix is grad.abs().recip.clamp(LOWER_LIMIT, UPPER_LIMIT) // init of ExpWindowDiagAdapt
#
is_in_first_step_size_window(tp::AbstractHMCAdaptorWithGradients) = tp.state.i <= tp.state.final_step_size_window
is_in_early_window(tp::AbstractHMCAdaptorWithGradients) = tp.state.i <= tp.state.early_end
switch_freq(tp::AbstractHMCAdaptorWithGradients) = is_in_early_window(tp) ? tp.early_mass_matrix_switch_freq : tp.mass_matrix_switch_freq
#
function adapt!(
    tp::Union{NutpieHMCAdaptor,NutpieHMCAdaptorNoGradInit},
    θ::AbstractVecOrMat{<:AbstractFloat},
    α::AbstractScalarOrVec{<:AbstractFloat},
    ∇logπ::AbstractVecOrMat{<:AbstractFloat}
)
    tp.state.i += 1

    adapt!(tp.ssa, θ, α)

    # TODO: do we resize twice? also in update?
    # !Q: Why do we check resizing several times during iteration? (also in adapt!)
    resize!(tp.pc, θ, ∇logπ) # Resize pre-conditioner if necessary.

    # determine whether to update mass matrix
    if is_in_first_step_size_window(tp)

        # Switch swaps the background (_bg) values for current, and resets the background values 
        # Frequency of the switch depends on the phase
        background_count(tp.pc) >= switch_freq(tp) && switch!(tp.pc)

        # TODO: implement a skipper for bad draws
        # !Q: Why does it always update? (as per Nuts-rs/Nutpie)
        adapt!(tp.pc, θ, α, ∇logπ, true)
    end
end

#####################################
# Adaptation: No switching - ala StanHMCAdaptor
#
is_in_window(tp::Union{NutpieHMCAdaptorNoSwitch,NutpieHMCAdaptorNoSwitchNoGradInit}) =
    tp.state.i >= tp.state.window_start && tp.state.i <= tp.state.window_end
is_window_end(tp::Union{NutpieHMCAdaptorNoSwitch,NutpieHMCAdaptorNoSwitchNoGradInit}) = tp.state.i in tp.state.window_splits
#
function adapt!(
    tp::Union{NutpieHMCAdaptorNoSwitch,NutpieHMCAdaptorNoSwitchNoGradInit},
    θ::AbstractVecOrMat{<:AbstractFloat},
    α::AbstractScalarOrVec{<:AbstractFloat},
    ∇logπ::AbstractVecOrMat{<:AbstractFloat}
)
    tp.state.i += 1

    adapt!(tp.ssa, θ, α)

    resize!(tp.pc, θ, ∇logπ) # Resize pre-conditioner if necessary.

    # Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/hmc/nuts/adapt_diag_e_nuts.hpp
    if is_in_window(tp)
        # We accumlate stats from θ online and only trigger the update of M⁻¹ in the end of window.
        is_update_M⁻¹ = is_window_end(tp)
        adapt!(tp.pc, θ, α, ∇logπ, is_update_M⁻¹)
    end

    if is_window_end(tp)
        reset!(tp.ssa)
        reset!(tp.pc)
    end
end



