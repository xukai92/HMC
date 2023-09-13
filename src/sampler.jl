# Update of hamiltonian and proposal

update(h::Hamiltonian, ::AbstractAdaptor) = h
function update(
    h::Hamiltonian,
    adaptor::Union{MassMatrixAdaptor,NaiveHMCAdaptor,StanHMCAdaptor},
)
    metric = renew(h.metric, getM⁻¹(adaptor))
    return @set h.metric = metric
end

update(τ::Trajectory, ::AbstractAdaptor) = τ
function update(
    τ::Trajectory,
    adaptor::Union{StepSizeAdaptor,NaiveHMCAdaptor,StanHMCAdaptor},
)
    # FIXME: this does not support change type of `ϵ` (e.g. Float to Vector)
    integrator = update_nom_step_size(τ.integrator, getϵ(adaptor))
    @set τ.integrator = integrator
end

function update(κ::AbstractMCMCKernel, adaptor::AbstractAdaptor)
    @set κ.τ = update(κ.τ, adaptor)
end

function resize(h::Hamiltonian, θ::AbstractVecOrMat{T}) where {T<:AbstractFloat}
    metric = h.metric
    if size(metric) != size(θ)
        metric = ConstructionBase.constructorof(typeof(metric))(size(θ))
        @set! h.metric = metric
    end
    return h
end

##
## Interface functions
##
function sample_init(
    rng::Union{AbstractRNG,AbstractVector{<:AbstractRNG}},
    h::Hamiltonian,
    θ::AbstractVecOrMat{<:AbstractFloat},
)
    # Ensure h.metric has the same dim as θ.
    h = resize(h, θ)
    # Initial transition
    t = Transition(phasepoint(rng, θ, h), NamedTuple())
    return h, t
end

function transition(
    rng::Union{AbstractRNG,AbstractVector{<:AbstractRNG}},
    h::Hamiltonian,
    κ::HMCKernel,
    z::PhasePoint,
)
    @unpack refreshment, τ = κ
    @set! τ.integrator = jitter(rng, τ.integrator)
    z = refresh(rng, refreshment, h, z)
    return transition(rng, τ, h, z)
end

Adaptation.adapt!(
    h::Hamiltonian,
    κ::AbstractMCMCKernel,
    adaptor::Adaptation.NoAdaptation,
    i::Int,
    n_adapts::Int,
    θ::AbstractVecOrMat{<:AbstractFloat},
    α::AbstractScalarOrVec{<:AbstractFloat},
) = h, κ, false

function Adaptation.adapt!(
    h::Hamiltonian,
    κ::AbstractMCMCKernel,
    adaptor::AbstractAdaptor,
    i::Int,
    n_adapts::Int,
    z::PhasePoint,
    # θ::AbstractVecOrMat{<:AbstractFloat},
    α::AbstractScalarOrVec{<:AbstractFloat},
)
    θ = z.θ
    isadapted = false
    if i <= n_adapts
        i == 1 && Adaptation.initialize!(adaptor, n_adapts)
        adapt!(adaptor, θ, α)
        i == n_adapts && finalize!(adaptor)
        h = update(h, adaptor)
        κ = update(κ, adaptor)
        isadapted = true
    end
    return h, κ, isadapted
end

# Nutpie adaptor requires access to gradients in the Hamiltonian
function Adaptation.adapt!(
    h::Hamiltonian,
    κ::AbstractMCMCKernel,
    adaptor::AbstractHMCAdaptorWithGradients,
    i::Int,
    n_adapts::Int,
    z::PhasePoint,
    # θ::AbstractVecOrMat{<:AbstractFloat},
    α::AbstractScalarOrVec{<:AbstractFloat},
)
    θ = z.θ
    ∇logπ = z.ℓπ.gradient
    isadapted = false
    if i <= n_adapts
        i == 1 && Adaptation.initialize!(adaptor, n_adapts,∇logπ)
        adapt!(adaptor, θ, α,∇logπ)
        i == n_adapts && finalize!(adaptor)
        h = update(h, adaptor)
        κ = update(κ, adaptor)
        isadapted = true
    end
    return h, κ, isadapted
end

"""
Progress meter update with all trajectory stats, iteration number and metric shown.
"""
function pm_next!(pm, stat::NamedTuple)
    ProgressMeter.next!(pm; showvalues=[tuple(s...) for s in pairs(stat)])
end

"""
Simple progress meter update without any show values.
"""
simple_pm_next!(pm, stat::NamedTuple) = ProgressMeter.next!(pm)

##
## Sampling functions
##

sample(
    h::Hamiltonian,
    κ::AbstractMCMCKernel,
    θ::AbstractVecOrMat{<:AbstractFloat},
    n_samples::Int,
    adaptor::AbstractAdaptor=NoAdaptation(),
    n_adapts::Int=min(div(n_samples, 10), 1_000);
    drop_warmup=false,
    verbose::Bool=true,
    progress::Bool=false,
    (pm_next!)::Function=pm_next!
) = sample(
    GLOBAL_RNG,
    h,
    κ,
    θ,
    n_samples,
    adaptor,
    n_adapts;
    drop_warmup=drop_warmup,
    verbose=verbose,
    progress=progress,
    (pm_next!)=pm_next!
)

"""
    sample(
        rng::AbstractRNG,
        h::Hamiltonian,
        κ::AbstractMCMCKernel,
        θ::AbstractVecOrMat{T},
        n_samples::Int,
        adaptor::AbstractAdaptor=NoAdaptation(),
        n_adapts::Int=min(div(n_samples, 10), 1_000);
        drop_warmup::Bool=false,
        verbose::Bool=true,
        progress::Bool=false
    )
Sample `n_samples` samples using the proposal `κ` under Hamiltonian `h`.
- The randomness is controlled by `rng`. 
    - If `rng` is not provided, `GLOBAL_RNG` will be used.
- The initial point is given by `θ`.
- The adaptor is set by `adaptor`, for which the default is no adaptation.
    - It will perform `n_adapts` steps of adaptation, for which the default is the minimum of `1_000` and 10% of `n_samples`
- `drop_warmup` controls to drop the samples during adaptation phase or not
- `verbose` controls the verbosity
- `progress` controls whether to show the progress meter or not
"""
function sample(
    rng::Union{AbstractRNG,AbstractVector{<:AbstractRNG}},
    h::Hamiltonian,
    κ::HMCKernel,
    θ::T,
    n_samples::Int,
    adaptor::AbstractAdaptor=NoAdaptation(),
    n_adapts::Int=min(div(n_samples, 10), 1_000);
    drop_warmup=false,
    verbose::Bool=true,
    progress::Bool=false,
    (pm_next!)::Function=pm_next!
) where {T<:AbstractVecOrMat{<:AbstractFloat}}
    @assert !(drop_warmup && (adaptor isa Adaptation.NoAdaptation)) "Cannot drop warmup samples if there is no adaptation phase."
    # Prepare containers to store sampling results
    n_keep = n_samples - (drop_warmup ? n_adapts : 0)
    θs, stats = Vector{T}(undef, n_keep), Vector{NamedTuple}(undef, n_keep)
    num_divergent_transitions = 0
    num_divergent_transitions_during_adaption = 0
    # Initial sampling
    h, t = sample_init(rng, h, θ)
    # Progress meter
    pm =
        progress ? ProgressMeter.Progress(n_samples, desc="Sampling", barlen=31) :
        nothing
    time = @elapsed for i = 1:n_samples
        # Make a transition
        t = transition(rng, h, κ, t.z)
        # Adapt h and κ; what mutable is the adaptor
        tstat = stat(t)
        h, κ, isadapted =
            adapt!(h, κ, adaptor, i, n_adapts, t.z, tstat.acceptance_rate)
        if isadapted
            num_divergent_transitions_during_adaption += tstat.numerical_error
        else
            num_divergent_transitions += tstat.numerical_error
        end
        tstat = merge(tstat, (is_adapt=isadapted,))
        # Update progress meter
        if progress
            percentage_divergent_transitions = num_divergent_transitions / i
            percentage_divergent_transitions_during_adaption =
                num_divergent_transitions_during_adaption / i
            if percentage_divergent_transitions > 0.25
                @warn "The level of numerical errors is high. Please check the model carefully." maxlog =
                    3
            end
            # Do include current iteration and mass matrix
            pm_next!(
                pm,
                (
                    iterations=i,
                    ratio_divergent_transitions=round(
                        percentage_divergent_transitions;
                        digits=2
                    ),
                    ratio_divergent_transitions_during_adaption=round(
                        percentage_divergent_transitions_during_adaption;
                        digits=2
                    ),
                    tstat...,
                    mass_matrix=h.metric,
                ),
            )
            # Report finish of adapation
        elseif verbose && isadapted && i == n_adapts
            @info "Finished $n_adapts adapation steps" adaptor κ.τ.integrator h.metric
        end
        # Store sample
        if !drop_warmup || i > n_adapts
            j = i - drop_warmup * n_adapts
            θs[j], stats[j] = t.z.θ, tstat
        end
    end
    # Report end of sampling
    if verbose
        EBFMI_est = EBFMI(map(s -> s.hamiltonian_energy, stats))
        average_acceptance_rate = mean(map(s -> s.acceptance_rate, stats))
        if θ isa AbstractVector
            n_chains = 1
        else
            n_chains = size(θ, 2)
            # Make sure that arrays are on CPU before printing.
            EBFMI_est = convert(Vector{eltype(EBFMI_est)}, EBFMI_est)
            average_acceptance_rate =
                convert(Vector{eltype(average_acceptance_rate)}, average_acceptance_rate)
            EBFMI_est = "[" * join(EBFMI_est, ", ") * "]"
            average_acceptance_rate = "[" * join(average_acceptance_rate, ", ") * "]"
        end
        @info "Finished $n_samples sampling steps for $n_chains chains in $time (s)" h κ EBFMI_est average_acceptance_rate
    end
    return θs, stats
end
