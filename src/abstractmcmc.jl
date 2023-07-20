"""
    HMCState

Represents the state of a [`HMCSampler`](@ref).

# Fields

$(FIELDS)

"""
struct HMCState{
    TTrans<:Transition,
    TMetric<:AbstractMetric,
    TKernel<:AbstractMCMCKernel,
    TAdapt<:Adaptation.AbstractAdaptor,
}
    "Index of current iteration."
    i::Int
    "Current [`Transition`](@ref)."
    transition::TTrans
    "Current [`AbstractMetric`](@ref), possibly adapted."
    metric::TMetric
    "Current [`AbstractMCMCKernel`](@ref)."
    κ::TKernel
    "Current [`AbstractAdaptor`](@ref)."
    adaptor::TAdapt
end

"""
    $(TYPEDSIGNATURES)

A convenient wrapper around `AbstractMCMC.sample` avoiding explicit construction of [`HMCSampler`](@ref).
"""

function AbstractMCMC.sample(
    rng::Random.AbstractRNG,
    model::LogDensityModel,
    sampler::AbstractHMCSampler,
    N::Integer;
    progress = true,
    verbose = false,
    callback = nothing,
    kwargs...,
)
    if callback === nothing
        callback = HMCProgressCallback(N, progress = progress, verbose = verbose)
        progress = false # don't use AMCMC's progress-funtionality
    end

    return AbstractMCMC.mcmcsample(
        rng,
        model,
        sampler,
        N;
        progress = progress,
        verbose = verbose,
        callback = callback,
        kwargs...,
    )
end

function AbstractMCMC.sample(
    rng::Random.AbstractRNG,
    model::LogDensityModel,
    sampler::AbstractHMCSampler,
    parallel::AbstractMCMC.AbstractMCMCEnsemble,
    N::Integer,
    nchains::Integer;
    progress = true,
    verbose = false,
    callback = nothing,
    kwargs...,
)

    if callback === nothing
        callback = HMCProgressCallback(N, progress = progress, verbose = verbose)
        progress = false # don't use AMCMC's progress-funtionality
    end

    return AbstractMCMC.mcmcsample(
        rng,
        model,
        sampler,
        parallel,
        N,
        nchains;
        progress = progress,
        verbose = verbose,
        callback = callback,
        kwargs...,
    )
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::LogDensityModel,
    spl::AbstractHMCSampler;
    init_params = nothing,
    kwargs...,
)
    # Unpack model
    logdensity = model.logdensity

    # Define metric
    metric = make_metric(spl, logdensity)

    # Construct the hamiltonian using the initial metric
    hamiltonian = Hamiltonian(metric, model)

    # Define integration algorithm
    # Find good eps if not provided one
    init_params = make_init_params(spl, logdensity, init_params)
    ϵ = make_step_size(rng, spl, hamiltonian, init_params)
    integrator = make_integrator(spl, ϵ)

    # Make kernel
    κ = make_kernel(spl, integrator)

    # Make adaptor
    adaptor = make_adaptor(spl, metric, integrator)

    # Get an initial sample.
    h, t = AdvancedHMC.sample_init(rng, hamiltonian, init_params)

    # Compute next transition and state.
    state = HMCState(0, t, metric, κ, adaptor)
    # Take actual first step.
    return AbstractMCMC.step(rng, model, spl, state; kwargs...)
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::LogDensityModel,
    spl::AbstractHMCSampler,
    state::HMCState;
    kwargs...,
)
    # Compute transition.
    i = state.i + 1
    t_old = state.transition
    adaptor = state.adaptor
    κ = state.κ
    metric = state.metric

    # Reconstruct hamiltonian.
    h = Hamiltonian(metric, model)

    # Make new transition.
    t = transition(rng, h, κ, t_old.z)

    # Adapt h and spl.
    tstat = stat(t)
    n_adapts = get_nadapts(spl)
    h, κ, isadapted = adapt!(h, κ, adaptor, i, n_adapts, t.z.θ, tstat.acceptance_rate)
    tstat = merge(tstat, (is_adapt = isadapted,))

    # Compute next transition and state.
    newstate = HMCState(i, t, h.metric, κ, adaptor)

    # Return `Transition` with additional stats added.
    return Transition(t.z, tstat), newstate
end

################
### Callback ###
################
"""
    HMCProgressCallback

A callback to be used with AbstractMCMC.jl's interface, replicating the
logging behavior of the non-AbstractMCMC [`sample`](@ref).

# Fields
$(FIELDS)
"""
struct HMCProgressCallback{P}
    "`Progress` meter from ProgressMeters.jl."
    pm::P
    "Specifies whether or not to use display a progress bar."
    progress::Bool
    "If `progress` is not specified and this is `true` some information will be logged upon completion of adaptation."
    verbose::Bool
    "Number of divergent transitions fo far."
    num_divergent_transitions::Ref{Int}
    num_divergent_transitions_during_adaption::Ref{Int}
end

function HMCProgressCallback(n_samples; progress = true, verbose = false)
    pm =
        progress ? ProgressMeter.Progress(n_samples, desc = "Sampling", barlen = 31) :
        nothing
    HMCProgressCallback(pm, progress, verbose, Ref(0), Ref(0))
end

function (cb::HMCProgressCallback)(rng, model, spl, t, state, i; nadapts = 0, kwargs...)
    progress = cb.progress
    verbose = cb.verbose
    pm = cb.pm

    metric = state.metric
    adaptor = state.adaptor
    κ = state.κ
    tstat = t.stat
    isadapted = tstat.is_adapt
    if isadapted
        cb.num_divergent_transitions_during_adaption[] += tstat.numerical_error
    else
        cb.num_divergent_transitions[] += tstat.numerical_error
    end

    # Update progress meter
    if progress
        percentage_divergent_transitions = cb.num_divergent_transitions[] / i
        percentage_divergent_transitions_during_adaption =
            cb.num_divergent_transitions_during_adaption[] / i
        if percentage_divergent_transitions > 0.25
            @warn "The level of numerical errors is high. Please check the model carefully." maxlog =
                3
        end
        # Do include current iteration and mass matrix
        pm_next!(
            pm,
            (
                iterations = i,
                ratio_divergent_transitions = round(
                    percentage_divergent_transitions;
                    digits = 2,
                ),
                ratio_divergent_transitions_during_adaption = round(
                    percentage_divergent_transitions_during_adaption;
                    digits = 2,
                ),
                tstat...,
                mass_matrix = metric,
            ),
        )
        # Report finish of adapation
    elseif verbose && isadapted && i == nadapts
        @info "Finished $nadapts adapation steps" adaptor κ.τ.integrator metric
    end
end

#############
### Utils ###
#############

function get_type_of_spl(::AbstractHMCSampler{T}) where {T<:Real}
    return T
end

#########

const SYMBOL_TO_INTEGRATOR_TYPE = Dict(
    :leapfrog => Leapfrog,
    :jitterleapfro => JitteredLeapfrog,
    :temperedleapfrog => TemperedLeapfrog,
)

function determine_integrator_constructor(integrator::Symbol)
    if !haskey(SYMBOL_TO_INTEGRATOR_TYPE, integrator)
        error("Integrator $integrator not supported.")
    end

    return SYMBOL_TO_INTEGRATOR_TYPE[integrator]
end

# If it's the "constructor" of an integrator or instantance of an integrator, do nothing.
determine_integrator_constructor(x::AbstractIntegrator) = x
determine_integrator_constructor(x::Type{<:AbstractIntegrator}) = x
determine_integrator_constructor(x) = error("Integrator $x not supported.")

#########

function make_init_params(spl::AbstractHMCSampler, logdensity, init_params)
    T = get_type_of_spl(spl)
    if init_params == nothing
        d = LogDensityProblems.dimension(logdensity)
        init_params = randn(rng, d)
    end
    return T.(init_params)
end

#########

function make_step_size(
    rng::Random.AbstractRNG,
    spl::AbstractHMCSampler,
    hamiltonian::Hamiltonian,
    init_params,
)
    ϵ = spl.init_ϵ
    if iszero(ϵ)
        ϵ = find_good_stepsize(rng, hamiltonian, init_params)
        T = get_type_of_spl(spl)
        ϵ = T(ϵ)
        @info string("Found initial step size ", ϵ)
    end
    return ϵ
end

function make_step_size(
    rng::Random.AbstractRNG,
    spl::HMCSampler,
    hamiltonian::Hamiltonian,
    init_params,
)
    return spl.κ.τ.integrator.ϵ
end

#########

function make_integrator(spl::AbstractHMCSampler, ϵ::Real)
    integrator = determine_integrator_constructor(spl.integrator)
    return integrator(ϵ)
end

function make_integrator(spl::HMCSampler, ϵ::Real)
    return spl.κ.τ.integrator
end

#########

make_metric(i...) = error("Metric $(typeof(i)) not supported.")
make_metric(i::Symbol, T::Type, d::Int) = make_metric(Val(i), T, d)
make_metric(i::Val{:diagonal}, T::Type, d::Int) = DiagEuclideanMetric(T, d)
make_metric(i::Val{:unit}, T::Type, d::Int) = UnitEuclideanMetric(T, d)
make_metric(i::Val{:dense}, T::Type, d::Int) = DenseEuclideanMetric(T, d)

function make_metric(spl::AbstractHMCSampler, logdensity)
    d = LogDensityProblems.dimension(logdensity)
    T = get_type_of_spl(spl)
    return make_metric(spl.metric, T, d)
end

function make_metric(spl::HMCSampler, logdensity)
    return spl.metric
end

#########

function make_adaptor(
    spl::Union{NUTS,HMCDA},
    metric::AbstractMetric,
    integrator::AbstractIntegrator,
)
    return StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(spl.δ, integrator))
end

function make_adaptor(spl::HMC, metric::AbstractMetric, integrator::AbstractIntegrator)
    return NoAdaptation()
end

function make_adaptor(
    spl::HMCSampler,
    metric::AbstractMetric,
    integrator::AbstractIntegrator,
)
    return spl.adaptor
end

#########

get_nadapts(spl::Union{HMCSampler,NUTS,HMCDA}) = spl.n_adapts
get_nadapts(spl::HMC) = 0

#########

function make_kernel(spl::NUTS, integrator::AbstractIntegrator)
    return HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
end

function make_kernel(spl::HMC, integrator::AbstractIntegrator)
    return HMCKernel(Trajectory{EndPointTS}(integrator, FixedNSteps(spl.n_leapfrog)))
end

function make_kernel(spl::HMCDA, integrator::AbstractIntegrator)
    return HMCKernel(Trajectory{EndPointTS}(integrator, FixedIntegrationTime(spl.λ)))
end

function make_kernel(spl::HMCSampler, integrator::AbstractIntegrator)
    return spl.κ
end
