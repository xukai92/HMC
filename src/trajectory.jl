####
#### Implementation for Hamiltonian dynamics trajectories
####
#### Developers' Notes
####
#### Not all functions that use `rng` require a fallback function with `Random.default_rng()`
#### as default. In short, only those exported to other libries need such a fallback
#### function. Internal uses shall always use the explict `rng` version. (Kai Xu 6/Jul/19)

"""
$(TYPEDEF)
A transition that contains the phase point and
other statistics of the transition.

# Fields
$(TYPEDFIELDS)
"""
struct Transition{P<:PhasePoint,NT<:NamedTuple}
    "Phase-point for the transition."
    z::P
    "Statistics related to the transition, e.g. energy."
    stat::NT
end

"Returns the statistics for transition `t`."
stat(t::Transition) = t.stat

"""
$(TYPEDEF)
Abstract type for HMC kernels. 
"""
abstract type AbstractMCMCKernel end

"""
$(TYPEDEF)
Abstract type for termination criteria for Hamiltonian trajectories, e.g. no-U-turn and fixed number of leapfrog integration steps. 
"""
abstract type AbstractTerminationCriterion end

"""
$(TYPEDEF)
Abstract type for a fixed number of leapfrog integration steps.
"""
abstract type StaticTerminationCriterion <: AbstractTerminationCriterion end

"""
$(TYPEDEF)
Abstract type for dynamic Hamiltonian trajectory termination criteria. 
"""
abstract type DynamicTerminationCriterion <: AbstractTerminationCriterion end

"""
$(TYPEDEF)
Static HMC with a fixed number of leapfrog steps.

# Fields
$(TYPEDFIELDS)

# References
1. Neal, R. M. (2011). MCMC using Hamiltonian dynamics. Handbook of Markov chain Monte Carlo, 2(11), 2. ([arXiv](https://arxiv.org/pdf/1206.1901))
"""
struct FixedNSteps <: StaticTerminationCriterion
    "Number of steps to simulate, i.e. length of trajectory will be `L + 1`."
    L::Int
end

"""
$(TYPEDEF)
Standard HMC implementation with a fixed integration time.

# Fields
$(TYPEDFIELDS)

# References
1. Neal, R. M. (2011). MCMC using Hamiltonian dynamics. Handbook of Markov chain Monte Carlo, 2(11), 2. ([arXiv](https://arxiv.org/pdf/1206.1901)) 
"""
struct FixedIntegrationTime{F<:AbstractFloat} <: StaticTerminationCriterion
    "Total length of the trajectory, i.e. take `floor(λ / integrator_step_size)` number of leapfrog steps."
    λ::F
end

##
## Sampling methods for trajectories.
##

"How to sample a phase-point from the simulated trajectory."
abstract type AbstractTrajectorySampler end

"Samples the end-point of the trajectory."
struct EndPointTS <: AbstractTrajectorySampler end

"""
$(TYPEDEF)

Trajectory slice sampler carried during the building of the tree.
It contains the slice variable and the number of acceptable condidates in the tree.

# Fields

$(TYPEDFIELDS)
"""
struct SliceTS{F<:AbstractFloat} <: AbstractTrajectorySampler
    "Sampled candidate `PhasePoint`."
    zcand::PhasePoint
    "Slice variable in log-space."
    ℓu::F
    "Number of acceptable candidates, i.e. those with probability larger than slice variable `u`."
    n::Int
end

Base.show(io::IO, s::SliceTS) = print(io, "SliceTS(ℓu=$(s.ℓu), n=$(s.n))")

"""
$(TYPEDEF)

Multinomial trajectory sampler carried during the building of the tree.
It contains the weight of the tree, defined as the total probabilities of the leaves.

# Fields

$(TYPEDFIELDS)
"""
struct MultinomialTS{F<:AbstractFloat} <: AbstractTrajectorySampler
    "Sampled candidate `PhasePoint`."
    zcand::PhasePoint
    "Total energy for the given tree, i.e. the sum of energies of all leaves."
    ℓw::F
end

"""
$(TYPEDEF)

Slice sampler for the starting single leaf tree.
Slice variable is initialized.
"""
SliceTS(rng::AbstractRNG, z0::PhasePoint) =
    SliceTS(z0, neg_energy(z0) - Random.randexp(rng), 1)

"""
$(TYPEDEF)

Multinomial sampler for the starting single leaf tree.
(Log) weights for leaf nodes are their (unnormalised) Hamiltonian energies.

Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/hmc/nuts/base_nuts.hpp#L226
"""
MultinomialTS(rng::AbstractRNG, z0::PhasePoint) = MultinomialTS(z0, zero(neg_energy(z0)))

"""
$(TYPEDEF)

Create a slice sampler for a single leaf tree:
- the slice variable is copied from the passed-in sampler `s` and
- the number of acceptable candicates is computed by comparing the slice variable against the current energy.
"""
function SliceTS(s::SliceTS, H0::AbstractFloat, zcand::PhasePoint)
    return SliceTS(zcand, s.ℓu, Int(s.ℓu <= neg_energy(zcand)))
end

"""
$(TYPEDEF)

Multinomial sampler for a trajectory consisting only a leaf node.
- tree weight is the (unnormalised) energy of the leaf.
"""
function MultinomialTS(s::MultinomialTS, H0::AbstractFloat, zcand::PhasePoint)
    return MultinomialTS(zcand, H0 + neg_energy(zcand))
end

function combine(rng::AbstractRNG, s1::SliceTS, s2::SliceTS)
    @assert s1.ℓu == s2.ℓu "Cannot combine two slice sampler with different slice variable"
    n = s1.n + s2.n
    zcand = n * rand(rng) < s1.n ? s1.zcand : s2.zcand
    return SliceTS(zcand, s1.ℓu, n)
end

function combine(zcand::PhasePoint, s1::SliceTS, s2::SliceTS)
    @assert s1.ℓu == s2.ℓu "Cannot combine two slice sampler with different slice variable"
    n = s1.n + s2.n
    return SliceTS(zcand, s1.ℓu, n)
end

function combine(rng::AbstractRNG, s1::MultinomialTS, s2::MultinomialTS)
    ℓw = logaddexp(s1.ℓw, s2.ℓw)
    zcand = ℓw < s1.ℓw + Random.randexp(rng) ? s1.zcand : s2.zcand
    return MultinomialTS(zcand, ℓw)
end

function combine(zcand::PhasePoint, s1::MultinomialTS, s2::MultinomialTS)
    ℓw = logaddexp(s1.ℓw, s2.ℓw)
    return MultinomialTS(zcand, ℓw)
end

mh_accept(rng::AbstractRNG, s::SliceTS, s′::SliceTS) = s.n * rand(rng) < s′.n

function mh_accept(rng::AbstractRNG, s::MultinomialTS, s′::MultinomialTS)
    return s.ℓw < s′.ℓw + Random.randexp(rng)
end

"""
$(TYPEDEF)

Numerically simulated Hamiltonian trajectories.
"""
struct Trajectory{
    TS<:AbstractTrajectorySampler,
    I<:AbstractIntegrator,
    TC<:AbstractTerminationCriterion,
}
    "Integrator used to simulate trajectory."
    integrator::I
    "Criterion to terminate the simulation."
    termination_criterion::TC
end

Trajectory{TS}(integrator::I, termination_criterion::TC) where {TS,I,TC} =
    Trajectory{TS,I,TC}(integrator, termination_criterion)

ConstructionBase.constructorof(::Type{<:Trajectory{TS}}) where {TS} = Trajectory{TS}

function Base.show(io::IO, τ::Trajectory{TS}) where {TS}
    print(io, "Trajectory{$TS}(integrator=$(τ.integrator), tc=$(τ.termination_criterion))")
end

nsteps(τ::Trajectory{TS,I,TC}) where {TS,I,TC<:FixedNSteps} = τ.termination_criterion.L
nsteps(τ::Trajectory{TS,I,TC}) where {TS,I,TC<:FixedIntegrationTime} =
    max(1, floor(Int, τ.termination_criterion.λ / nom_step_size(τ.integrator)))

##
## Kernel interface
##

struct HMCKernel{R,T<:Trajectory} <: AbstractMCMCKernel
    refreshment::R
    τ::T
end

HMCKernel(τ::Trajectory) = HMCKernel(FullMomentumRefreshment(), τ)

"""
$(SIGNATURES)

Make a MCMC transition from phase point `z` using the trajectory `τ` under Hamiltonian `h`.

NOTE: This is a RNG-implicit fallback function for `transition(Random.default_rng(), τ, h, z)`
"""
function transition(τ::Trajectory, h::Hamiltonian, z::PhasePoint)
    return transition(Random.default_rng(), τ, h, z)
end

###
### Actual trajectory implementations
###

function transition(
    rng::Union{AbstractRNG,AbstractVector{<:AbstractRNG}},
    τ::Trajectory{TS,I,TC},
    h::Hamiltonian,
    z::PhasePoint,
) where {TS<:AbstractTrajectorySampler,I,TC<:StaticTerminationCriterion}
    H0 = energy(z)

    z′, is_accept, α = sample_phasepoint(rng, τ, h, z)
    # Do the actual accept / reject
    z = accept_phasepoint!(z, z′, is_accept)    # NOTE: this function changes `z′` in place in matrix-parallel mode
    # Reverse momentum variable to preserve reversibility
    z = PhasePoint(z.θ, -z.r, z.ℓπ, z.ℓκ)
    # Get cached hamiltonian energy 
    H, H′ = energy(z), energy(z′)
    tstat = merge(
        (
            n_steps = nsteps(τ),
            is_accept = is_accept,
            acceptance_rate = α,
            log_density = z.ℓπ.value,
            hamiltonian_energy = H,
            hamiltonian_energy_error = H - H0,
            # check numerical error in proposed phase point. 
            numerical_error = !all(isfinite, H′),
        ),
        stat(τ.integrator),
    )
    return Transition(z, tstat)
end

# Return the accepted phase point
function accept_phasepoint!(
    z::T,
    z′::T,
    is_accept::Bool,
) where {T<:PhasePoint{<:AbstractVector}}
    if is_accept
        return z′
    else
        return z
    end
end
function accept_phasepoint!(z::T, z′::T, is_accept) where {T<:PhasePoint{<:AbstractMatrix}}
    # Revert unaccepted proposals in `z′`
    if !all(is_accept)
        # Convert logical indexing to number indexing to support CUDA.jl
        # NOTE: for x::CuArray, x[:,Vector{Bool}]  is NOT supported
        #                       x[:,CuVector{Int}] is NOT supported
        #                       x[:,Vector{Int}]   is     supported
        is_reject = Vector(findall(!, is_accept))
        z′.θ[:, is_reject] = z.θ[:, is_reject]
        z′.r[:, is_reject] = z.r[:, is_reject]
        z′.ℓπ.value[is_reject] = z.ℓπ.value[is_reject]
        z′.ℓπ.gradient[:, is_reject] = z.ℓπ.gradient[:, is_reject]
        z′.ℓκ.value[is_reject] = z.ℓκ.value[is_reject]
        z′.ℓκ.gradient[:, is_reject] = z.ℓκ.gradient[:, is_reject]
    end
    # Always return `z′` as any unaccepted proposal is already reverted
    # NOTE: This in place treatment of `z′` is for memory efficient consideration.
    #       We can also copy `z′ and avoid mutating the original `z′`. But this is
    #       not efficient and immutability of `z′` is not important in this local scope.
    return z′
end

### Use end-point from the trajectory as a proposal and apply MH correction

function sample_phasepoint(rng, τ::Trajectory{EndPointTS}, h, z)
    z′ = step(τ.integrator, h, z, nsteps(τ))
    is_accept, α = mh_accept_ratio(rng, energy(z), energy(z′))
    return z′, is_accept, α
end

### Multinomial sampling from trajectory

function randcat(
    rng::AbstractRNG,
    zs::AbstractVector{<:PhasePoint},
    unnorm_ℓp::AbstractVector,
)
    p = exp.(unnorm_ℓp .- logsumexp(unnorm_ℓp))
    i = randcat(rng, p)
    return zs[i]
end

# zs is in the form of Vector{PhasePoint{Matrix}} and has shape [n_steps][dim, n_chains]
function randcat(rng, zs::AbstractVector{<:PhasePoint}, unnorm_ℓP::AbstractMatrix)
    z = similar(first(zs))
    P = exp.(unnorm_ℓP .- logsumexp(unnorm_ℓP; dims = 2)) # (n_chains, n_steps)
    is = randcat(rng, P')
    foreach(enumerate(is)) do (i_chain, i_step)
        zi = zs[i_step]
        z.θ[:, i_chain] = zi.θ[:, i_chain]
        z.r[:, i_chain] = zi.r[:, i_chain]
        z.ℓπ.value[i_chain] = zi.ℓπ.value[i_chain]
        z.ℓπ.gradient[:, i_chain] = zi.ℓπ.gradient[:, i_chain]
        z.ℓκ.value[i_chain] = zi.ℓκ.value[i_chain]
        z.ℓκ.gradient[:, i_chain] = zi.ℓκ.gradient[:, i_chain]
    end
    return z
end

function sample_phasepoint(rng, τ::Trajectory{MultinomialTS}, h, z)
    n_steps = abs(nsteps(τ))
    # TODO: Deal with vectorized-mode generically.
    #       Currently the direction of multiple chains are always coupled
    n_steps_fwd = rand_coupled(rng, 0:n_steps)
    zs_fwd = step(τ.integrator, h, z, n_steps_fwd; fwd = true, full_trajectory = Val(true))
    n_steps_bwd = n_steps - n_steps_fwd
    zs_bwd = step(τ.integrator, h, z, n_steps_bwd; fwd = false, full_trajectory = Val(true))
    zs = vcat(reverse(zs_bwd)..., z, zs_fwd...)
    ℓweights = -energy.(zs)
    if eltype(ℓweights) <: AbstractVector
        ℓweights = cat(ℓweights...; dims = 2)
    end
    unnorm_ℓprob = ℓweights
    z′ = randcat(rng, zs, unnorm_ℓprob)
    # Computing adaptation statistics for dual averaging as done in NUTS
    Hs = -ℓweights
    ΔH = Hs .- energy(z)
    α = exp.(min.(0, -ΔH))  # this is a matrix for vectorized mode and a vector otherwise
    α = typeof(α) <: AbstractVector ? mean(α) : vec(mean(α; dims = 2))
    return z′, true, α
end

###
### Advanced HMC implementation with (adaptive) dynamic trajectory length.
###

##
## Variants of no-U-turn criteria
##

"""
$(TYPEDEF)
Classic No-U-Turn criterion as described in Eq. (9) in [1].

Informally, this will terminate the trajectory expansion if continuing
the simulation either forwards or backwards in time will decrease the
distance between the left-most and right-most positions.

# Fields
$(TYPEDFIELDS)

# References
1. Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn Sampler: adaptively setting path lengths in Hamiltonian Monte Carlo. Journal of Machine Learning Research, 15(1), 1593-1623. ([arXiv](http://arxiv.org/abs/1111.4246))
"""
Base.@kwdef struct ClassicNoUTurn{F<:AbstractFloat} <: DynamicTerminationCriterion
    max_depth::Int = 10
    Δ_max::F = 1000.0
end

"""
$(TYPEDEF)
Generalised No-U-Turn criterion as described in Section A.4.2 in [1].

# Fields
$(TYPEDFIELDS)

# References
1. Betancourt, M. (2017). A Conceptual Introduction to Hamiltonian Monte Carlo. [arXiv preprint arXiv:1701.02434](https://arxiv.org/abs/1701.02434).
"""
Base.@kwdef struct GeneralisedNoUTurn{F<:AbstractFloat} <: DynamicTerminationCriterion
    max_depth::Int = 10
    Δ_max::F = 1000.0
end

"""
$(TYPEDEF)
Generalised No-U-Turn criterion as described in Section A.4.2 in [1] with 
added U-turn check as described in [2].

# Fields
$(TYPEDFIELDS)

# References
1. Betancourt, M. (2017). A Conceptual Introduction to Hamiltonian Monte Carlo. [arXiv preprint arXiv:1701.02434](https://arxiv.org/abs/1701.02434).
2. [https://github.com/stan-dev/stan/pull/2800](https://github.com/stan-dev/stan/pull/2800)
"""
Base.@kwdef struct StrictGeneralisedNoUTurn{F<:AbstractFloat} <: DynamicTerminationCriterion
    max_depth::Int = 10
    Δ_max::F = 1000.0
end
GeneralisedNoUTurn(tc::StrictGeneralisedNoUTurn) =
    GeneralisedNoUTurn(tc.max_depth, tc.Δ_max)

struct TurnStatistic{T}
    "Integral or sum of momenta along the integration path."
    rho::T
end
TurnStatistic() = TurnStatistic(undef)

TurnStatistic(::ClassicNoUTurn, ::PhasePoint) = TurnStatistic()
TurnStatistic(::Union{GeneralisedNoUTurn,StrictGeneralisedNoUTurn}, z::PhasePoint) =
    TurnStatistic(z.r)

combine(ts::TurnStatistic{T}, ::TurnStatistic{T}) where {T<:UndefInitializer} = ts
combine(tsl::T, tsr::T) where {T<:TurnStatistic} = TurnStatistic(tsl.rho + tsr.rho)

###
### The doubling tree algorithm for expanding trajectory.
###

"""
    Termination

Termination reasons
- `dynamic`: due to stoping criteria
- `numerical`: due to large energy deviation from starting (possibly numerical errors)
"""
struct Termination
    dynamic::Bool
    numerical::Bool
end

Base.show(io::IO, d::Termination) =
    print(io, "Termination(dynamic=$(d.dynamic), numerical=$(d.numerical))")
Base.:*(d1::Termination, d2::Termination) =
    Termination(d1.dynamic || d2.dynamic, d1.numerical || d2.numerical)
isterminated(d::Termination) = d.dynamic || d.numerical

"""
$(SIGNATURES)

Check termination of a Hamiltonian trajectory.
"""
function Termination(s::SliceTS, nt::Trajectory, H0::F, H′::F) where {F<:AbstractFloat}
    return Termination(false, !(s.ℓu < nt.termination_criterion.Δ_max + -H′))
end

"""
$(SIGNATURES)

Check termination of a Hamiltonian trajectory.
"""
function Termination(
    s::MultinomialTS,
    nt::Trajectory,
    H0::F,
    H′::F,
) where {F<:AbstractFloat}
    return Termination(false, !(-H0 < nt.termination_criterion.Δ_max + -H′))
end

"""
A full binary tree trajectory with only necessary leaves and information stored.
"""
struct BinaryTree
    zleft::Any   # left most leaf node
    zright::Any  # right most leaf node
    ts::Any      # turn statistics
    sum_α::Any   # MH stats, i.e. sum of MH accept prob for all leapfrog steps
    nα::Any      # total # of leap frog steps, i.e. phase points in a trajectory
    ΔH_max::Any  # energy in tree with largest absolute different from initial energy
end

"""
    maxabs(a, b)

Return the value with the largest absolute value.
"""
maxabs(a, b) = abs(a) > abs(b) ? a : b

"""
$(SIGNATURES)
Merge a left tree `treeleft` and a right tree `treeright` under given Hamiltonian `h`,
then draw a new candidate sample and update related statistics for the resulting tree.
"""
function combine(treeleft::BinaryTree, treeright::BinaryTree)
    return BinaryTree(
        treeleft.zleft,
        treeright.zright,
        combine(treeleft.ts, treeright.ts),
        treeleft.sum_α + treeright.sum_α,
        treeleft.nα + treeright.nα,
        maxabs(treeleft.ΔH_max, treeright.ΔH_max),
    )
end

"""
$(SIGNATURES)
Detect U turn for two phase points (`zleft` and `zright`) under given Hamiltonian `h`
using the (original) no-U-turn cirterion.

Ref: https://arxiv.org/abs/1111.4246, https://arxiv.org/abs/1701.02434
"""
function isterminated(::ClassicNoUTurn, h::Hamiltonian, t::BinaryTree)
    # z0 is starting point and z1 is ending point
    z0, z1 = t.zleft, t.zright
    Δθ = z1.θ - z0.θ
    s = (dot(Δθ, ∂H∂r(h, -z0.r)) >= 0) || (dot(-Δθ, ∂H∂r(h, z1.r)) >= 0)
    return Termination(s, false)
end

"""
$(SIGNATURES)
Detect U turn for two phase points (`zleft` and `zright`) under given Hamiltonian `h`
using the generalised no-U-turn criterion.

Ref: https://arxiv.org/abs/1701.02434
"""
function isterminated(::GeneralisedNoUTurn, h::Hamiltonian, t::BinaryTree)
    rho = t.ts.rho
    s = generalised_uturn_criterion(rho, ∂H∂r(h, t.zleft.r), ∂H∂r(h, t.zright.r))
    return Termination(s, false)
end

"""
$(SIGNATURES)
Detect U turn for two phase points (`zleft` and `zright`) under given Hamiltonian `h`
using the generalised no-U-turn criterion with additional U-turn checks.

Ref: https://arxiv.org/abs/1701.02434 https://github.com/stan-dev/stan/pull/2800
"""
function isterminated(tc::StrictGeneralisedNoUTurn, h::Hamiltonian, t, tleft, tright)
    # (Non-strict) generalised U-turn check
    s1 = isterminated(GeneralisedNoUTurn(tc), h, t)

    # U-turn checks for left and right subtree
    # See https://discourse.mc-stan.org/t/nuts-misses-u-turns-runs-in-circles-until-max-treedepth/9727/33
    # for a visualisation.
    s2 = check_left_subtree(h, t, tleft, tright)
    s3 = check_right_subtree(h, t, tleft, tright)
    return s1 * s2 * s3
end

"""
$(SIGNATURES)
Do a U-turn check between the leftmost phase point of `t` and the leftmost 
phase point of `tright`, the right subtree.
"""
function check_left_subtree(h::Hamiltonian, t::T, tleft::T, tright::T) where {T<:BinaryTree}
    rho = tleft.ts.rho + tright.zleft.r
    s = generalised_uturn_criterion(rho, ∂H∂r(h, t.zleft.r), ∂H∂r(h, tright.zleft.r))
    return Termination(s, false)
end

"""
$(SIGNATURES)
Do a U-turn check between the rightmost phase point of `t` and the rightmost
phase point of `tleft`, the left subtree.
"""
function check_right_subtree(
    h::Hamiltonian,
    t::T,
    tleft::T,
    tright::T,
) where {T<:BinaryTree}
    rho = tleft.zright.r + tright.ts.rho
    s = generalised_uturn_criterion(rho, ∂H∂r(h, tleft.zright.r), ∂H∂r(h, t.zright.r))
    return Termination(s, false)
end

function generalised_uturn_criterion(rho, p_sharp_minus, p_sharp_plus)
    return (dot(rho, p_sharp_minus) <= 0) || (dot(rho, p_sharp_plus) <= 0)
end

function isterminated(
    tc::TC,
    h::Hamiltonian,
    t::BinaryTree,
    _tleft,
    _tright,
) where {TC<:Union{ClassicNoUTurn,GeneralisedNoUTurn}}
    return isterminated(tc, h, t)
end

"Recursivly build a tree for a given depth `j`."
function build_tree(
    rng::AbstractRNG,
    nt::Trajectory{TS,I,TC},
    h::Hamiltonian,
    z::PhasePoint,
    sampler::AbstractTrajectorySampler,
    v::Int,
    j::Int,
    H0::AbstractFloat,
) where {
    TS<:AbstractTrajectorySampler,
    I<:AbstractIntegrator,
    TC<:DynamicTerminationCriterion,
}
    if j == 0
        # Base case - take one leapfrog step in the direction v.
        z′ = step(nt.integrator, h, z, v)
        H′ = energy(z′)
        ΔH = H′ - H0
        α′ = exp(min(0, -ΔH))
        sampler′ = TS(sampler, H0, z′)
        return BinaryTree(z′, z′, TurnStatistic(nt.termination_criterion, z′), α′, 1, ΔH),
        sampler′,
        Termination(sampler′, nt, H0, H′)
    else
        # Recursion - build the left and right subtrees.
        tree′, sampler′, termination′ = build_tree(rng, nt, h, z, sampler, v, j - 1, H0)
        # Expand tree if not terminated
        if !isterminated(termination′)
            # Expand left
            if v == -1
                tree′′, sampler′′, termination′′ =
                    build_tree(rng, nt, h, tree′.zleft, sampler, v, j - 1, H0) # left tree
                treeleft, treeright = tree′′, tree′
                # Expand right
            else
                tree′′, sampler′′, termination′′ =
                    build_tree(rng, nt, h, tree′.zright, sampler, v, j - 1, H0) # right tree
                treeleft, treeright = tree′, tree′′
            end
            tree′ = combine(treeleft, treeright)
            sampler′ = combine(rng, sampler′, sampler′′)
            termination′ =
                termination′ *
                termination′′ *
                isterminated(nt.termination_criterion, h, tree′, treeleft, treeright)
        end
        return tree′, sampler′, termination′
    end
end

function transition(
    rng::AbstractRNG,
    τ::Trajectory{TS,I,TC},
    h::Hamiltonian,
    z0::PhasePoint,
) where {
    TS<:AbstractTrajectorySampler,
    I<:AbstractIntegrator,
    TC<:DynamicTerminationCriterion,
}
    H0 = energy(z0)
    tree = BinaryTree(
        z0,
        z0,
        TurnStatistic(τ.termination_criterion, z0),
        zero(H0),
        zero(Int),
        zero(H0),
    )
    sampler = TS(rng, z0)
    termination = Termination(false, false)
    zcand = z0

    j = 0
    while !isterminated(termination) && j < τ.termination_criterion.max_depth
        # Sample a direction; `-1` means left and `1` means right
        vleft = rand(rng, Bool)
        if vleft
            # Create a tree with depth `j` on the left
            tree′, sampler′, termination′ =
                build_tree(rng, τ, h, tree.zleft, sampler, -1, j, H0)
            treeleft, treeright = tree′, tree
        else
            # Create a tree with depth `j` on the right
            tree′, sampler′, termination′ =
                build_tree(rng, τ, h, tree.zright, sampler, 1, j, H0)
            treeleft, treeright = tree, tree′
        end
        # Perform a MH step and increse depth if not terminated
        if !isterminated(termination′)
            j = j + 1   # increment tree depth
            if mh_accept(rng, sampler, sampler′)
                zcand = sampler′.zcand
            end
        end
        # Combine the proposed tree and the current tree (no matter terminated or not)
        tree = combine(treeleft, treeright)
        # Update sampler
        sampler = combine(zcand, sampler, sampler′)
        # update termination
        termination =
            termination *
            termination′ *
            isterminated(τ.termination_criterion, h, tree, treeleft, treeright)
    end

    H = energy(zcand)
    tstat = merge(
        (
            n_steps = tree.nα,
            is_accept = true,
            acceptance_rate = tree.sum_α / tree.nα,
            log_density = zcand.ℓπ.value,
            hamiltonian_energy = H,
            hamiltonian_energy_error = H - H0,
            max_hamiltonian_energy_error = tree.ΔH_max,
            tree_depth = j,
            numerical_error = termination.numerical,
        ),
        stat(τ.integrator),
    )

    return Transition(zcand, tstat)
end

###
### Initialisation of step size
###

"""
A single Hamiltonian integration step.

NOTE: this function is intended to be used in `find_good_stepsize` only.
"""
function A(h, z, ϵ)
    z′ = step(Leapfrog(ϵ), h, z)
    H′ = energy(z′)
    return z′, H′
end

"Find a good initial leap-frog step-size via heuristic search."
function find_good_stepsize(
    rng::AbstractRNG,
    h::Hamiltonian,
    θ::AbstractVector{T};
    max_n_iters::Int = 100,
) where {T<:Real}
    # Initialize searching parameters
    ϵ′ = ϵ = T(1 // 10)
    # minimal, crossing, maximal log accept ratio
    log_a_min = 2 * T(loghalf)
    log_a_cross = T(loghalf)
    log_a_max = log(T(3 // 4))
    d = T(2)
    invd = inv(d)
    # Create starting phase point
    r = rand(rng, h.metric, h.kinetic) # sample momentum variable
    z = phasepoint(h, θ, r)
    H = energy(z)

    # Make a proposal phase point to decide direction
    _, H′ = A(h, z, ϵ)
    ΔH = H - H′ # compute the energy difference; `exp(ΔH)` is the MH accept ratio
    ratio_too_high = ΔH > log_a_cross

    # Crossing step: increase/decrease ϵ until accept ratio cross a_cross.
    for _ = 1:max_n_iters
        # `ratio_too_high` being  `true` means MH ratio too high
        #     - this means our step size is too small, thus we increase
        # `ratio_to_high` being `false` means MH ratio too small
        #     - this means our step size is too large, thus we decrease
        ϵ′ = ratio_too_high ? d * ϵ : invd * ϵ
        _, H′ = A(h, z, ϵ)
        ΔH = H - H′
        @debug "Crossing step" direction H′ ϵ α = min(1, exp(ΔH))
        if xor(ratio_too_high, ΔH > log_a_cross)
            break
        else
            ϵ = ϵ′
        end
    end
    # Note after the for loop,
    # `ϵ` and `ϵ′` are the two neighbour step sizes across `a_cross`.

    # Bisection step: ensure final accept ratio: a_min < a < a_max.
    # See https://en.wikipedia.org/wiki/Bisection_method

    ϵ, ϵ′ = minmax(ϵ, ϵ′) # ensure ϵ < ϵ′;
    # Here we want to use a value between these two given the
    # criteria that this value also gives us a MH ratio between `a_min` and `a_max`.
    # This condition is quite mild and only intended to avoid cases where
    # the middle value of `ϵ` and `ϵ′` is too extreme.
    for _ = 1:max_n_iters
        ϵ_mid = middle(ϵ, ϵ′)
        _, H′ = A(h, z, ϵ_mid)
        ΔH = H - H′
        @debug "Bisection step" H′ ϵ_mid α = min(1, exp(ΔH))
        if ΔH > log_a_max
            ϵ = ϵ_mid
        elseif ΔH < log_a_min
            ϵ′ = ϵ_mid
        else
            ϵ = ϵ_mid
            break
        end
    end

    return ϵ
end

function find_good_stepsize(
    h::Hamiltonian,
    θ::AbstractVector{<:AbstractFloat};
    max_n_iters::Int = 100,
)
    return find_good_stepsize(Random.default_rng(), h, θ; max_n_iters = max_n_iters)
end

"Perform MH acceptance based on energy, i.e. negative log probability."
function mh_accept_ratio(
    rng::AbstractRNG,
    Horiginal::T,
    Hproposal::T,
) where {T<:AbstractFloat}
    accept = Hproposal < Horiginal + Random.randexp(rng, T)
    α = min(one(T), exp(Horiginal - Hproposal))
    return accept, α
end

function mh_accept_ratio(
    rng::Union{AbstractRNG,AbstractVector{<:AbstractRNG}},
    Horiginal::AbstractVector{<:T},
    Hproposal::AbstractVector{<:T},
) where {T<:AbstractFloat}
    # NOTE: There is a chance that sharing the RNG over multiple
    #       chains for accepting / rejecting might couple
    #       the chains. We need to revisit this more rigirously 
    #       in the future. See discussions at 
    #       https://github.com/TuringLang/AdvancedHMC.jl/pull/166#pullrequestreview-367216534
    accept = if rng isa AbstractRNG
        Hproposal .< Horiginal .+ Random.randexp(rng, T, length(Hproposal))
    else
        Hproposal .< Horiginal .+ Random.randexp.(rng, (T,))
    end
    α = min.(one(T), exp.(Horiginal .- Hproposal))
    return accept, α
end
