using Random

### integrator.jl

import AdvancedHMC: ∂H∂θ, ∂H∂r, DualValue, PhasePoint, phasepoint, step
using AdvancedHMC:
    @unpack, TYPEDEF, TYPEDFIELDS, AbstractScalarOrVec, AbstractLeapfrog, step_size

"""
$(TYPEDEF)

Generalized leapfrog integrator with fixed step size `ϵ`.

# Fields

$(TYPEDFIELDS)
"""
struct GeneralizedLeapfrog{T<:AbstractScalarOrVec{<:AbstractFloat}} <: AbstractLeapfrog{T}
    "Step size."
    ϵ::T
    n::Int
end
Base.show(io::IO, l::GeneralizedLeapfrog) =
    print(io, "GeneralizedLeapfrog(ϵ=$(round.(l.ϵ; sigdigits=3)), n=$(l.n))")

# Fallback to ignore return_cache & cache kwargs for other ∂H∂θ
function ∂H∂θ_cache(h, θ, r; return_cache = false, cache = nothing) where {T}
    dv = ∂H∂θ(h, θ, r)
    return return_cache ? (dv, nothing) : dv
end

# TODO Make sure vectorization works
# TODO Check if tempering is valid
function step(
    lf::GeneralizedLeapfrog{T},
    h::Hamiltonian,
    z::P,
    n_steps::Int = 1;
    fwd::Bool = n_steps > 0,  # simulate hamiltonian backward when n_steps < 0
    full_trajectory::Val{FullTraj} = Val(false),
) where {T<:AbstractScalarOrVec{<:AbstractFloat},P<:PhasePoint,FullTraj}
    n_steps = abs(n_steps)  # to support `n_steps < 0` cases

    ϵ = fwd ? step_size(lf) : -step_size(lf)
    ϵ = ϵ'

    res = if FullTraj
        Vector{P}(undef, n_steps)
    else
        z
    end

    for i = 1:n_steps
        θ_init, r_init = z.θ, z.r
        # Tempering
        #r = temper(lf, r, (i=i, is_half=true), n_steps)
        #! Eq (16) of Girolami & Calderhead (2011)
        r_half = copy(r_init)
        local cache
        for j = 1:lf.n
            # Reuse cache for the first iteration
            if j == 1
                @unpack value, gradient = z.ℓπ
            elseif j == 2 # cache intermediate values that depends on θ only (which are unchanged)
                retval, cache = ∂H∂θ_cache(h, θ_init, r_half; return_cache = true)
                @unpack value, gradient = retval
            else # reuse cache
                @unpack value, gradient = ∂H∂θ_cache(h, θ_init, r_half; cache = cache)
            end
            r_half = r_init - ϵ / 2 * gradient
            # println("r_half: ", r_half)
        end
        #! Eq (17) of Girolami & Calderhead (2011)
        θ_full = copy(θ_init)
        term_1 = ∂H∂r(h, θ_init, r_half) # unchanged across the loop
        for j = 1:lf.n
            θ_full = θ_init + ϵ / 2 * (term_1 + ∂H∂r(h, θ_full, r_half))
            # println("θ_full :", θ_full)
        end
        #! Eq (18) of Girolami & Calderhead (2011)
        @unpack value, gradient = ∂H∂θ(h, θ_full, r_half)
        r_full = r_half - ϵ / 2 * gradient
        # println("r_full: ", r_full)
        # Tempering
        #r = temper(lf, r, (i=i, is_half=false), n_steps)
        # Create a new phase point by caching the logdensity and gradient
        z = phasepoint(h, θ_full, r_full; ℓπ = DualValue(value, gradient))
        # Update result
        if FullTraj
            res[i] = z
        else
            res = z
        end
        if !isfinite(z)
            # Remove undef
            if FullTraj
                res = res[isassigned.(Ref(res), 1:n_steps)]
            end
            break
        end
        # @assert false
    end
    return res
end

# TODO Make the order of θ and r consistent with neg_energy
∂H∂θ(h::Hamiltonian, θ::AbstractVecOrMat, r::AbstractVecOrMat) = ∂H∂θ(h, θ)
∂H∂r(h::Hamiltonian, θ::AbstractVecOrMat, r::AbstractVecOrMat) = ∂H∂r(h, r)

### hamiltonian.jl

import AdvancedHMC: refresh, phasepoint
using AdvancedHMC: FullMomentumRefreshment, PartialMomentumRefreshment, AbstractMetric

# To change L180 of hamiltonian.jl
phasepoint(
    rng::Union{AbstractRNG,AbstractVector{<:AbstractRNG}},
    θ::AbstractVecOrMat{T},
    h::Hamiltonian,
) where {T<:Real} = phasepoint(h, θ, rand_momentum(rng, h.metric, h.kinetic, θ))

# To change L191 of hamiltonian.jl
refresh(
    rng::Union{AbstractRNG,AbstractVector{<:AbstractRNG}},
    ::FullMomentumRefreshment,
    h::Hamiltonian,
    z::PhasePoint,
) = phasepoint(h, z.θ, rand_momentum(rng, h.metric, h.kinetic, z.θ))

# To change L215 of hamiltonian.jl
refresh(
    rng::Union{AbstractRNG,AbstractVector{<:AbstractRNG}},
    ref::PartialMomentumRefreshment,
    h::Hamiltonian,
    z::PhasePoint,
) = phasepoint(
    h,
    z.θ,
    ref.α * z.r + sqrt(1 - ref.α^2) * rand_momentum(rng, h.metric, h.kinetic, z.θ),
)

### metric.jl

import AdvancedHMC: _rand
using AdvancedHMC: AbstractMetric
using LinearAlgebra: eigen, cholesky, Symmetric

abstract type AbstractRiemannianMetric <: AbstractMetric end

abstract type AbstractHessianMap end

struct IdentityMap <: AbstractHessianMap end

(::IdentityMap)(x) = x

struct SoftAbsMap{T} <: AbstractHessianMap
    α::T
end

# TODO Register softabs with ReverseDiff
#! The definition of SoftAbs from Page 3 of Betancourt (2012)
function softabs(X, α = 20.0)
    F = eigen(X) # ReverseDiff cannot diff through `eigen`
    Q = hcat(F.vectors)
    λ = F.values
    softabsλ = λ .* coth.(α * λ)
    return Q * diagm(softabsλ) * Q', Q, λ, softabsλ
end

(map::SoftAbsMap)(x) = softabs(x, map.α)[1]

struct DenseRiemannianMetric{
    T,
    TM<:AbstractHessianMap,
    A<:Union{Tuple{Int},Tuple{Int,Int}},
    AV<:AbstractVecOrMat{T},
    TG,
    T∂G∂θ,
} <: AbstractRiemannianMetric
    size::A
    G::TG # TODO store G⁻¹ here instead
    ∂G∂θ::T∂G∂θ
    map::TM
    _temp::AV
end

# TODO Make dense mass matrix support matrix-mode parallel
function DenseRiemannianMetric(size, G, ∂G∂θ, map = IdentityMap()) where {T<:AbstractFloat}
    _temp = Vector{Float64}(undef, size[1])
    return DenseRiemannianMetric(size, G, ∂G∂θ, map, _temp)
end
# DenseEuclideanMetric(::Type{T}, D::Int) where {T} = DenseEuclideanMetric(Matrix{T}(I, D, D))
# DenseEuclideanMetric(D::Int) = DenseEuclideanMetric(Float64, D)
# DenseEuclideanMetric(::Type{T}, sz::Tuple{Int}) where {T} = DenseEuclideanMetric(Matrix{T}(I, first(sz), first(sz)))
# DenseEuclideanMetric(sz::Tuple{Int}) = DenseEuclideanMetric(Float64, sz)

# renew(ue::DenseEuclideanMetric, M⁻¹) = DenseEuclideanMetric(M⁻¹)

Base.size(e::DenseRiemannianMetric) = e.size
Base.size(e::DenseRiemannianMetric, dim::Int) = e.size[dim]
Base.show(io::IO, dem::DenseRiemannianMetric) = print(io, "DenseRiemannianMetric(...)")

function rand_momentum(
    rng::Union{AbstractRNG,AbstractVector{<:AbstractRNG}},
    metric::DenseRiemannianMetric{T},
    kinetic,
    θ::AbstractVecOrMat,
) where {T}
    r = _randn(rng, T, size(metric)...)
    G⁻¹ = inv(metric.map(metric.G(θ)))
    chol = cholesky(Symmetric(G⁻¹))
    ldiv!(chol.U, r)
    return r
end

### hamiltonian.jl

import AdvancedHMC: phasepoint, neg_energy, ∂H∂θ, ∂H∂r
using LinearAlgebra: logabsdet, tr

# QUES Do we want to change everything to position dependent by default?
# Add θ to ∂H∂r for DenseRiemannianMetric
phasepoint(
    h::Hamiltonian{<:DenseRiemannianMetric},
    θ::T,
    r::T;
    ℓπ = ∂H∂θ(h, θ),
    ℓκ = DualValue(neg_energy(h, r, θ), ∂H∂r(h, θ, r)),
) where {T<:AbstractVecOrMat} = PhasePoint(θ, r, ℓπ, ℓκ)

# Negative kinetic energy
#! Eq (13) of Girolami & Calderhead (2011)
function neg_energy(
    h::Hamiltonian{<:DenseRiemannianMetric},
    r::T,
    θ::T,
) where {T<:AbstractVecOrMat}
    G = h.metric.map(h.metric.G(θ))
    D = size(G, 1)
    # Need to consider the normalizing term as it is no longer same for different θs
    logZ = 1 / 2 * (D * log(2π) + logdet(G)) # it will be user's responsibility to make sure G is SPD and logdet(G) is defined
    mul!(h.metric._temp, inv(G), r)
    return -logZ - dot(r, h.metric._temp) / 2
end

# QUES L31 of hamiltonian.jl now reads a bit weird (semantically)
function ∂H∂θ(
    h::Hamiltonian{<:DenseRiemannianMetric{T,<:IdentityMap}},
    θ::AbstractVecOrMat{T},
    r::AbstractVecOrMat{T},
) where {T}
    ℓπ, ∂ℓπ∂θ = h.∂ℓπ∂θ(θ)
    G = h.metric.map(h.metric.G(θ))
    invG = inv(G)
    ∂G∂θ = h.metric.∂G∂θ(θ)
    d = length(∂ℓπ∂θ)
    return DualValue(
        ℓπ,
        #! Eq (15) of Girolami & Calderhead (2011)
        -mapreduce(vcat, 1:d) do i
            ∂G∂θᵢ = ∂G∂θ[:, :, i]
            ∂ℓπ∂θ[i] - 1 / 2 * tr(invG * ∂G∂θᵢ) + 1 / 2 * r' * invG * ∂G∂θᵢ * invG * r
            # Gr = G \ r
            # ∂ℓπ∂θ[i] - 1 / 2 * tr(G \ ∂G∂θᵢ) + 1 / 2 * Gr' * ∂G∂θᵢ * Gr
            # 1 / 2 * tr(invG * ∂G∂θᵢ)
            # 1 / 2 * r' * invG * ∂G∂θᵢ * invG * r
        end,
    )
end

# Ref: https://www.wolframalpha.com/input?i=derivative+of+x+*+coth%28a+*+x%29
#! Based on middle of the right column of Page 3 of Betancourt (2012) "Note that whenλi=λj, such as for the diagonal elementsor degenerate eigenvalues, this becomes the derivative"
dsoftabsdλ(α, λ) = coth(α * λ) + λ * α * -csch(λ * α)^2

#! J as defined in middle of the right column of Page 3 of Betancourt (2012)
function make_J(λ::AbstractVector{T}, α::T) where {T<:AbstractFloat}
    d = length(λ)
    J = Matrix{T}(undef, d, d)
    for i = 1:d, j = 1:d
        J[i, j] =
            (λ[i] == λ[j]) ? dsoftabsdλ(α, λ[i]) :
            ((λ[i] * coth(α * λ[i]) - λ[j] * coth(α * λ[j])) / (λ[i] - λ[j]))
    end
    return J
end

∂H∂θ(
    h::Hamiltonian{<:DenseRiemannianMetric{T,<:SoftAbsMap}},
    θ::AbstractVecOrMat{T},
    r::AbstractVecOrMat{T},
) where {T} = ∂H∂θ_cache(h, θ, r)
function ∂H∂θ_cache(
    h::Hamiltonian{<:DenseRiemannianMetric{T,<:SoftAbsMap}},
    θ::AbstractVecOrMat{T},
    r::AbstractVecOrMat{T};
    return_cache = false,
    cache = nothing,
) where {T}
    # Terms that only dependent on θ can be cached in θ-unchanged loops
    if isnothing(cache)
        ℓπ, ∂ℓπ∂θ = h.∂ℓπ∂θ(θ)
        H = h.metric.G(θ)
        ∂H∂θ = h.metric.∂G∂θ(θ)

        G, Q, λ, softabsλ = softabs(H, h.metric.map.α)

        R = diagm(1 ./ softabsλ)

        # softabsΛ = diagm(softabsλ)
        # M = inv(softabsΛ) * Q' * r
        # M = R * Q' * r # equiv to above but avoid inv

        J = make_J(λ, h.metric.map.α)

        #! Based on the two equations from the right column of Page 3 of Betancourt (2012)
        term_1_cached = Q * (R .* J) * Q'
    else
        ℓπ, ∂ℓπ∂θ, ∂H∂θ, Q, softabsλ, J, term_1_cached = cache
    end
    d = length(∂ℓπ∂θ)
    D = diagm((Q' * r) ./ softabsλ)
    term_2_cached = Q * D * J * D * Q'
    g =
        -mapreduce(vcat, 1:d) do i
            ∂H∂θᵢ = ∂H∂θ[:, :, i]
            # ∂ℓπ∂θ[i] - 1 / 2 * tr(term_1_cached * ∂H∂θᵢ) + 1 / 2 * M' * (J .* (Q' * ∂H∂θᵢ * Q)) * M # (v1)
            # NOTE Some further optimization can be done here: cache the 1st product all together
            ∂ℓπ∂θ[i] - 1 / 2 * tr(term_1_cached * ∂H∂θᵢ) + 1 / 2 * tr(term_2_cached * ∂H∂θᵢ) # (v2) cache friendly
        end

    dv = DualValue(ℓπ, g)
    return return_cache ? (dv, (; ℓπ, ∂ℓπ∂θ, ∂H∂θ, Q, softabsλ, J, term_1_cached)) : dv
end

#! Eq (14) of Girolami & Calderhead (2011)
function ∂H∂r(
    h::Hamiltonian{<:DenseRiemannianMetric},
    θ::AbstractVecOrMat,
    r::AbstractVecOrMat,
)
    H = h.metric.G(θ)
    # if !all(isfinite, H)
    #     println("θ: ", θ)
    #     println("H: ", H)
    # end
    G = h.metric.map(H)
    # return inv(G) * r
    # println("G \ r: ", G \ r)
    return G \ r # NOTE it's actually pretty weird that ∂H∂θ returns DualValue but ∂H∂r doesn't
end
