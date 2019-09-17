####
#### Numerical methods for simulating Hamiltonian trajectory.
####


abstract type AbstractIntegrator end

abstract type AbstractLeapfrog{T} <: AbstractIntegrator end

jitter(::AbstractLeapfrog, ϵ) = ϵ
temper_first(::AbstractLeapfrog, r, ::Int, ::Int) = r
temper_second(::AbstractLeapfrog, r, ::Int, ::Int) = r

function step(
    lf::AbstractLeapfrog{T},
    h::Hamiltonian,
    z::PhasePoint,
    n_steps::Int=1;
    fwd::Bool=n_steps > 0   # simulate hamiltonian backward when n_steps < 0,
) where {T<:AbstractFloat}
    n_steps = abs(n_steps)  # to support `n_steps < 0` cases
    ϵ = fwd ? lf.ϵ : -lf.ϵ
    ϵ = jitter(lf, ϵ)

    @unpack θ, r = z
    @unpack value, gradient = ∂H∂θ(h, θ)
    for i = 1:n_steps
        r = temper_first(lf, r, i, n_steps)
        r = r - ϵ / 2 * gradient    # take a half leapfrog step for momentum variable
        ∇r = ∂H∂r(h, r)
        θ = θ + ϵ * ∇r              # take a full leapfrog step for position variable
        @unpack value, gradient = ∂H∂θ(h, θ)
        r = r - ϵ / 2 * gradient    # take a half leapfrog step for momentum variable
        r = temper_second(lf, r, i, n_steps)
        # Create a new phase point by caching the logdensity and gradient
        z = phasepoint(h, θ, r; ℓπ=DualValue(value, gradient))
        !isfinite(z) && break
    end
    return z
end

struct Leapfrog{T<:AbstractFloat} <: AbstractLeapfrog{T}
    ϵ       ::  T
end
Base.show(io::IO, l::Leapfrog) = print(io, "Leapfrog(ϵ=$(round(l.ϵ; sigdigits=3)))")

### Jittering

struct JitteredLeapfrog{T<:AbstractFloat} <: AbstractLeapfrog{T}
    ϵ       ::  T
    jitter  ::  T
end

function Base.show(io::IO, l::JitteredLeapfrog)
    print(io, "JitteredLeapfrog(ϵ=$(round(l.ϵ; sigdigits=3)), jitter=$(round(l.jitter; sigdigits=3)))")
end

# Jitter step size; ref: https://github.com/stan-dev/stan/blob/1bb054027b01326e66ec610e95ef9b2a60aa6bec/src/stan/mcmc/hmc/base_hmc.hpp#L177-L178
jitter(lf::JitteredLeapfrog, ϵ) = ϵ * (1 + lf.jitter * (2 * rand() - 1))

### Tempering

struct TemperedLeapfrog{T<:AbstractFloat} <: AbstractLeapfrog{T}
    ϵ       ::  T
    α       ::  T
end

function Base.show(io::IO, l::TemperedLeapfrog)
    print(io, "TemperedLeapfrog(ϵ=$(round(l.ϵ; sigdigits=3)), α=$(round(l.α; sigdigits=3)))")
end

function temper_first(lf::TemperedLeapfrog, r, i::Int, n_steps::Int)
    # `ceil` includes mid if `n_steps` is odd, e.g. `<= ceil(5 / 2)` => `<= 3` 
    return i <= ceil(Int, n_steps / 2) ? r * sqrt(lf.α) : r / sqrt(lf.α)
end

function temper_second(lf::TemperedLeapfrog, r, i::Int, n_steps::Int)
    # `floor` excludes mid if `n_steps` is odd, e.g. `<= floor(5 / 2)` => `<= 2` 
    return i <= floor(Int, n_steps / 2) ? r * sqrt(lf.α) : r / sqrt(lf.α)
end
