function mh_accept(rng::AbstractRNG, H::AbstractFloat, H_new::AbstractFloat)
    logα = min(0, H - H_new)
    return log(rand(rng)) < logα, exp(logα)
end
mh_accept(H::AbstractFloat, H_new::AbstractFloat) = mh_accept(GLOBAL_RNG, logα)

function sample(h::Hamiltonian, prop::AbstractProposal, θ::AbstractVector{T}, n_samples::Int) where {T<:Real}
    θs = Vector{Vector{T}}(undef, n_samples)
    Hs = Vector{T}(undef, n_samples)
    αs = Vector{T}(undef, n_samples)
    time = @elapsed for i = 1:n_samples
        θs[i], Hs[i], αs[i] = step(h, prop, i == 1 ? θ : θs[i-1])
    end
    @info "Finished sampling with $time (s)" typeof(h) typeof(prop) EBFMI(Hs) mean(αs)
    return θs
end

function sample(h::Hamiltonian, prop::AbstractProposal, θ::AbstractVector{T}, n_samples::Int, adapter::AbstractAdapter,
                n_adapts::Int=min(div(n_samples, 10), 1_000)) where {T<:Real}
    θs = Vector{Vector{T}}(undef, n_samples)
    Hs = Vector{T}(undef, n_samples)
    αs = Vector{T}(undef, n_samples)
    time = @elapsed for i = 1:n_samples
        θs[i], Hs[i], αs[i] = step(h, prop, i == 1 ? θ : θs[i-1])
        if i <= n_adapts
            adapt!(adapter, αs[i])
            ϵ = getss(adapter)
            prop = prop(ϵ)
            i == n_adapts && @info "Finished $n_adapts adapation steps" typeof(adapter) ϵ
        end
    end
    @info "Finished $n_samples sampling steps in $time (s)" typeof(h) typeof(prop) EBFMI(Hs) mean(αs)
    return θs
end

function step(rng::AbstractRNG, h::Hamiltonian, prop::TakeLastProposal{I}, θ::AbstractVector{T}) where {T<:Real,I<:AbstractIntegrator}
    r = rand_momentum(rng, h)
    H = hamiltonian_energy(h, θ, r)
    θ_new, r_new = propose(prop, h, θ, r)
    H_new = hamiltonian_energy(h, θ_new, r_new)
    # Accept via MH criteria
    is_accept, α = mh_accept(rng, H, H_new)
    if !is_accept
        return θ, H, α
    end
    return θ_new, H_new, α
end

function step(rng::AbstractRNG, h::Hamiltonian, prop::SliceNUTS{I}, θ::AbstractVector{T}) where {T<:Real,I<:AbstractIntegrator}
    r = rand_momentum(rng, h)
    θ_new, r_new, α = propose(rng, prop, h, θ, r)
    H_new = hamiltonian_energy(h, θ_new, r_new)
    # We always accept in NUTS
    return θ_new, H_new, α
end

step(h::Hamiltonian, p::AbstractProposal, θ::AbstractVector{T}) where {T<:Real} = step(GLOBAL_RNG, h, p, θ)
