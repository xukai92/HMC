using Test, AdvancedHMC
using Random
include("common.jl")

ϵ = 0.01
lf = Leapfrog(ϵ)

θ_init = randn(D)
h = Hamiltonian(UnitEuclideanMetric(D), logπ, ∂logπ∂θ)
τ = NUTS(Leapfrog(find_good_eps(h, θ_init)))
r_init = AdvancedHMC.rand(h.metric)

@testset "Passing random number generator" begin
    for seed in [1234, 5678, 90]
        rng = MersenneTwister(seed)
        # θ1, r1 = AdvancedHMC.transition(rng, τ, h, θ_init, r_init)
        z = AdvancedHMC.phasepoint(h, θ_init, r_init)
        z1′, _ = AdvancedHMC.transition(rng, τ, h, z)

        rng = MersenneTwister(seed)
        # θ2, r2 = AdvancedHMC.transition(rng, τ, h, θ_init, r_init)
        z = AdvancedHMC.phasepoint(h, θ_init, r_init)
        z2′, _ = AdvancedHMC.transition(rng, τ, h, z)

        @test z1′.θ == z2′.θ
        @test z1′.r == z2′.r
    end
end
