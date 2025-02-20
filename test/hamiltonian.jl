using ReTest, AdvancedHMC
using AdvancedHMC: GaussianKinetic, DualValue, PhasePoint
using LinearAlgebra: dot, diagm
using ComponentArrays

@testset "Hamiltonian" begin
    f = x -> dot(x, x)
    g = x -> 2x
    metric = UnitEuclideanMetric(10)
    h1 = Hamiltonian(metric, f, g)
    h2 = Hamiltonian(metric, GaussianKinetic(), f, g)
    @test h1 == h2
end

@testset "PhasePoint" begin
    for T in [Float32, Float64]
        init_z1() = PhasePoint(
            [T(NaN)],
            [T(NaN)],
            DualValue(zero(T), [zero(T)]),
            DualValue(zero(T), [zero(T)]),
        )
        init_z2() = PhasePoint(
            [T(Inf)],
            [T(Inf)],
            DualValue(zero(T), [zero(T)]),
            DualValue(zero(T), [zero(T)]),
        )

        # (HongGe) we no longer throw warning messages for numerical errors.
        # @test_logs (:warn, "The current proposal will be rejected due to numerical error(s).") init_z1()
        # @test_logs (:warn, "The current proposal will be rejected due to numerical error(s).") init_z2()

        z1 = init_z1()
        z2 = init_z2()

        @test z1.ℓπ.value == z2.ℓπ.value
        @test z1.ℓκ.value == z2.ℓκ.value
    end
end

@testset "Metric Base Array" begin
    n_tests = 10

    for T in [Float32, Float64]
        for _ = 1:n_tests
            θ_init = randn(T, D)
            r_init = randn(T, D)

            h = Hamiltonian(UnitEuclideanMetric(T, D), ℓπ, ∂ℓπ∂θ)
            @test -AdvancedHMC.neg_energy(h, r_init, θ_init) == sum(abs2, r_init) / 2
            @test AdvancedHMC.∂H∂r(h, r_init) == r_init
            @test typeof(AdvancedHMC.∂H∂r(h, r_init)) == typeof(r_init)

            M⁻¹ = ones(T, D) + abs.(randn(T, D))
            h = Hamiltonian(DiagEuclideanMetric(M⁻¹), ℓπ, ∂ℓπ∂θ)
            @test -AdvancedHMC.neg_energy(h, r_init, θ_init) ≈
                  r_init' * diagm(0 => M⁻¹) * r_init / 2
            @test AdvancedHMC.∂H∂r(h, r_init) == M⁻¹ .* r_init
            @test typeof(AdvancedHMC.∂H∂r(h, r_init)) == typeof(r_init)

            m = randn(T, D, D)
            M⁻¹ = m' * m
            h = Hamiltonian(DenseEuclideanMetric(M⁻¹), ℓπ, ∂ℓπ∂θ)
            @test -AdvancedHMC.neg_energy(h, r_init, θ_init) ≈ r_init' * M⁻¹ * r_init / 2
            @test AdvancedHMC.∂H∂r(h, r_init) == M⁻¹ * r_init
            @test typeof(AdvancedHMC.∂H∂r(h, r_init)) == typeof(r_init)
        end
    end
end

@testset "Metric ComponentArrays" begin
    n_tests = 10
    for T in [Float32, Float64]
        for _ = 1:n_tests
            θ_init = ComponentArray(a = randn(T, D), b = randn(T, D))
            r_init = ComponentArray(a = randn(T, D), b = randn(T, D))

            h = Hamiltonian(UnitEuclideanMetric(T, 2*D), ℓπ, ∂ℓπ∂θ)
            @test -AdvancedHMC.neg_energy(h, r_init, θ_init) == sum(abs2, r_init) / 2
            @test AdvancedHMC.∂H∂r(h, r_init) == r_init
            @test typeof(AdvancedHMC.∂H∂r(h, r_init)) == typeof(r_init)

            M⁻¹ = ones(T, 2*D) + abs.(randn(T, 2*D))
            h = Hamiltonian(DiagEuclideanMetric(M⁻¹), ℓπ, ∂ℓπ∂θ)
            @test -AdvancedHMC.neg_energy(h, r_init, θ_init) ≈
                  r_init' * diagm(0 => M⁻¹) * r_init / 2
            @test AdvancedHMC.∂H∂r(h, r_init) == M⁻¹ .* r_init
            @test typeof(AdvancedHMC.∂H∂r(h, r_init)) == typeof(r_init)

            m = randn(T, 2*D, 2*D)
            M⁻¹ = m' * m
            h = Hamiltonian(DenseEuclideanMetric(M⁻¹), ℓπ, ∂ℓπ∂θ)
            @test -AdvancedHMC.neg_energy(h, r_init, θ_init) ≈ r_init' * M⁻¹ * r_init / 2
            @test all(AdvancedHMC.∂H∂r(h, r_init) .== M⁻¹ * r_init)
            @test typeof(AdvancedHMC.∂H∂r(h, r_init)) == typeof(r_init)
        end
    end
end
