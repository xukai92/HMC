using ReTest, Random, AdvancedHMC, ForwardDiff

using OrdinaryDiffEq
using LinearAlgebra: dot
using Statistics: mean

@testset "Integrator" begin
    ϵ = 0.01
    lf = Leapfrog(ϵ)

    θ_init = randn(D)
    h = Hamiltonian(UnitEuclideanMetric(D), ℓπ, ∂ℓπ∂θ)
    r_init = AdvancedHMC.rand_momentum(Random.default_rng(), h.metric, h.kinetic, θ_init)

    n_steps = 10

    @testset "step" begin
        z = AdvancedHMC.phasepoint(h, copy(θ_init), copy(r_init))
        z_step_loop = z

        t_step_loop = @elapsed for i = 1:n_steps
            z_step_loop = AdvancedHMC.step(lf, h, z_step_loop)
        end

        t_step = @elapsed z_step = AdvancedHMC.step(lf, h, z, n_steps)

        @info "Performance of loop of step() v.s. step()" n_steps t_step_loop t_step t_step_loop /
                                                                                     t_step

        @test z_step_loop.θ ≈ z_step.θ atol = DETATOL
        @test z_step_loop.r ≈ z_step.r atol = DETATOL
    end

    @testset "jitter" begin
        @testset "Leapfrog" begin
            ϵ0 = 0.1
            lf = Leapfrog(ϵ0)
            @test lf.ϵ == ϵ0
            @test AdvancedHMC.nom_step_size(lf) == ϵ0
            @test AdvancedHMC.step_size(lf) == ϵ0

            lf2 = AdvancedHMC.jitter(Random.default_rng(), lf)
            @test lf2 === lf
            @test AdvancedHMC.nom_step_size(lf2) == ϵ0
            @test AdvancedHMC.step_size(lf2) == ϵ0
        end

        @testset "JitteredLeapfrog" begin
            ϵ0 = 0.1
            lf = JitteredLeapfrog(ϵ0, 0.5)
            @test lf.ϵ0 == ϵ0
            @test AdvancedHMC.nom_step_size(lf) == ϵ0
            @test lf.ϵ == ϵ0
            @test AdvancedHMC.step_size(lf) == lf.ϵ

            lf2 = AdvancedHMC.jitter(Random.default_rng(), lf)
            @test lf2.ϵ0 == ϵ0
            @test AdvancedHMC.nom_step_size(lf2) == ϵ0
            @test lf2.ϵ != ϵ0
            @test AdvancedHMC.step_size(lf2) == lf2.ϵ
        end
    end

    @testset "update_nom_step_size" begin
        @testset "Leapfrog" begin
            ϵ0 = 0.1
            lf = Leapfrog(ϵ0)
            @test AdvancedHMC.nom_step_size(lf) == ϵ0

            lf2 = AdvancedHMC.update_nom_step_size(lf, 0.5)
            @test lf2 !== lf
            @test AdvancedHMC.nom_step_size(lf2) == 0.5
            @test AdvancedHMC.step_size(lf2) == 0.5
        end

        @testset "JitteredLeapfrog" begin
            ϵ0 = 0.1
            lf = JitteredLeapfrog(ϵ0, 0.5)
            @test AdvancedHMC.nom_step_size(lf) == ϵ0

            lf2 = AdvancedHMC.update_nom_step_size(lf, 0.2)
            @test lf2 !== lf
            @test AdvancedHMC.nom_step_size(lf2) == 0.2
            # check that we've only updated nominal step size
            @test AdvancedHMC.step_size(lf2) == ϵ0
        end
    end

    @testset "temper" begin
        αsqrt = 2.0
        lf = TemperedLeapfrog(ϵ, αsqrt^2)
        r = ones(5)
        r1 = AdvancedHMC.temper(lf, r, (i = 1, is_half = true), 3)
        r2 = AdvancedHMC.temper(lf, r, (i = 1, is_half = false), 3)
        r3 = AdvancedHMC.temper(lf, r, (i = 2, is_half = true), 3)
        r4 = AdvancedHMC.temper(lf, r, (i = 2, is_half = false), 3)
        r5 = AdvancedHMC.temper(lf, r, (i = 3, is_half = true), 3)
        r6 = AdvancedHMC.temper(lf, r, (i = 3, is_half = false), 3)
        @test r1 == αsqrt * ones(5)
        @test r2 == αsqrt * ones(5)
        @test r3 == αsqrt * ones(5)
        @test r4 == inv(αsqrt) * ones(5)
        @test r5 == inv(αsqrt) * ones(5)
        @test r6 == inv(αsqrt) * ones(5)
        @test_throws BoundsError AdvancedHMC.temper(lf, r, (i = 4, is_half = false), 3)

    end

    @testset "Analytical solution to Eq (2.11) of Neal (2011)" begin
        struct NegU
            dim::Int
        end

        LogDensityProblems.logdensity(::NegU, x) = -dot(x, x) / 2
        LogDensityProblems.dimension(d::NegU) = d.dim
        LogDensityProblems.capabilities(::Type{NegU}) =
            LogDensityProblems.LogDensityOrder{0}()

        negU = NegU(1)

        ϵ = 0.01
        for lf in [Leapfrog(ϵ), DiffEqIntegrator(ϵ, VerletLeapfrog())]
            q_init = randn(1)
            h = Hamiltonian(UnitEuclideanMetric(1), negU, ForwardDiff)
            p_init =
                AdvancedHMC.rand_momentum(Random.default_rng(), h.metric, h.kinetic, q_init)

            q, p = copy(q_init), copy(p_init)
            z = AdvancedHMC.phasepoint(h, q, p)

            n_steps = 10_000
            qs = zeros(n_steps)
            ps = zeros(n_steps)
            Hs = zeros(n_steps)
            for i = 1:n_steps
                z = AdvancedHMC.step(lf, h, z)
                qs[i] = z.θ[1]
                ps[i] = z.r[1]
                Hs[i] = -AdvancedHMC.neg_energy(z)
            end

            # Throw first 1_000 steps
            qs = qs[1_000:end]
            ps = ps[1_000:end]
            Hs = Hs[1_000:end]

            # Check if all points located at a cirle centered at the origin
            rs = sqrt.(qs .^ 2 + ps .^ 2)
            @test all(x -> abs(x - mean(rs)) < 2e-3, rs)

            # Check if the Hamiltonian energy is stable
            @test all(x -> abs(x - mean(Hs)) < 2e-3, Hs)
        end
    end

end
