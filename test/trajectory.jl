using ReTest, AdvancedHMC, Random
using Statistics: mean
using LinearAlgebra: dot

function makeplot(plt, traj_θ, ts_list...)
    function plotturn!(traj_θ, ts)
        s = 9.0
        idcs_nodiv = ts .== false
        idcs_div = ts .== true
        idcs_nodiv[1] = idcs_div[1] = false # avoid plotting the first point
        plt.scatter(
            traj_θ[1, idcs_nodiv],
            traj_θ[2, idcs_nodiv],
            s = s,
            c = "black",
            label = "¬div",
        )
        plt.scatter(
            traj_θ[1, idcs_div],
            traj_θ[2, idcs_div],
            s = s,
            c = "red",
            label = "div",
        )
        plt.scatter(traj_θ[1, 1], traj_θ[2, 1], s = s, c = "yellow", label = "init")
    end

    fig = plt.figure(figsize = (16, 3))

    for (i, ts, title) in zip(
        1:length(ts_list),
        ts_list,
        [
            "Hand original (v = 1)",
            "AHMC original (v = 1)",
            "Hand generalised (v = 1)",
            "AHMC generalised (v = 1)",
        ],
    )
        plt.subplot(1, 4, i)
        plotturn!(traj_θ, ts)
        plt.gca().set_title(title)
        plt.legend()
    end

    return fig
end

function gettraj(rng, h, ϵ = 0.1, n_steps = 50)
    lf = Leapfrog(ϵ)

    q_init = randn(rng, D)
    p_init = AdvancedHMC.rand_momentum(rng, h.metric, h.kinetic, q_init)
    z = AdvancedHMC.phasepoint(h, q_init, p_init)

    traj_z = Vector(undef, n_steps)
    traj_z[1] = z
    for i = 2:n_steps
        traj_z[i] = AdvancedHMC.step(lf, h, traj_z[i-1])
    end

    return traj_z
end

function hand_isturn(z0, z1, rho, v = 1)
    θ0minusθ1 = z0.θ - z1.θ
    s = (dot(-θ0minusθ1, -z0.r) >= 0) || (dot(θ0minusθ1, z1.r) >= 0)
    return s
end

ahmc_isturn(h, z0, z1, rho, v = 1) =
    AdvancedHMC.isterminated(
        ClassicNoUTurn(),
        h,
        AdvancedHMC.BinaryTree(z0, z1, AdvancedHMC.TurnStatistic(), 0, 0, 0.0),
    ).dynamic

function hand_isturn_generalised(z0, z1, rho, v = 1)
    s = (dot(rho, -z0.r) >= 0) || (dot(-rho, z1.r) >= 0)
    return s
end

ahmc_isturn_generalised(h, z0, z1, rho, v = 1) =
    AdvancedHMC.isterminated(
        GeneralisedNoUTurn(),
        h,
        AdvancedHMC.BinaryTree(z0, z1, AdvancedHMC.TurnStatistic(rho), 0, 0, 0.0),
    ).dynamic

function ahmc_isturn_strictgeneralised(h, z0, z1, rho, v = 1)
    t = AdvancedHMC.isterminated(
        StrictGeneralisedNoUTurn(),
        h,
        AdvancedHMC.BinaryTree(z0, z1, AdvancedHMC.TurnStatistic(rho), 0, 0, 0.0),
        AdvancedHMC.BinaryTree(z0, z0, AdvancedHMC.TurnStatistic(rho - z1.r), 0, 0, 0.0),
        AdvancedHMC.BinaryTree(z1, z1, AdvancedHMC.TurnStatistic(rho - z0.r), 0, 0, 0.0),
    )
    return t.dynamic
end

"""
Check whether the subtree checks adequately detect U-turns.
"""
function check_subtree_u_turns(h, z0, z1, rho)
    t = AdvancedHMC.BinaryTree(z0, z1, AdvancedHMC.TurnStatistic(rho), 0, 0, 0.0)
    # The left and right subtree are created in such a way that the
    # check_left_subtree and check_right_subtree checks should be equivalent
    # to the general no U-turn check.
    tleft = AdvancedHMC.BinaryTree(z0, z0, AdvancedHMC.TurnStatistic(rho - z1.r), 0, 0, 0.0)
    tright =
        AdvancedHMC.BinaryTree(z1, z1, AdvancedHMC.TurnStatistic(rho - z0.r), 0, 0, 0.0)

    s1 = AdvancedHMC.isterminated(GeneralisedNoUTurn(), h, t)
    s2 = AdvancedHMC.check_left_subtree(h, t, tleft, tright)
    s3 = AdvancedHMC.check_right_subtree(h, t, tleft, tright)
    @test s1 == s2 == s3
end


@testset "Trajectory" begin
    ϵ = 0.01
    lf = Leapfrog(ϵ)

    θ_init = randn(D)
    h = Hamiltonian(UnitEuclideanMetric(D), ℓπ, ∂ℓπ∂θ)
    τ = Trajectory{MultinomialTS}(
        Leapfrog(find_good_stepsize(h, θ_init)),
        GeneralisedNoUTurn(),
    )
    r_init = AdvancedHMC.rand_momentum(Random.default_rng(), h.metric, h.kinetic, θ_init)

    @testset "Passing RNG" begin
        τ_with_jittered_lf = Trajectory{MultinomialTS}(
            JitteredLeapfrog(find_good_stepsize(h, θ_init), 1.0),
            GeneralisedNoUTurn(),
        )
        for τ_test in [τ, τ_with_jittered_lf], seed in [1234, 5678, 90]
            rng = MersenneTwister(seed)
            z = AdvancedHMC.phasepoint(h, θ_init, r_init)
            z1′ = AdvancedHMC.transition(rng, τ_test, h, z).z

            rng = MersenneTwister(seed)
            z = AdvancedHMC.phasepoint(h, θ_init, r_init)
            z2′ = AdvancedHMC.transition(rng, τ_test, h, z).z

            @test z1′.θ == z2′.θ
            @test z1′.r == z2′.r
        end
    end

    @testset "TreeSampler" begin
        n_samples = 10_000
        z1 = AdvancedHMC.phasepoint(h, zeros(D), r_init)
        z2 = AdvancedHMC.phasepoint(h, ones(D), r_init)

        rng = MersenneTwister(1234)

        ℓu = rand()
        n1 = 2
        s1 = AdvancedHMC.SliceTS(z1, ℓu, n1)
        n2 = 1
        s2 = AdvancedHMC.SliceTS(z2, ℓu, n2)
        s3 = AdvancedHMC.combine(rng, s1, s2)
        @test s3.ℓu == ℓu
        @test s3.n == n1 + n2


        s3_θ = Vector(undef, n_samples)
        for i = 1:n_samples
            s3_θ[i] = AdvancedHMC.combine(rng, s1, s2).zcand.θ
        end
        @test mean(s3_θ) ≈ ones(D) * n2 / (n1 + n2) rtol = 0.01

        w1 = 100
        s1 = AdvancedHMC.MultinomialTS(z1, log(w1))
        w2 = 150
        s2 = AdvancedHMC.MultinomialTS(z2, log(w2))
        s3 = AdvancedHMC.combine(rng, s1, s2)
        @test s3.ℓw ≈ log(w1 + w2)

        s3_θ = Vector(undef, n_samples)
        for i = 1:n_samples
            s3_θ[i] = AdvancedHMC.combine(rng, s1, s2).zcand.θ
        end
        @test mean(s3_θ) ≈ ones(D) * w2 / (w1 + w2) rtol = 0.01
    end

    @testset "TerminationCriterion" begin
        tc = AdvancedHMC.ClassicNoUTurn()
        z1 = AdvancedHMC.phasepoint(h, θ_init, randn(D))
        z2 = AdvancedHMC.phasepoint(h, θ_init, randn(D))
        ts1 = AdvancedHMC.TurnStatistic(tc, z1)
        ts2 = AdvancedHMC.TurnStatistic(tc, z2)
        ts3 = AdvancedHMC.combine(ts1, ts2)
        @test ts1 == ts2 == ts3

        tc = AdvancedHMC.GeneralisedNoUTurn()
        r1 = randn(D)
        z1 = AdvancedHMC.phasepoint(h, θ_init, r1)
        r2 = randn(D)
        z2 = AdvancedHMC.phasepoint(h, θ_init, r2)
        ts1 = AdvancedHMC.TurnStatistic(tc, z1)
        ts2 = AdvancedHMC.TurnStatistic(tc, z2)
        ts3 = AdvancedHMC.combine(ts1, ts2)
        @test ts3.rho == r1 + r2
    end

    @testset "Termination" begin
        t00 = AdvancedHMC.Termination(false, false)
        t01 = AdvancedHMC.Termination(false, true)
        t10 = AdvancedHMC.Termination(true, false)
        t11 = AdvancedHMC.Termination(true, true)

        @test AdvancedHMC.isterminated(t00) == false
        @test AdvancedHMC.isterminated(t01) == true
        @test AdvancedHMC.isterminated(t10) == true
        @test AdvancedHMC.isterminated(t11) == true

        @test AdvancedHMC.isterminated(t00 * t00) == false
        @test AdvancedHMC.isterminated(t00 * t01) == true
        @test AdvancedHMC.isterminated(t00 * t10) == true
        @test AdvancedHMC.isterminated(t00 * t11) == true

        @test AdvancedHMC.isterminated(t01 * t00) == true
        @test AdvancedHMC.isterminated(t01 * t01) == true
        @test AdvancedHMC.isterminated(t01 * t10) == true
        @test AdvancedHMC.isterminated(t01 * t11) == true

        @test AdvancedHMC.isterminated(t10 * t00) == true
        @test AdvancedHMC.isterminated(t10 * t01) == true
        @test AdvancedHMC.isterminated(t10 * t10) == true
        @test AdvancedHMC.isterminated(t10 * t11) == true

        @test AdvancedHMC.isterminated(t11 * t00) == true
        @test AdvancedHMC.isterminated(t11 * t01) == true
        @test AdvancedHMC.isterminated(t11 * t10) == true
        @test AdvancedHMC.isterminated(t11 * t11) == true
    end

    @testset "BinaryTree" begin
        z = AdvancedHMC.phasepoint(h, θ_init, randn(D))

        t1 = AdvancedHMC.BinaryTree(z, z, AdvancedHMC.TurnStatistic(), 0.1, 1, -2.0)
        t2 = AdvancedHMC.BinaryTree(z, z, AdvancedHMC.TurnStatistic(), 1.1, 2, 1.0)
        t3 = AdvancedHMC.combine(t1, t2)

        @test t3.sum_α ≈ 1.2 atol = 1e-9
        @test t3.nα == 3
        @test t3.ΔH_max == -2.0

        t4 = AdvancedHMC.BinaryTree(z, z, AdvancedHMC.TurnStatistic(), 1.1, 2, 3.0)
        t5 = AdvancedHMC.combine(t1, t4)

        @test t5.ΔH_max == 3.0
    end

    ### Test ClassicNoUTurn and GeneralisedNoUTurn
    @testset "ClassicNoUTurn" begin
        n_tests = 4
        for _ = 1:n_tests
            seed = abs(rand(Int8) + 128)
            rng = MersenneTwister(seed)
            @testset "seed = $seed" begin
                traj_z = gettraj(rng, h)
                traj_θ = hcat(map(z -> z.θ, traj_z)...)
                traj_r = hcat(map(z -> z.r, traj_z)...)
                rho = cumsum(traj_r, dims = 2)

                ts_hand_isturn_fwd =
                    hand_isturn.(
                        Ref(traj_z[1]),
                        traj_z,
                        [rho[:, i] for i = 1:length(traj_z)],
                        Ref(1),
                    )
                ts_ahmc_isturn_fwd =
                    ahmc_isturn.(
                        Ref(h),
                        Ref(traj_z[1]),
                        traj_z,
                        [rho[:, i] for i = 1:length(traj_z)],
                        Ref(1),
                    )

                ts_hand_isturn_generalised_fwd =
                    hand_isturn_generalised.(
                        Ref(traj_z[1]),
                        traj_z,
                        [rho[:, i] for i = 1:length(traj_z)],
                        Ref(1),
                    )
                ts_ahmc_isturn_generalised_fwd =
                    ahmc_isturn_generalised.(
                        Ref(h),
                        Ref(traj_z[1]),
                        traj_z,
                        [rho[:, i] for i = 1:length(traj_z)],
                        Ref(1),
                    )

                ts_ahmc_isturn_strictgeneralised_fwd =
                    ahmc_isturn_strictgeneralised.(
                        Ref(h),
                        Ref(traj_z[1]),
                        traj_z,
                        [rho[:, i] for i = 1:length(traj_z)],
                        Ref(1),
                    )

                check_subtree_u_turns.(
                    Ref(h),
                    Ref(traj_z[1]),
                    traj_z,
                    [rho[:, i] for i = 1:length(traj_z)],
                )

                @test ts_hand_isturn_fwd[2:end] ==
                      ts_ahmc_isturn_fwd[2:end] ==
                      ts_hand_isturn_generalised_fwd[2:end] ==
                      ts_ahmc_isturn_generalised_fwd[2:end] ==
                      ts_ahmc_isturn_strictgeneralised_fwd[2:end]

                if length(ARGS) > 0 && ARGS[1] == "--plot"
                    import PyPlot
                    fig = makeplot(
                        PyPlot,
                        traj_θ,
                        ts_hand_isturn_fwd,
                        ts_ahmc_isturn_fwd,
                        ts_hand_isturn_generalised_fwd,
                        ts_ahmc_isturn_generalised_fwd,
                    )
                    fig.savefig("seed=$seed.png")
                end
            end
        end
    end

end
