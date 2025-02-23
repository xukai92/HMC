using ReTest, Random, AdvancedHMC

@testset "Metric" begin
    @testset "Sample momentum variables from metric via vector of RNGs" begin
        D = 10
        n_chains = 5
        θ = randn(D, n_chains)
        rng = [MersenneTwister(1) for _ = 1:n_chains]
        for metric in [
            UnitEuclideanMetric((D, n_chains)),
            DiagEuclideanMetric((D, n_chains)),
            # DenseEuclideanMetric((D, n_chains)) # not supported ATM
        ]
            r = AdvancedHMC.rand_momentum(rng, metric, GaussianKinetic(), θ)
            all_same = true
            for i = 2:n_chains
                all_same = all_same && r[:, i] == r[:, 1]
            end
            @test all_same
        end
    end

    @testset "Resize metric" begin
        D = 10
        rng = MersenneTwister(1)
        θ = randn(rng, D)
        ℓπ(θ) = 1
        for metric in
            [UnitEuclideanMetric(1), DiagEuclideanMetric(1), DenseEuclideanMetric(1)]
            h = Hamiltonian(metric, ℓπ, ℓπ)
            h = AdvancedHMC.resize(h, θ)
            @test size(h.metric) == size(θ)
        end
    end
end
