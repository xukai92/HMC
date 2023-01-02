using ReTest, Random, AdvancedHMC, ForwardDiff, AbstractMCMC, MCMCChains
using Statistics: mean
include("common.jl")

@testset "MCMCChains w/ gdemo" begin
    rng = MersenneTwister(0)

    n_samples = 5_000
    n_adapts = 5_000

    θ_init = randn(rng, 2)

    model = AdvancedHMC.LogDensityModel(LogDensityProblemsAD.ADgradient(Val(:ForwardDiff), ℓπ_gdemo))
    init_eps = Leapfrog(1e-3)
    κ = NUTS(init_eps)
    metric = DiagEuclideanMetric(2)
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, κ.τ.integrator))

    samples = AbstractMCMC.sample(
        rng, model, κ, metric, adaptor, n_adapts + n_samples;
        nadapts = n_adapts,
        init_params = θ_init,
        chain_type = Chains,
        progress=false,
        verbose=false
    );

    # Transform back to original space.
    # NOTE: We're not correcting for the `logabsdetjac` here since, but
    # we're only interested in the mean it doesn't matter.
    for i in 1:size(samples, 1)
        samples.value.data[i,1:end-1,1] .= invlink_gdemo(samples.value.data[i,1:end-1,1])
    end

    m_est = mean(samples[n_adapts + 1:end]) 

    @test m_est[:,2] ≈ [49 / 24, 7 / 6] atol=RNDATOL
end
