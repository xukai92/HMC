### Define the target distribution
using Distributions: logpdf, MvNormal

D = 10
target = MvNormal(zeros(D), ones(D))
ℓπ(θ) = logpdf(target, θ)

### Load an AD library - AdvancedHMC will use it for gradient
using ForwardDiff   # or using Zygote

### Build up a HMC sampler to draw samples
using ForwardDiff, AdvancedHMC  

# Sampling parameter settings
n_samples, n_adapts = 12_000, 2_000

# Draw a random starting points
θ_init = rand(D)

# Define metric space, Hamiltonian, sampling method and adaptor
metric = DiagEuclideanMetric(D)
h = Hamiltonian(metric, ℓπ) # do Hamiltonian(metric, ℓπ, ∂ℓπ∂θ) if you have a hand-coded gradient function ∂ℓπ∂θ
int = Leapfrog(find_good_eps(h, θ_init))
prop = NUTS{MultinomialTS,GeneralisedNoUTurn}(int)
adaptor = StanHMCAdaptor(
    Preconditioner(metric), 
    NesterovDualAveraging(0.8, int)
)

# Draw samples via simulating Hamiltonian dynamics
# - `samples` will store the samples
# - `stats` will store statistics for each sample
samples, stats = sample(h, prop, θ_init, n_samples, adaptor, n_adapts; progress=true)