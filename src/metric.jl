"""
$(TYPEDEF)

Abstract type for preconditioning metrics. 
"""
abstract type AbstractMetric end

_string_M⁻¹(mat::AbstractMatrix, n_chars::Int = 32) = _string_M⁻¹(diag(mat), n_chars)
function _string_M⁻¹(vec::AbstractVector, n_chars::Int = 32)
    s_vec = string(vec)
    l = length(s_vec)
    s_dots = " ...]"
    n_diag_chars = n_chars - length(s_dots)
    return s_vec[1:min(n_diag_chars, end)] * (l > n_diag_chars ? s_dots : "")
end

struct UnitEuclideanMetric{T,A<:Union{Tuple{Int},Tuple{Int,Int}}} <: AbstractMetric
    M⁻¹::UniformScaling{T}
    size::A
end

UnitEuclideanMetric(::Type{T}, sz) where {T} =
    UnitEuclideanMetric(UniformScaling{T}(one(T)), sz)
UnitEuclideanMetric(sz) = UnitEuclideanMetric(Float64, sz)
UnitEuclideanMetric(::Type{T}, dim::Int) where {T} =
    UnitEuclideanMetric(UniformScaling{T}(one(T)), (dim,))
UnitEuclideanMetric(dim::Int) = UnitEuclideanMetric(Float64, (dim,))

renew(ue::UnitEuclideanMetric, M⁻¹) = UnitEuclideanMetric(M⁻¹, ue.size)

Base.eltype(::UnitEuclideanMetric{T}) where {T} = T
Base.size(e::UnitEuclideanMetric) = e.size
Base.size(e::UnitEuclideanMetric, dim::Int) = e.size[dim]
Base.show(io::IO, uem::UnitEuclideanMetric) =
    print(io, "UnitEuclideanMetric($(_string_M⁻¹(ones(uem.size))))")

struct DiagEuclideanMetric{T,A<:AbstractVecOrMat{T}} <: AbstractMetric
    # Diagnal of the inverse of the mass matrix
    M⁻¹::A
    # Sqare root of the inverse of the mass matrix
    sqrtM⁻¹::A
    # Pre-allocation for intermediate variables
    _temp::A
end

function DiagEuclideanMetric(M⁻¹::AbstractVecOrMat{T}) where {T<:AbstractFloat}
    return DiagEuclideanMetric(M⁻¹, sqrt.(M⁻¹), similar(M⁻¹))
end
DiagEuclideanMetric(::Type{T}, sz) where {T} = DiagEuclideanMetric(ones(T, sz...))
DiagEuclideanMetric(sz) = DiagEuclideanMetric(Float64, sz)
DiagEuclideanMetric(::Type{T}, dim::Int) where {T} = DiagEuclideanMetric(ones(T, dim))
DiagEuclideanMetric(dim::Int) = DiagEuclideanMetric(Float64, dim)

renew(ue::DiagEuclideanMetric, M⁻¹) = DiagEuclideanMetric(M⁻¹)

Base.eltype(::DiagEuclideanMetric{T}) where {T} = T
Base.size(e::DiagEuclideanMetric, dim...) = size(e.M⁻¹, dim...)
Base.show(io::IO, dem::DiagEuclideanMetric) =
    print(io, "DiagEuclideanMetric($(_string_M⁻¹(dem.M⁻¹)))")

struct DenseEuclideanMetric{
    T,
    AV<:AbstractVecOrMat{T},
    AM<:Union{AbstractMatrix{T},AbstractArray{T,3}},
    TcholM⁻¹<:UpperTriangular{T},
} <: AbstractMetric
    # Inverse of the mass matrix
    M⁻¹::AM
    # U of the Cholesky decomposition of the mass matrix
    cholM⁻¹::TcholM⁻¹
    # Pre-allocation for intermediate variables
    _temp::AV
end

# TODO: make dense mass matrix support matrix-mode parallel
function DenseEuclideanMetric(
    M⁻¹::Union{AbstractMatrix{T},AbstractArray{T,3}},
) where {T<:AbstractFloat}
    _temp = Vector{T}(undef, Base.front(size(M⁻¹)))
    return DenseEuclideanMetric(M⁻¹, cholesky(Symmetric(M⁻¹)).U, _temp)
end
DenseEuclideanMetric(::Type{T}, D::Int) where {T} = DenseEuclideanMetric(Matrix{T}(I, D, D))
DenseEuclideanMetric(D::Int) = DenseEuclideanMetric(Float64, D)
DenseEuclideanMetric(::Type{T}, sz::Tuple{Int}) where {T} =
    DenseEuclideanMetric(Matrix{T}(I, first(sz), first(sz)))
DenseEuclideanMetric(sz::Tuple{Int}) = DenseEuclideanMetric(Float64, sz)

renew(ue::DenseEuclideanMetric, M⁻¹) = DenseEuclideanMetric(M⁻¹)

Base.eltype(::DenseEuclideanMetric{T}) where {T} = T
Base.size(e::DenseEuclideanMetric, dim...) = size(e._temp, dim...)
Base.show(io::IO, dem::DenseEuclideanMetric) =
    print(io, "DenseEuclideanMetric(diag=$(_string_M⁻¹(dem.M⁻¹)))")

# `rand` functions for `metric` types.

function rand_momentum(
    rng::Union{AbstractRNG,AbstractVector{<:AbstractRNG}},
    metric::UnitEuclideanMetric{T},
    kinetic::GaussianKinetic,
    ::AbstractVecOrMat,
) where {T}
    r = _randn(rng, T, size(metric)...)
    return r
end

function rand_momentum(
    rng::Union{AbstractRNG,AbstractVector{<:AbstractRNG}},
    metric::DiagEuclideanMetric{T},
    kinetic::GaussianKinetic,
    ::AbstractVecOrMat,
) where {T}
    r = _randn(rng, T, size(metric)...)
    r ./= metric.sqrtM⁻¹
    return r
end

function rand_momentum(
    rng::Union{AbstractRNG,AbstractVector{<:AbstractRNG}},
    metric::DenseEuclideanMetric{T},
    kinetic::GaussianKinetic,
    ::AbstractVecOrMat,
) where {T}
    r = _randn(rng, T, size(metric)...)
    ldiv!(metric.cholM⁻¹, r)
    return r
end
