@doc raw"""
    TruncationParameters(; maxbonddimension::Int=typemax(Int), tolerance::Float64=1e-14)

A struct to hold parameters for truncation of matrices in the maxvol algorithm.
- `maxbonddimension`: The maximum allowed bond dimension (rank) after truncation. Default is `typemax(Int)`, which means no limit.
- `tolerance`: The absolute tolerance for truncation. Default is `1e-14`.
"""
@kwdef struct TruncationParameters
    maxbonddimension::Int=typemax(Int)
    tolerance::Float64=1e-14
    scaletolerance::Bool=false
end

function maxvol(
    A::Matrix;
    truncationparameters::TruncationParameters=TruncationParameters(),
    leftorthogonal::Bool=true
)
    LU = TCI.rrlu(
        A;
        leftorthogonal=true,
        abstol=truncationparameters.tolerance,
        maxrank=truncationparameters.maxbonddimension
    )
    return leftorthogonal ? TCI.rowindices(LU) : TCI.colindices(LU)
end

function truncate(
    A::Matrix,
    indices::AbstractVector{<:Integer};
    leftorthogonal::Bool=true
)
    if leftorthogonal
        return A[indices, :], indices
    else
        return A[:, indices], indices
    end
end

function truncate(
    A::Matrix;
    truncationparameters::TruncationParameters=TruncationParameters(),
    leftorthogonal::Bool=true
)
    indices = maxvol(A; truncationparameters, leftorthogonal)
    return truncate(A, indices; leftorthogonal)
end
