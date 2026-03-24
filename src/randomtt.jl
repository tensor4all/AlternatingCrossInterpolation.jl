function randomtt(
    ::Type{ValueType},
    sitedimensions::AbstractVector{<:AbstractVector{<:Integer}},
    bonddimensions::AbstractVector{<:Integer};
    clampbonddimensions::Bool = false
) where {ValueType}
    if length(bonddimensions) == length(sitedimensions) - 1
        bonddimensions = [1; bonddimensions; 1]
    end
    @assert length(bonddimensions) == length(sitedimensions) + 1

    if clampbonddimensions
        bonddimensions = min.(bonddimensions, prodbonddimensions(sitedimensions))
    end

    tensors = [
        randn(ValueType, bonddimensions[site], sitedimensions[site]..., bonddimensions[nextsite])
        for (site, nextsite) in Sweep(1:length(sitedimensions)+1, forward=true)
    ]
    return TCI.TensorTrain(tensors)
end

function randomtt(
    ::Type{ValueType},
    sitedimensions::AbstractVector{<:AbstractVector{<:Integer}},
    χ::Int
) where {ValueType}
    bonddimensions = fill(χ, length(sitedimensions) + 1)
    return randomtt(ValueType, sitedimensions, bonddimensions; clampbonddimensions=true)
end

function randomtt(::Type{ValueType}, N::Int, d::Int, χ::Int; nlegs=1) where {ValueType}
    return randomtt(ValueType, fill(fill(d, nlegs), N), χ)
end
