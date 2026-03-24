function issquare(A::Matrix)
    return allequal(size(A))
end

function prodbonddimensions(sitedimensions::AbstractVector{<:AbstractVector{<:Integer}})
    proddimsleft = [1; cumprod(prod.(sitedimensions))]
    proddimsright = [reverse(cumprod(reverse(prod.(sitedimensions)))); 1]
    return min.(proddimsleft, proddimsright)
end
