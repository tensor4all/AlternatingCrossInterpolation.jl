function contractframe(
    currentframe,
    sitetensor;
    forward::Bool,
)
    newframe = contractonchain(
        currentframe, [ndims(currentframe,)], sitetensor, [1]; swap=!forward
    )
    newshape = if forward
        (:, last(size(newframe)))
    else
        (first(size(newframe)), :)
    end

    return reshape(newframe, newshape)
end

function updateframe(
    currentframe,
    sitetensor;
    forward::Bool,
    indices::AbstractVector{<:Integer}=Int[],
    truncationparameters::TruncationParameters=TruncationParameters()
)
    newframe = contractframe(currentframe, sitetensor; forward)

    if isempty(indices)
        return truncate(newframe; truncationparameters, leftorthogonal=forward)
    else
        return truncate(newframe, indices; leftorthogonal=forward)
    end
end

function frame(
    tt::TensorTrain{ValueType, 3};
    indices::AbstractVector{<:AbstractVector{<:Integer}}=fill(Int[], length(tt)),
    truncationparameters::TruncationParameters=TruncationParameters(),
    leftorthogonal::Bool=true
) where {ValueType}
    frame = Origin(0)([ones(ValueType, 1, 1)])
    for (ell, nextell) in Sweep(1:length(tt), forward=leftorthogonal)
        bondindex = min(ell, nextell)
        E, indices[bondindex] = updateframe(
            last(frame), tt[ell];
            forward=leftorthogonal, indices=indices[bondindex],
            truncationparameters
        )
        push!(frame, E)
    end

    push!(frame, ones(ValueType, 1, 1))

    if !leftorthogonal
        reverse!(frame)
    end
    
    return frame, indices
end

function attachframes(
    leftenv::Matrix{ValueType},
    T::Array{ValueType},
    rightenv::Matrix{ValueType}
) where ValueType
    @debug "attach matrix sizes:" size(leftenv) size(T) size(rightenv)
    LT = contract(leftenv, [ndims(leftenv)], T, [1])
    LTR = contract(LT, [ndims(LT)], rightenv, [1])
    return LTR
end

function removeframes(
    leftenv::Matrix{ValueType},
    T::Array{ValueType},
    rightenv::Matrix{ValueType}
) where ValueType
    @assert issquare(leftenv)
    @assert issquare(rightenv)
    @debug "remove matrix sizes:" size(leftenv) size(T) size(rightenv)
    TR = reshape(T, :, last(size(T))) / rightenv
    LTR = leftenv \ reshape(TR, first(size(T)), :)
    return reshape(LTR, size(T))
end
