function _check_scalar_sitedims(sitedims)
    if any(length.(sitedims) .!= 1)
        throw(ArgumentError("Global pivot enrichment currently supports only scalar site dimensions, i.e. TensorTrain{T,3}."))
    end
    return nothing
end

function delta_tensortrain(
    ::Type{ValueType},
    sitedims::AbstractVector{<:AbstractVector{<:Integer}},
    pivot::AbstractVector{<:Integer},
    amplitude::ValueType,
)::TensorTrain{ValueType,3} where {ValueType}
    length(pivot) == length(sitedims) || throw(DimensionMismatch("pivot length must match the number of TT sites"))
    _check_scalar_sitedims(sitedims)

    tensors = Array{ValueType,3}[]
    for site in eachindex(sitedims)
        d = only(sitedims[site])
        1 <= pivot[site] <= d || throw(BoundsError(1:d, pivot[site]))
        A = zeros(ValueType, 1, d, 1)
        A[1, pivot[site], 1] = site == lastindex(sitedims) ? amplitude : one(ValueType)
        push!(tensors, A)
    end
    return TensorTrain(tensors)
end

function elementwise_target(op::Function, inputs::Vector{<:TensorTrain})
    return q -> op((input(q) for input in inputs)...)
end

function globalpivotinput(
    solution::TensorTrain{ValueType,3},
    maxsamplevalue::Float64,
) where {ValueType}
    sitedims = TCI.sitedims(solution)
    _check_scalar_sitedims(sitedims)
    localdims = only.(sitedims)
    emptyIset = [TCI.MultiIndex[] for _ in 1:length(solution)]
    emptyJset = [TCI.MultiIndex[] for _ in 1:length(solution)]
    return TCI.GlobalPivotSearchInput{ValueType}(
        localdims,
        solution,
        maxsamplevalue,
        emptyIset,
        emptyJset,
    )
end

function findglobalpivots(
    solution::TensorTrain{ValueType,3},
    f,
    abstol::Float64;
    globalpivotfinder::Union{Nothing,TCI.AbstractGlobalPivotFinder}=nothing,
    nsearchglobalpivot::Int=5,
    maxnglobalpivot::Int=5,
    tolmarginglobalsearch::Float64=10.0,
    maxsamplevalue::Float64=0.0,
    verbosity::Int=0,
) where {ValueType}
    finder = isnothing(globalpivotfinder) ? TCI.DefaultGlobalPivotFinder(
        nsearch=nsearchglobalpivot,
        maxnglobalpivot=maxnglobalpivot,
        tolmarginglobalsearch=tolmarginglobalsearch,
    ) : globalpivotfinder

    input = globalpivotinput(solution, maxsamplevalue)
    return finder(input, f, abstol; verbosity=verbosity)
end

function spikeenrich(
    solution::TensorTrain{ValueType,3},
    f,
    pivots::AbstractVector{<:AbstractVector{<:Integer}},
) where {ValueType}
    enriched = solution
    nadded = 0
    maxsamplevalue = 0.0
    sitedims = TCI.sitedims(solution)

    for pivot in pivots
        exact = f(pivot)
        maxsamplevalue = max(maxsamplevalue, Float64(abs(exact)))
        residual = exact - enriched(pivot)
        iszero(residual) && continue
        spike = delta_tensortrain(ValueType, sitedims, pivot, residual)
        enriched = TCI.add(enriched, spike; tolerance=0.0, maxbonddim=typemax(Int))
        nadded += 1
    end

    return enriched, nadded, maxsamplevalue
end
