mutable struct ElementwiseProblem{ValueType,N}
    inputs::Vector{TensorTrain{ValueType,N}}
    solution::TensorTrain{ValueType,N}

    leftframes::OffsetMatrix{Matrix{ValueType},Matrix{Matrix{ValueType}}}
    rightframes::OffsetMatrix{Matrix{ValueType},Matrix{Matrix{ValueType}}}

    pivoterrors::Vector{Float64}
    maxsamplevalue::Float64
    inputcaches::Vector{TCI.TTCache{ValueType}}

    function ElementwiseProblem{ValueType,N}(
        inputs::Vector{TensorTrain{ValueType,N}},
        initial_guess::TensorTrain{ValueType,N}
    ) where {ValueType,N}
        Ninputs = length(inputs)
        Nsites = length(first(inputs))
        flatdims = [[prod(d)] for d in TCI.sitedims(first(inputs))]

        problem = new{ValueType,N}(
            inputs,
            deepcopy(initial_guess),
            Origin(1, 0)(Matrix{Matrix{ValueType}}(undef, Ninputs, Nsites + 1)),
            Origin(1, 1)(Matrix{Matrix{ValueType}}(undef, Ninputs, Nsites + 1)),
            zeros(Nsites),
            0.0,
            [TCI.TTCache(input, flatdims) for input in inputs]
        )

        problem.leftframes[:, 0] .= Ref(ones(ValueType, 1, 1))
        problem.rightframes[:, Nsites+1] .= Ref(ones(ValueType, 1, 1))
        return problem
    end
end

function ElementwiseProblem(
    inputs::Vector{TensorTrain{ValueType,N}},
    initial_guess::TensorTrain{ValueType,N}
) where {ValueType,N}
    problem = ElementwiseProblem{ValueType,N}(inputs, initial_guess)
    initializeproblem!(problem)
    return problem
end

function Base.length(problem::ElementwiseProblem{ValueType,N}) where {ValueType,N}
    return length(problem.solution)
end

function eachsiteindex(problem::ElementwiseProblem{ValueType,N}) where {ValueType,N}
    return 1:length(problem)
end

function eachbondindex(problem::ElementwiseProblem{ValueType,N}) where {ValueType,N}
    return eachsiteindex(problem)[begin:end-1]
end

function eachinputindex(problem::ElementwiseProblem{ValueType,N}) where {ValueType,N}
    return eachindex(problem.inputs)
end

Base.eachindex(problem::ElementwiseProblem{ValueType,N}) where {ValueType,N} = eachsiteindex(problem)

function updateleftframe(
    leftframe::AbstractMatrix{ValueType},
    sitetensor::AbstractArray{ValueType,3},
    Iset::AbstractVector{<:Integer}
) where {ValueType}
    newframe = reshape(contract(leftframe, [2], sitetensor, [1]), :, size(sitetensor)[end])
    return newframe[Iset, :]
end

function updateleftframe!(
    problem::ElementwiseProblem{ValueType,N},
    inputindex::Integer, siteindex::Integer,
    Iset::AbstractVector{<:Integer}
) where {ValueType,N}
    problem.leftframes[inputindex, siteindex] = updateleftframe(
        problem.leftframes[inputindex, siteindex-1],
        problem.inputs[inputindex][siteindex],
        Iset
    )
    nothing
end

function updateleftframes!(
    problem::ElementwiseProblem{ValueType,N},
    siteindex::Integer,
    Iset::AbstractVector{<:Integer}
) where {ValueType,N}
    for inputindex in eachinputindex(problem)
        updateleftframe!(problem, inputindex, siteindex, Iset)
    end
    nothing
end

function updaterightframe(
    rightframe::AbstractMatrix{ValueType},
    sitetensor::AbstractArray{ValueType,3},
    Jset::AbstractVector{<:Integer}
) where {ValueType}
    newframe = reshape(contract(sitetensor, ndims(sitetensor), rightframe, [1]), size(sitetensor, 1), :)
    return newframe[:, Jset]
end

function updaterightframe!(
    problem::ElementwiseProblem{ValueType,N},
    inputindex::Integer, siteindex::Integer,
    Jset::AbstractVector{<:Integer}
) where {ValueType,N}
    problem.rightframes[inputindex, siteindex] = updaterightframe(
        problem.rightframes[inputindex, siteindex+1],
        problem.inputs[inputindex][siteindex],
        Jset
    )
    nothing
end

function updaterightframes!(
    problem::ElementwiseProblem{ValueType,N},
    siteindex::Integer,
    Jset::AbstractVector{<:Integer}
) where {ValueType,N}
    for inputindex in eachinputindex(problem)
        updaterightframe!(problem, inputindex, siteindex, Jset)
    end
    nothing
end

function pitensor(
    left::AbstractMatrix{ValueType},
    Tleft::AbstractArray{ValueType,3},
    Tright::AbstractArray{ValueType,3},
    right::AbstractMatrix{ValueType}
) where {ValueType}
    @debug "Pi tensor" size(left) size(Tleft) size(Tright) size(right)
    size(left, 2) ≠ size(Tleft, 1) || size(Tleft, 3) ≠ size(Tright, 1) || size(Tright, 3) ≠ size(right, 1)
    L = contract(left, [2], Tleft, [1])
    R = contract(Tright, [3], right, [1])
    return contract(L, [3], R, [1])
end

function pitensor(
    problem::ElementwiseProblem{ValueType,N},
    inputindex::Integer, bondindex::Integer
) where {ValueType,N}
    return pitensor(
        problem.leftframes[inputindex, bondindex-1],
        problem.inputs[inputindex][bondindex],
        problem.inputs[inputindex][bondindex+1],
        problem.rightframes[inputindex, bondindex+2]
    )
end

function updatemaxsample!(problem::ElementwiseProblem{ValueType,N}, Π::AbstractArray{ValueType}) where {ValueType,N}
    problem.maxsamplevalue = max(problem.maxsamplevalue, maximum(abs, Π))
end

function localupdate!(
    op::Function,
    problem::ElementwiseProblem{ValueType,N},
    bondindex::Integer;
    leftorthogonal::Bool,
    truncationparameters::TruncationParameters
) where {ValueType,N}
    @debug "Local update" bondindex leftorthogonal
    Πs = [pitensor(problem, k, bondindex) for k in eachinputindex(problem)]
    Π = op.(Πs...)

    if truncationparameters.scaletolerance
        updatemaxsample!(problem, Π)
    end
    abstol = effectivetolerance(problem, truncationparameters)

    luci = TCI.MatrixLUCI(
        reshape(Π, prod(size(Π)[1:2]), prod(size(Π)[3:4]));
        leftorthogonal=leftorthogonal,
        maxrank=truncationparameters.maxbonddimension,
        abstol=abstol
    )

    problem.solution.sitetensors[bondindex] = reshape(TCI.left(luci), size(Π, 1), size(Π, 2), :)
    problem.solution.sitetensors[bondindex+1] = reshape(TCI.right(luci), :, size(Π, 3), size(Π, 4))

    if leftorthogonal
        updateleftframes!(problem, bondindex, TCI.rowindices(luci))
    else
        updaterightframes!(problem, bondindex + 1, TCI.colindices(luci))
    end
    problem.pivoterrors[bondindex] = TCI.lastpivoterror(luci)
    nothing
end

function sweep(indices::AbstractVector{<:Integer}; forward::Bool)
    if forward
        return indices
    else
        return Iterators.reverse(indices)
    end
end

function initializeproblem!(problem::ElementwiseProblem{ValueType,N}) where {ValueType,N}
    for siteindex in length(problem):-1:2
        # Bring solution into CI-canonical form
        A = problem.solution[siteindex]
        luci = TCI.MatrixLUCI(reshape(A, size(A, 1), :); leftorthogonal=false)
        problem.solution.sitetensors[siteindex] = reshape(TCI.right(luci), :, size(A, 2), size(A, 3))
        Anext = problem.solution[siteindex-1]
        Tnext = reshape(Anext, :, size(Anext, 3)) * TCI.left(luci)
        problem.solution.sitetensors[siteindex-1] = reshape(Tnext, size(Anext, 1), size(Anext, 2), :)

        # Initialize frame matrices
        Jset = TCI.colindices(luci)
        for inputindex in eachinputindex(problem)
            R = updaterightframe(
                problem.rightframes[inputindex, siteindex+1],
                problem.inputs[inputindex][siteindex],
                Jset
            )
            @debug "Updated right frame at $siteindex" size(R)
            problem.rightframes[inputindex, siteindex] = R
        end
    end
    nothing
end

function effectivetolerance(
    problem::ElementwiseProblem,
    truncationparameters::TruncationParameters
)
    if truncationparameters.scaletolerance && problem.maxsamplevalue > 0.0
        return truncationparameters.tolerance * problem.maxsamplevalue
    end
    return truncationparameters.tolerance
end

function validateelementwise(
    inputs,
    initial_guess,
    max_iters,
    min_iters,
    truncationparameters,
    nsearchglobalpivot,
    maxnglobalpivot,
    tolmarginglobalsearch
)
    isempty(inputs) && throw(ArgumentError("`inputs` must contain at least one tensor train."))
    nsites = length(first(inputs))
    nsites > 0 || throw(ArgumentError("Input tensor trains must contain at least one site."))
    all(length(input) == nsites for input in inputs) || throw(ArgumentError(
        "All input tensor trains must have the same number of sites."
    ))
    allequal(TCI.sitedims.(inputs)) || throw(ArgumentError(
        "All input tensor trains must have the same local dimensions."
    ))
    all(all(>(0), size(tensor)) for input in inputs for tensor in input) || throw(ArgumentError(
        "All input tensor dimensions must be positive."
    ))
    all(size(first(input), 1) == 1 && size(last(input), ndims(last(input))) == 1 for input in inputs) || throw(
        ArgumentError("Input tensor trains must have one-dimensional boundary bonds.")
    )
    max_iters >= 1 || throw(ArgumentError("`max_iters` must be at least 1."))
    min_iters >= 1 || throw(ArgumentError("`min_iters` must be at least 1."))
    min_iters <= max_iters || throw(ArgumentError("`min_iters` must not exceed `max_iters`."))
    truncationparameters.maxbonddimension >= 1 || throw(ArgumentError(
        "`maxbonddimension` must be at least 1."
    ))
    isfinite(truncationparameters.tolerance) && truncationparameters.tolerance >= 0 || throw(
        ArgumentError("The truncation tolerance must be finite and non-negative.")
    )
    nsearchglobalpivot >= 0 || throw(ArgumentError("`nsearchglobalpivot` must be non-negative."))
    maxnglobalpivot >= 0 || throw(ArgumentError("`maxnglobalpivot` must be non-negative."))
    isfinite(tolmarginglobalsearch) && tolmarginglobalsearch >= 0 || throw(ArgumentError(
        "`tolmarginglobalsearch` must be finite and non-negative."
    ))

    if !isnothing(initial_guess)
        length(initial_guess) == nsites || throw(ArgumentError(
            "The initial guess must have the same number of sites as the inputs."
        ))
        TCI.sitedims(initial_guess) == TCI.sitedims(first(inputs)) || throw(ArgumentError(
            "The initial guess must have the same local dimensions as the inputs."
        ))
        all(all(>(0), size(tensor)) for tensor in initial_guess) || throw(ArgumentError(
            "All initial-guess tensor dimensions must be positive."
        ))
        size(first(initial_guess), 1) == 1 &&
            size(last(initial_guess), ndims(last(initial_guess))) == 1 || throw(
            ArgumentError("The initial guess must have one-dimensional boundary bonds.")
        )
        all(TCI.linkdims(initial_guess) .<= truncationparameters.maxbonddimension) || throw(
            ArgumentError("The initial-guess bond dimensions must not exceed `maxbonddimension`.")
        )
    end
    nothing
end

function elementwiseonesite(op, inputs)
    result = TCI.TensorTrain([op.(getindex.(inputs, 1)...)])
    return result, Int[], Float64[]
end

function framecontainsrow(frame::AbstractMatrix, row::AbstractVector)
    return any(existing -> existing == row, eachrow(frame))
end

function framecontainscolumn(frame::AbstractMatrix, column::AbstractVector)
    return any(existing -> existing == column, eachcol(frame))
end

function padsolutionbonds!(problem::ElementwiseProblem{ValueType,N}, growth) where {ValueType,N}
    for (bondindex, amount) in enumerate(growth)
        amount == 0 && continue
        left = problem.solution.sitetensors[bondindex]
        right = problem.solution.sitetensors[bondindex + 1]
        problem.solution.sitetensors[bondindex] = cat(
            left,
            zeros(ValueType, size(left)[1:end-1]..., amount);
            dims=N
        )
        problem.solution.sitetensors[bondindex + 1] = cat(
            right,
            zeros(ValueType, amount, size(right)[2:end]...);
            dims=1
        )
    end
    nothing
end

function addglobalpivots!(problem::ElementwiseProblem, pivots, maxbonddimension)
    nbonds = length(problem) - 1
    bounds = prodbonddimensions(TCI.sitedims(problem.solution))[2:end-1]
    bonddimensions = TCI.linkdims(problem.solution)
    growth = zeros(Int, nbonds)
    nadded = 0

    for pivot in pivots
        length(pivot) == length(problem) || throw(ArgumentError(
            "A global pivot must contain one index per tensor-train site."
        ))
        pivotadded = false
        for bondindex in 1:nbonds
            bonddimensions[bondindex] < min(bounds[bondindex], maxbonddimension) || continue
            leftvalues = [
                TCI.evaluateleft(cache, view(pivot, 1:bondindex))
                for cache in problem.inputcaches
            ]
            rightvalues = [
                TCI.evaluateright(cache, view(pivot, bondindex+1:length(pivot)))
                for cache in problem.inputcaches
            ]
            represented = all(eachinputindex(problem)) do inputindex
                framecontainsrow(problem.leftframes[inputindex, bondindex], leftvalues[inputindex]) &&
                    framecontainscolumn(
                        problem.rightframes[inputindex, bondindex + 1],
                        rightvalues[inputindex]
                    )
            end
            represented && continue

            for inputindex in eachinputindex(problem)
                problem.leftframes[inputindex, bondindex] = vcat(
                    problem.leftframes[inputindex, bondindex],
                    transpose(leftvalues[inputindex])
                )
                problem.rightframes[inputindex, bondindex + 1] = hcat(
                    problem.rightframes[inputindex, bondindex + 1],
                    rightvalues[inputindex]
                )
            end
            bonddimensions[bondindex] += 1
            growth[bondindex] += 1
            pivotadded = true
        end
        nadded += pivotadded
    end

    padsolutionbonds!(problem, growth)
    return nadded
end

# Reuses TensorCrossInterpolation.jl's floating-zone walk (`src/globalsearch.jl`)
# so ACI and TCI apply the same global error search.
function findglobalpivots(
    op,
    problem::ElementwiseProblem{ValueType},
    abstol;
    nsearchglobalpivot,
    maxnglobalpivot,
    tolmarginglobalsearch,
    rng=Random.default_rng()
) where {ValueType}
    nsearchglobalpivot == 0 && return Vector{Vector{Int}}()
    maxnglobalpivot == 0 && return Vector{Vector{Int}}()

    flatdims = prod.(TCI.sitedims(problem.solution))
    cachedsolution = TCI.TTCache(problem.solution, [[d] for d in flatdims])
    threshold = Float64(abstol * tolmarginglobalsearch)
    exactvalue = function(index)
        value = op((cache(index) for cache in problem.inputcaches)...)
        problem.maxsamplevalue = max(problem.maxsamplevalue, abs(value))
        return value
    end

    candidates = Tuple{Vector{Int},Float64}[]
    for _ in 1:nsearchglobalpivot
        initial = [rand(rng, 1:d) for d in flatdims]
        pivot, error = TCI._floatingzone(
            cachedsolution,
            exactvalue;
            initp=initial,
            earlystoptol=threshold,
            nsweeps=100
        )
        if error > threshold && all(first(candidate) != pivot for candidate in candidates)
            push!(candidates, (copy(pivot), error))
        end
    end
    sort!(candidates; by=last, rev=true)
    return first.(candidates[1:min(maxnglobalpivot, length(candidates))])
end

function ranksaturated(ranks, min_iters, maxbonddimension)
    length(ranks) >= min_iters || return false
    return all(last(ranks, min_iters) .>= maxbonddimension)
end

function elementwise(op::Function, inputs::Vector{<:TensorTrain}; kwargs...)
    isempty(inputs) && throw(ArgumentError("`inputs` must contain at least one tensor train."))
    throw(ArgumentError("All input tensor trains must have the same value type and tensor order."))
end

@doc raw"""
    elementwise(
        op::Function,
        inputs::Vector{<:TensorTrain{ValueType,N}};
        max_iters::Integer=20,
        min_iters::Integer=2,
        truncationparameters::TruncationParameters=TruncationParameters(typemax(Int), 1e-12, true),
        initial_guess::Union{Nothing,TensorTrain}=nothing,
        nsearchglobalpivot::Integer=5,
        maxnglobalpivot::Integer=5,
        tolmarginglobalsearch::Real=10.0
    ) where {ValueType,N}

Compute the elementwise application of `op` to `inputs` with alternating cross
interpolation. The return value is `(result, ranks, errors)`, where `ranks` and
`errors` contain the maximum bond dimension and absolute pivot error after each
completed sweep. A one-site input is evaluated exactly and returns empty
histories.

After every sweep, a global floating-zone search checks points outside the
bond-local crosses. Pivots whose error exceeds the effective tolerance times
`tolmarginglobalsearch` are injected before the next sweep. Convergence requires
both stable ranks and no global pivots during the last `min_iters` sweeps. Set
`nsearchglobalpivot=0` to disable this guard.

The sweep also stops when the rank remains at
`truncationparameters.maxbonddimension` for `min_iters` sweeps, because no new
pivots can then be added.

# Arguments
- `op`: Function applied elementwise to one value from each input tensor train.
- `inputs`: Non-empty tensor trains with equal site counts and local dimensions.
- `max_iters`: Maximum number of sweeps.
- `min_iters`: Number of history entries required for convergence checks.
- `truncationparameters`: Bond cap, tolerance, and tolerance-scaling mode.
- `initial_guess`: Optional compatible initial tensor train. A random guess is used by default.
- `nsearchglobalpivot`: Number of random starts per global search.
- `maxnglobalpivot`: Maximum number of distinct global pivots injected per sweep.
- `tolmarginglobalsearch`: Multiplier applied to the effective tolerance when accepting global pivots.

# Throws
Throws `ArgumentError` for empty or incompatible inputs, an incompatible initial
guess, invalid iteration limits, invalid truncation parameters, or invalid
global-search parameters.

# Examples
```julia
import TensorCrossInterpolation as TCI
A = TCI.TensorTrain([reshape([1.0, 2.0], 1, 2, 1)])
B = TCI.TensorTrain([reshape([3.0, 4.0], 1, 2, 1)])
result, ranks, errors = elementwise(*, [A, B])
@assert result([1]) == 3.0
@assert result([2]) == 8.0
@assert isempty(ranks) && isempty(errors)
```
"""
function elementwise(
    op::Function,
    inputs::Vector{<:TensorTrain{ValueType,N}};
    max_iters::Integer=20,
    min_iters::Integer=2,
    truncationparameters::TruncationParameters=TruncationParameters(typemax(Int), 1e-12, true),
    initial_guess::Union{Nothing,TensorTrain}=nothing,
    nsearchglobalpivot::Integer=5,
    maxnglobalpivot::Integer=5,
    tolmarginglobalsearch::Real=10.0
) where {ValueType,N}
    validateelementwise(
        inputs,
        initial_guess,
        max_iters,
        min_iters,
        truncationparameters,
        nsearchglobalpivot,
        maxnglobalpivot,
        tolmarginglobalsearch
    )
    length(first(inputs)) == 1 && return elementwiseonesite(op, inputs)

    guess = if isnothing(initial_guess)
        randomtt(
            ValueType,
            TCI.sitedims(first(inputs)),
            min.(
                min.([TCI.linkdims(input) for input in inputs]...),
                truncationparameters.maxbonddimension
            )
        )
    else
        initial_guess
    end
    problem = ElementwiseProblem(inputs, guess)
    ranks = Int[]
    errors = Float64[]
    nglobalpivots = Int[]
    globalpivots = Vector{Vector{Int}}()

    function convergencecriterion(iteration)
        iteration >= min_iters || return false
        errors[iteration] <= effectivetolerance(problem, truncationparameters) || return false
        any(last(ranks, min_iters) .> ranks[iteration-min_iters+1]) && return false
        return all(last(nglobalpivots, min_iters) .== 0)
    end

    for iteration in 1:max_iters
        forward = isodd(iteration)
        for bondindex in sweep(eachbondindex(problem); forward)
            localupdate!(op, problem, bondindex; leftorthogonal=forward, truncationparameters)
        end

        @debug "Sweep $iteration, $(forward ? "forward" : "backward")" bonddimensions = "$(TCI.linkdims(problem.solution))" pivoterrors = "$(problem.pivoterrors)"

        push!(ranks, TCI.rank(problem.solution))
        push!(errors, maximum(problem.pivoterrors))

        globalpivots = if last(ranks) < truncationparameters.maxbonddimension
            findglobalpivots(
                op,
                problem,
                effectivetolerance(problem, truncationparameters);
                nsearchglobalpivot,
                maxnglobalpivot,
                tolmarginglobalsearch
            )
        else
            Vector{Vector{Int}}()
        end
        addglobalpivots!(problem, globalpivots, truncationparameters.maxbonddimension)
        push!(nglobalpivots, length(globalpivots))

        convergencecriterion(iteration) && break
        ranksaturated(ranks, min_iters, truncationparameters.maxbonddimension) && break
    end

    # Absorb pivots injected by the final allowed iteration before returning.
    if !isempty(globalpivots)
        for bondindex in eachbondindex(problem)
            localupdate!(op, problem, bondindex; leftorthogonal=true, truncationparameters)
        end
    end

    return problem.solution, ranks, errors
end
