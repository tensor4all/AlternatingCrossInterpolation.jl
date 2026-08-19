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

function environments(
    problem::ElementwiseProblem{ValueType,3}, bondindex::Integer
) where {ValueType}
    sol = problem.solution
    # sum over local dimensions
    sum_L = [sum(sol.sitetensors[b]; dims=2) for b in 1:bondindex-1]
    sum_R = [sum(sol.sitetensors[b]; dims=2) for b in reverse(bondindex+2:length(sol.sitetensors))]
    sum_L = [dropdims(s, dims=2) for s in sum_L]
    sum_R = [dropdims(s, dims=2) for s in sum_R]
    isempty(sum_L) && (sum_L = [fill(one(ValueType), 1, 1)])
    isempty(sum_R) && (sum_R = [fill(one(ValueType), 1, 1)])
    L = reduce(*, sum_L)
    R = reduce((a, b) -> b * a, sum_R)
    @assert size(L)[begin] == 1
    @assert size(R)[end] == 1
    return dropdims(L, dims=1), dropdims(R, dims=ndims(R))
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

function updatemaxsample!(
    problem::ElementwiseProblem{ValueType,N},
    Π::AbstractMatrix{ValueType},
    bw::Function
    ) where {ValueType,N}
    maxΠ_bw = 0.0
    for I in CartesianIndices(Π)
        maxΠ_bw = max(maxΠ_bw, abs(bw(I[1],I[2]) * Π[I]))
    end
    problem.maxsamplevalue = max(problem.maxsamplevalue, maxΠ_bw)
end

function updatemaxsample!(problem::ElementwiseProblem{ValueType,N}, Π::AbstractMatrix{ValueType},::Nothing) where {ValueType,N}
    problem.maxsamplevalue = max(problem.maxsamplevalue, maximum(abs, Π))
end

function localupdate!(
    op::Function,
    problem::ElementwiseProblem{ValueType,N},
    bondindex::Integer;
    leftorthogonal::Bool,
    truncationparameters::TruncationParameters,
    errorweighting::ErrorWeighting,
    environment_mode::Bool,
    normalize_environment::Bool, # only relevant if environment_mode is true
) where {ValueType,N}
    @debug "Local update" bondindex leftorthogonal
    Πs = [pitensor(problem, k, bondindex) for k in eachinputindex(problem)]
    Π = op.(Πs...)

    bw = nothing
    if errorweighting.is_nontrivial
        updatecombinedIJ!(errorweighting, bondindex)
        bw = bondweighting(errorweighting, bondindex)
    end
    if environment_mode
        # sum frames
        L, R = environments(problem, bondindex)
        ml, mr = normalize_environment ? (maximum(abs, L), maximum(abs, R)) : (1.0, 1.0)
        # L, R live on virtual bonds χ₁, χ₂ (physical legs already summed in environments).
        # Combined LU indices i,j map via CartesianIndices of (χ₁,d) and (d,χ₂).
        cl = CartesianIndices(size(Π)[1:2])
        cr = CartesianIndices(size(Π)[3:4])
        if isnothing(bw)
            bw = (i, j) -> abs(L[cl[i][1]] * R[cr[j][2]]) / (ml * mr)
        else
            bw_error = bw
            bw = (i, j) -> abs(bw_error(i, j) * L[cl[i][1]] * R[cr[j][2]]) / (ml * mr)
        end
    end

    Πmat = reshape(Π, prod(size(Π)[1:2]), prod(size(Π)[3:4]))
    if truncationparameters.scaletolerance
        updatemaxsample!(problem, Πmat, bw)
    end

    abstol = if truncationparameters.scaletolerance
        truncationparameters.tolerance * problem.maxsamplevalue
    else
        truncationparameters.tolerance
    end
    luci = TCI.MatrixLUCI(
        Πmat;
        bondweighting=bw,
        leftorthogonal=leftorthogonal,
        maxrank=truncationparameters.maxbonddimension,
        abstol=abstol
    )

    if errorweighting.is_nontrivial
        updatepivots!(errorweighting, bondindex, luci)
    end

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

@doc raw"""
    elementwise(
        op::Function,
        inputs::Vector{<:TensorTrain{ValueType,N}};
        max_iters::Integer=20, min_iters::Integer=2,
        truncationparameters::TruncationParameters=TruncationParameters(typemax(Int), 1e-12, true),
        initial_guess::TensorTrain=randomtt(ValueType, TCI.sitedims(inputs[1]), min.([TCI.linkdims(X) for X in inputs]...))
    ) where {ValueType,N}
    
Compute the elementwise application of `op` to the input tensor trains in `inputs` using an alternating optimization procedure. The function returns a tuple containing the resulting tensor train, a vector of bond dimensions at each iteration, and a vector of pivot errors at each iteration. For example, to compute a Hadamard product of two tensor trains `A` and `B`, you could call:
```julia
result, ranks, errors = elementwise(*, [A, B])
``` 

# Arguments
- `op::Function`: The function to apply elementwise to the input tensor trains.
- `inputs::Vector{<:TensorTrain{ValueType,N}}`: A vector of tensor trains to which the function will be applied. All tensor trains must have the same number of sites and the same local dimensions.
- `max_iters::Integer=20`: The maximum number of iterations to perform.
- `min_iters::Integer=2`: The minimum number of iterations to perform before checking for convergence.
- `truncationparameters::TruncationParameters=TruncationParameters(typemax(Int), 1e-12, true)`: Parameters controlling the truncation of the tensor train during the optimization process.
- `initial_guess::TensorTrain=randomtt(ValueType, TCI.sitedims(inputs[1]), min.([TCI.linkdims(X) for X in inputs]...))`: An initial guess for the resulting tensor train. If not provided, a random tensor train with appropriate dimensions will be generated.
"""
function elementwise(
    op::Function,
    inputs::Vector{<:TensorTrain{ValueType,N}};
    max_iters::Integer=20, min_iters::Integer=2,
    truncationparameters::TruncationParameters=TruncationParameters(typemax(Int), 1e-12, true),
    initial_guess::TensorTrain=randomtt(ValueType, TCI.sitedims(inputs[1]), min.([TCI.linkdims(X) for X in inputs]...)),
    weighting::Union{Function, Nothing}=nothing,
    environment_mode::Bool=false,
    normalize_environment::Bool=false, # only relevant if environment_mode is true
) where {ValueType,N}
    if any(length.(inputs) .!= length(inputs[1]))
        throw(ArgumentError("All input tensor trains must have the same number of sites."))
    end
    if !allequal(TCI.sitedims, inputs)
        throw(ArgumentError("All input tensor trains must have the same local dimensions."))
    end

    problem = ElementwiseProblem{ValueType,N}(inputs, initial_guess)
    @debug "Frame sizes" size.(problem.rightframes[1, :]) size.(problem.rightframes[2, :])
    errorweighting = ErrorWeighting(problem, weighting, !isnothing(weighting))
    initializeproblem!(problem)

    ranks = Int[]
    errors = Float64[]

    function convergencecriterion(iteration)
        tol = if truncationparameters.scaletolerance
            truncationparameters.tolerance * problem.maxsamplevalue
        else
            truncationparameters.tolerance
        end
        if iteration < min_iters
            return false
        elseif errors[iteration] > tol
            return false
        elseif any(last(ranks, min_iters) .> ranks[iteration-min_iters+1])
            return false
        else
            return true
        end
    end

    for iteration in 1:max_iters
        forward = isodd(iteration)
        for bondindex in sweep(eachbondindex(problem); forward)
            localupdate!(
                op,
                problem,
                bondindex; errorweighting=errorweighting,
                leftorthogonal=forward,
                truncationparameters,
                environment_mode=environment_mode,
                normalize_environment=normalize_environment
            )
        end

        @debug "Sweep $iteration, $(forward ? "forward" : "backward")" bonddimensions = "$(TCI.linkdims(problem.solution))" pivoterrors = "$(problem.pivoterrors)"

        push!(ranks, TCI.rank(problem.solution))
        push!(errors, maximum(problem.pivoterrors))

        if convergencecriterion(iteration)
            break
        end
    end
    return problem.solution, ranks, errors
end
