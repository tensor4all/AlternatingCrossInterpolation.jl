mutable struct ElementwiseProblem{ValueType,N}
    inputs::Vector{TensorTrain{ValueType,N}}
    solution::TensorTrain{ValueType,N}

    leftframes::OffsetMatrix{Matrix{ValueType},Matrix{Matrix{ValueType}}}
    rightframes::OffsetMatrix{Matrix{ValueType},Matrix{Matrix{ValueType}}}

    pivoterrors::Vector{Float64}
    maxsamplevalue::Float64

    function ElementwiseProblem{ValueType,N}(
        inputs::Vector{TensorTrain{ValueType,N}},
        initial_guess::TensorTrain{ValueType,N}
    ) where {ValueType,N}
        Ninputs = length(inputs)
        Nsites = length(first(inputs))

        @assert all(length.(inputs) .== Nsites) "All input tensor trains must have the same number of sites."
        @assert allequal(TCI.sitedims, inputs) "All input tensor trains must have the same local dimensions."

        problem = new{ValueType,N}(
            inputs,
            deepcopy(initial_guess),
            Origin(1, 0)(Matrix{Matrix{ValueType}}(undef, Ninputs, Nsites + 1)),
            Origin(1, 1)(Matrix{Matrix{ValueType}}(undef, Ninputs, Nsites + 1)),
            zeros(Nsites),
            0.0 # maxsamplevalue
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
