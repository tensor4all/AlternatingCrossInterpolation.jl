mutable struct PivotTracker
    Iset::Vector{Vector{Vector{Int}}}
    Jset::Vector{Vector{Vector{Int}}}
    localdims::Vector{Int}
    Icombined_act::Vector{Vector{Int}}
    Jcombined_act::Vector{Vector{Int}}
    bond_act::Int

    # only for N=3 for now
    function PivotTracker(problem::ElementwiseProblem{ValueType,3}) where ValueType
        n = length(first(problem.inputs))
        pt = new(
            [Vector{Vector{Int}}() for _ in 1:n],
            [Vector{Vector{Int}}() for _ in 1:n],
            only.(TCI.sitedims(first(problem.inputs))),
            [Vector{Int}()],
            [Vector{Int}()],
            0,
        )
        return initialize_pivots!(pt, problem)
    end
end

function initialize_pivots!(
    pt::PivotTracker, problem::ElementwiseProblem{ValueType,3}
) where ValueType
    n = length(problem)
    pt.Iset = [Vector{Vector{Int}}() for _ in 1:n]
    pt.Jset = [Vector{Vector{Int}}() for _ in 1:n]
    pt.Iset[1] = [Int[]]
    pt.Jset[n] = [Int[]]
    pt.Icombined_act = [Vector{Int}()]
    pt.Jcombined_act = [Vector{Int}()]
    pt.bond_act = 0

    tensors = [copy(problem.solution[site]) for site in 1:n]
    for siteindex in n:-1:2
        A = tensors[siteindex]
        luci = TCI.MatrixLUCI(reshape(A, size(A, 1), :); leftorthogonal=false)
        Jcombined = TCI.kronecker(pt.localdims[siteindex], pt.Jset[siteindex])
        pt.Jset[siteindex - 1] = Jcombined[TCI.colindices(luci)]

        tensors[siteindex] = reshape(TCI.right(luci), :, size(A, 2), size(A, 3))
        Anext = tensors[siteindex - 1]
        Tnext = reshape(Anext, :, size(Anext, 3)) * TCI.left(luci)
        tensors[siteindex - 1] = reshape(Tnext, size(Anext, 1), size(Anext, 2), :)
    end
    return pt
end

function updatecombinedIJ!(pt::PivotTracker, bondindex::Integer)
    pt.Icombined_act = TCI.kronecker(pt.Iset[bondindex], pt.localdims[bondindex])
    pt.Jcombined_act = TCI.kronecker(pt.localdims[bondindex+1], pt.Jset[bondindex+1])
    pt.bond_act = bondindex
    return nothing
end

function updatepivots!(pt::PivotTracker, b::Integer, luci::TCI.MatrixLUCI)
    Icombined = if b==pt.bond_act
        pt.Icombined_act
    else
        TCI.kronecker(pt.Iset[b], pt.localdims[b])
    end
    Jcombined = if b==pt.bond_act
        pt.Jcombined_act
    else
        TCI.kronecker(pt.localdims[b+1], pt.Jset[b+1])
    end
    pt.Iset[b+1] = Icombined[TCI.rowindices(luci)]
    pt.Jset[b] = Jcombined[TCI.colindices(luci)]
    return nothing
end
