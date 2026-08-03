struct ErrorWeighting
    pt::PivotTracker
    weighting::Function
    is_nontrivial::Bool
    eval_cache::Vector{Int}

    function ErrorWeighting(
        problem::ElementwiseProblem{ValueType,3},
        weighting::Union{Function,Nothing}=nothing,
        is_nontrivial::Bool=false,
    ) where ValueType
        pt = PivotTracker(problem)
        w = isnothing(weighting) ? (v -> one(ValueType)) : weighting
        new(
            pt,
            w,
            is_nontrivial,
            Vector{Int}(undef, length(first(problem.inputs)))
        )
    end
end

function bondweighting(errorweighting::ErrorWeighting, bondindex::Integer)
    errorweighting.pt.bond_act==bondindex || throw(ArgumentError("bondindex must be the current active bond $(errorweighting.pt.bond_act), but is $bondindex"))
    function _bw(i,j)
        errorweighting.eval_cache[1:bondindex] .= errorweighting.pt.Icombined_act[i]
        errorweighting.eval_cache[bondindex+1:end] .= errorweighting.pt.Jcombined_act[j]
        return errorweighting.weighting(errorweighting.eval_cache)
    end
    return _bw
end

function updatepivots!(errorweighting::ErrorWeighting, bondindex::Integer, luci::TCI.MatrixLUCI)
    updatepivots!(errorweighting.pt, bondindex, luci)
end

function updatecombinedIJ!(errorweighting::ErrorWeighting, bondindex::Integer)
    updatecombinedIJ!(errorweighting.pt, bondindex)
    return nothing
end
