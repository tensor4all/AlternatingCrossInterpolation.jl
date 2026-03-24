mutable struct Sweep
    axis::Vector{Int}
    forward::Bool
    function Sweep(axis; forward::Bool=true)
        return new(Vector{Int}(axis), forward)
    end
end

function Sweep(
    axis, nsweep;
    sweepstrategy::Symbol=:backandforth,
)
    forward = sweepstrategy == :forward ||
              (sweepstrategy == :backandforth && isodd(nsweep))
    return Sweep(axis; forward=forward)
end

function Base.iterate(sweep::Sweep, i)
    thissite = sweep.axis[i]
    i += sweep.forward ? +1 : -1
    if i < firstindex(sweep.axis) || i > lastindex(sweep.axis)
        return nothing
    else
        return (thissite, sweep.axis[i]), i
    end
end

function Base.iterate(sweep::Sweep)
    if isempty(sweep.axis)
        return nothing
    end
    i = sweep.forward ? firstindex(sweep.axis) : lastindex(sweep.axis)
    return iterate(sweep, i)
end

function Base.length(sweep::Sweep)
    return length(sweep.axis) - 1
end
