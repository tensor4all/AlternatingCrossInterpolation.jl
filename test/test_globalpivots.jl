import AlternatingCrossInterpolation as ACI
import TensorCrossInterpolation as TCI
using Test

function zerott(::Type{T}, sitedims) where {T}
    return TCI.TensorTrain([zeros(T, 1, d..., 1) for d in sitedims])
end

@testset "global pivot helpers" begin
    sitedims = fill([2], 3)
    pivot = [2, 1, 2]

    delta = ACI.delta_tensortrain(Float64, sitedims, pivot, 3.5)
    @test delta(pivot) == 3.5
    @test delta([1, 1, 2]) == 0.0
    @test delta([2, 2, 2]) == 0.0
    @test TCI.rank(delta) == 1

    base = zerott(Float64, sitedims)
    f(q) = q == pivot ? 3.5 : 0.0
    enriched, nadded, maxsample = ACI.spikeenrich(base, f, [pivot])
    @test nadded == 1
    @test maxsample == 3.5
    @test enriched(pivot) ≈ 3.5
    @test enriched([1, 1, 2]) ≈ 0.0
end

struct FixedPivotFinder <: TCI.AbstractGlobalPivotFinder
    pivots::Vector{Vector{Int}}
end

function (finder::FixedPivotFinder)(
    input::TCI.GlobalPivotSearchInput{ValueType},
    f,
    abstol::Float64;
    verbosity::Int=0,
    rng=nothing,
)::Vector{Vector{Int}} where {ValueType}
    return finder.pivots
end

@testset "elementwise global pivot enrichment" begin
    sitedims = fill([2], 4)
    pivot = [2, 1, 2, 1]
    # Dense TTs avoid singular local LU that occurs for sparse delta inputs.
    input = ACI.randomtt(Float64, sitedims, 3)
    initial = ACI.randomtt(Float64, sitedims, 1)

    result, ranks, errors = ACI.elementwise(
        identity,
        [input];
        initial_guess=deepcopy(initial),
        max_iters=1,
        min_iters=2,
        useglobalpivots=true,
        globalpivotfinder=FixedPivotFinder([pivot]),
        truncationparameters=ACI.TruncationParameters(1, 1e-12, true),
    )

    @test result(pivot) ≈ input(pivot)
    @test length(ranks) == 1
    @test length(errors) == 1
    # Local updates are capped at bond dimension 1; spike enrichment raises the rank.
    @test TCI.rank(result) > 1
end

@testset "elementwise rejects global pivots for non-scalar site dimensions" begin
    input = ACI.randomtt(Float64, [[2, 2], [2, 2]], 1)
    @test_throws ArgumentError ACI.elementwise(
        identity,
        [input];
        max_iters=1,
        min_iters=2,
        useglobalpivots=true,
    )
end

@testset "default global pivot finder wrapper" begin
    sitedims = fill([2], 3)
    current = zerott(Float64, sitedims)
    f(q) = 1.0

    pivots = ACI.findglobalpivots(
        current,
        f,
        1e-12;
        nsearchglobalpivot=3,
        maxnglobalpivot=2,
        tolmarginglobalsearch=1.0,
    )

    @test length(pivots) == 2
    @test all(length(p) == 3 for p in pivots)
    @test all(all(1 .<= p .<= 2) for p in pivots)
end
