@testset "PivotTracker initialize_pivots!" begin
    R = 4
    localdims = fill([2], R)
    guess = ACI.randomtt(Float64, localdims, [3, 2, 1]; clampbonddimensions=true)
    A = ACI.randomtt(Float64, localdims, 2)
    B = ACI.randomtt(Float64, localdims, 2)

    # Pristine problem (inner ctor): must match what initialize_pivots! / initializeproblem! both see
    problem = ACI.ElementwiseProblem{Float64,3}([A, B], guess)
    pt = ACI.PivotTracker(problem)

    @test pt.Iset[1] == [Int[]]
    @test pt.Jset[R] == [Int[]]
    @test all(isempty, pt.Iset[2:R])  # left sets filled only during forward updates

    # Right sets seeded for sites 1..R-1; lengths equal CI ranks from the same sweep
    for b in 1:R-1
        @test !isempty(pt.Jset[b])
        @test all(length(j) == R - b for j in pt.Jset[b])
    end

    # After initializeproblem!, right-frame column counts must match length.(Jset)
    ACI.initializeproblem!(problem)
    for site in 2:R
        χ = size(problem.rightframes[1, site], 2)
        @test length(pt.Jset[site - 1]) == χ
    end

    # Combined indices at bond 1 must be well-defined and correctly sized
    ACI.updatecombinedIJ!(pt, 1)
    @test length(pt.Icombined_act) == pt.localdims[1] * length(pt.Iset[1])
    @test length(pt.Jcombined_act) == pt.localdims[2] * length(pt.Jset[2])
    @test pt.bond_act == 1
end
