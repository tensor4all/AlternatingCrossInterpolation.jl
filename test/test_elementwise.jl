@testset "ElementwiseProblem struct" begin
    R = 6
    localdims = fill([4], R)
    linkdims = [2, 8, 10, 5, 3]
    A = ACI.randomtt(Float64, localdims, linkdims)
    B = ACI.randomtt(Float64, localdims, linkdims)
    problem = ACI.ElementwiseProblem([A, B], B)

    @test problem.inputs[1] == A
    @test problem.inputs[2] == B
    @test problem.pivoterrors == zeros(R)

    @test length(problem) == R
    @test all(ACI.eachsiteindex(problem) .== 1:R)
    @test all(ACI.eachbondindex(problem) .== 1:R-1)
    @test ACI.eachinputindex(problem) == [1, 2]
end

@testset "pitensor" begin
    A = rand(1, 4)
    B = rand(4, 2, 5)
    C = rand(5, 3, 6)
    D = rand(6, 7)

    Pi = ACI.pitensor(A, B, C, D)
    @test size(Pi) == (1, 2, 3, 7)

    AB = A * reshape(B, 4, 10)
    CD = reshape(C, 15, 6) * D
    @test reshape(Pi, 2, 21) ≈ reshape(AB, 2, 5) * reshape(CD, 5, 21)
end

function fulltensor(tt::TCI.AbstractTensorTrain)
    sitetensors = TCI.sitetensors(tt)
    return reduce(sitetensors) do U, V
        ACI.contract(U, [ndims(U)], V, [1])
    end
end

@testset "ElementwiseProblem: Gaussians" begin
    R = 7

    f(x) = exp(-sum(x .^ 2))
    g(x) = prod(cos.(x)) * exp(-sum((x .- 4) .^ 2))

    qg = QG.DiscretizedGrid{2}(R, (0., 0.), (5., 5.), unfoldingscheme=:fused)
    Fqtt, _, _ = TCI.crossinterpolate2(
        Float64,
        q -> f(QG.quantics_to_origcoord(qg, q)),
        QG.localdimensions(qg),
        tolerance=1e-12
    )
    Gqtt, _, _ = TCI.crossinterpolate2(
        Float64,
        q -> g(QG.quantics_to_origcoord(qg, q)),
        QG.localdimensions(qg),
        tolerance=1e-12
    )

    Ftt = TCI.tensortrain(Fqtt)
    Gtt = TCI.tensortrain(Gqtt)
    FGtt, = ACI.elementwise(.*, [Ftt, Gtt])

    fullF = fulltensor(Ftt)
    fullG = fulltensor(Gtt)
    fullFGexact = fullF .* fullG
    fullFG = fulltensor(FGtt)

    @test maximum(abs, fullFG .- fullFGexact) < 1e-10
end

@testset "ElementwiseProblem: Gaussians with tolerance scaling" begin
    R = 7
    for scale in (1.e4, 1.e-4)
        # put Gaussians closer to each other this time
        f(x) = scale * exp(-sum(x .^ 2))
        g(x) = scale * prod(cos.(x)) * exp(-sum((x .- 0.2) .^ 2))

        qg = QG.DiscretizedGrid{2}(R, (0., 0.), (5., 5.), unfoldingscheme=:fused)
        Fqtt, _, _ = TCI.crossinterpolate2(
            Float64,
            q -> f(QG.quantics_to_origcoord(qg, q)),
            QG.localdimensions(qg),
            tolerance=1e-12
        )
        Gqtt, _, _ = TCI.crossinterpolate2(
            Float64,
            q -> g(QG.quantics_to_origcoord(qg, q)),
            QG.localdimensions(qg),
            tolerance=1e-12
        )

        Ftt = TCI.tensortrain(Fqtt)
        Gtt = TCI.tensortrain(Gqtt)
        FGtt, = ACI.elementwise(.*, [Ftt, Gtt]; truncationparameters=ACI.TruncationParameters(typemax(Int), 1e-12, true))

        fullF = fulltensor(Ftt)
        fullG = fulltensor(Gtt)
        fullFGexact = fullF .* fullG
        fullFG = fulltensor(FGtt)

        @test maximum(abs, fullFG .- fullFGexact) / maximum(abs, fullFGexact) < 1e-10
    end
end


@testset "Elementwise multiplication" begin
    f(x) = prod(cos.(22 .* x))# * exp(-sum(x .^ 2))
    g(x) = prod(cos.(23 .* x)) * exp(-sum((x .- 4) .^ 2))
    R = 12

    qg = QG.DiscretizedGrid{1}(R, (0.,), (5.,), unfoldingscheme=:fused)
    Fqtt, _, _ = TCI.crossinterpolate2(
        Float64,
        q -> f(QG.quantics_to_origcoord(qg, q)),
        QG.localdimensions(qg),
        tolerance=1e-12
    )
    Gqtt, _, _ = TCI.crossinterpolate2(
        Float64,
        q -> g(QG.quantics_to_origcoord(qg, q)),
        QG.localdimensions(qg),
        tolerance=1e-12
    )

    # FGqtt = TTContractions.ci.multiply(Fqtt, Gqtt; tolerance=1e-12, maxbonddim=100)
    FGqtt, = ACI.elementwise(
        *, TCI.tensortrain.([Fqtt, Gqtt]);
        truncationparameters=ACI.TruncationParameters(100, 1e-12, true),
        # initial_guess=TCI.tensortrain([rand(20, 2, 20) for _ in 1:R])
    )


    xindices = Int.(round.(range(1, stop=2^R, length=500)))
    xgrid = [QG.grididx_to_origcoord(qg, (i,))[1] for i in xindices]
    xquantics = [QG.grididx_to_quantics(qg, (i,)) for i in xindices]
    Fdata = [Fqtt(q) for q in xquantics]
    Gdata = [Gqtt(q) for q in xquantics]
    FGdata = [FGqtt(q) for q in xquantics]
    Exactdata = [f([x]) * g([x]) for x in xgrid]

    @test maximum(abs, FGdata .- Exactdata) < 1e-10
end

@testset "Elementwise multiplication with initial guess" begin
    f(x) = prod(cos.(22 .* x))# * exp(-sum(x .^ 2))
    g(x) = prod(cos.(23 .* x)) * exp(-sum((x .- 4) .^ 2))
    R = 12

    qg = QG.DiscretizedGrid{1}(R, (0.,), (5.,), unfoldingscheme=:fused)
    Fqtt, _, _ = TCI.crossinterpolate2(
        Float64,
        q -> f(QG.quantics_to_origcoord(qg, q)),
        QG.localdimensions(qg),
        tolerance=1e-12
    )
    Gqtt, _, _ = TCI.crossinterpolate2(
        Float64,
        q -> g(QG.quantics_to_origcoord(qg, q)),
        QG.localdimensions(qg),
        tolerance=1e-12
    )

    initialguess_bad = ACI.randomtt(Float64, TCI.sitedims(Fqtt), fill(20, R + 1), clampbonddimensions=false)
    @test_throws BoundsError ACI.elementwise(
        *, TCI.tensortrain.([Fqtt, Gqtt]);
        truncationparameters=ACI.TruncationParameters(100, 1e-12, true),
        initial_guess=initialguess_bad
        )
        
    initialguess_good = ACI.randomtt(Float64, TCI.sitedims(Fqtt), fill(20, R + 1), clampbonddimensions=true)
    FGqtt, = ACI.elementwise(
        *, TCI.tensortrain.([Fqtt, Gqtt]);
        truncationparameters=ACI.TruncationParameters(100, 1e-12, true),
        initial_guess=initialguess_good
    )

    xindices = Int.(round.(range(1, stop=2^R, length=500)))
    xgrid = [QG.grididx_to_origcoord(qg, (i,))[1] for i in xindices]
    xquantics = [QG.grididx_to_quantics(qg, (i,)) for i in xindices]
    Fdata = [Fqtt(q) for q in xquantics]
    Gdata = [Gqtt(q) for q in xquantics]
    FGdata = [FGqtt(q) for q in xquantics]
    Exactdata = [f([x]) * g([x]) for x in xgrid]

    @test maximum(abs, FGdata .- Exactdata) < 1e-10
end

