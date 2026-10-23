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

function twopeaktt(nsites)
    localdim = 4
    gauss(peak, value) = exp(-0.5 * (value - peak)^2)
    tensors = Array{Float64,3}[]
    for site in 1:nsites
        leftdim, rightdim = site == 1 ? (1, 2) : site == nsites ? (2, 1) : (2, 2)
        tensor = zeros(leftdim, localdim, rightdim)
        for right in 1:rightdim, value in 1:localdim, left in 1:leftdim
            tensor[left, value, right] = if site == 1 && left == 1 && right == 1
                3.0 * gauss(1, value)
            elseif site == 1 && left == 1 && right == 2
                2.0 * gauss(4, value)
            elseif site == nsites && left == 2 && right == 1
                gauss(4, value)
            elseif left == 1 && right == 1
                gauss(1, value)
            elseif left == 2 && right == 2
                gauss(4, value)
            else
                0.0
            end
        end
        push!(tensors, tensor)
    end
    return TCI.TensorTrain(tensors)
end

@testset "One-site elementwise operation" begin
    A = TCI.TensorTrain([reshape([1.0, 2.0], 1, 2, 1)])
    B = TCI.TensorTrain([reshape([3.0, 4.0], 1, 2, 1)])

    result, ranks, errors = ACI.elementwise(*, [A, B])

    @test result([1]) == 3.0
    @test result([2]) == 8.0
    @test isempty(ranks)
    @test isempty(errors)
end

@testset "Public input validation" begin
    A = TCI.TensorTrain([ones(1, 2, 1), ones(1, 2, 1)])
    shorter = TCI.TensorTrain([ones(1, 2, 1)])
    different_site = TCI.TensorTrain([ones(1, 3, 1), ones(1, 2, 1)])
    emptytt = TCI.TensorTrain{Float64,3}(Array{Float64,3}[])
    large_guess = ACI.randomtt(Float64, fill([2], 2), [2])
    complex_guess = TCI.TensorTrain([ones(ComplexF64, 1, 2, 1) for _ in 1:2])

    @test_throws ArgumentError ACI.elementwise(identity, TCI.TensorTrain[])
    @test_throws ArgumentError ACI.elementwise(identity, [emptytt])
    @test_throws ArgumentError ACI.elementwise(identity, TCI.TensorTrain{Float64,3}[A, shorter])
    @test_throws ArgumentError ACI.elementwise(identity, [A, different_site])
    @test_throws ArgumentError ACI.elementwise(identity, [A]; max_iters=0)
    @test_throws ArgumentError ACI.elementwise(identity, [A]; min_iters=0)
    @test_throws ArgumentError ACI.elementwise(identity, [A]; max_iters=1, min_iters=2)
    @test_throws ArgumentError ACI.elementwise(
        identity,
        [A];
        truncationparameters=ACI.TruncationParameters(0, 1e-12, false)
    )
    @test_throws ArgumentError ACI.elementwise(
        identity,
        [A];
        truncationparameters=ACI.TruncationParameters(1, NaN, false)
    )
    @test_throws ArgumentError ACI.elementwise(
        identity,
        [A];
        truncationparameters=ACI.TruncationParameters(1, 1e-12, false),
        initial_guess=large_guess
    )
    @test_throws ArgumentError ACI.elementwise(identity, [A]; initial_guess=complex_guess)
    @test_throws ArgumentError ACI.elementwise(identity, [A]; nsearchglobalpivot=-1)
    @test_throws ArgumentError ACI.elementwise(identity, [A]; maxnglobalpivot=-1)
    @test_throws ArgumentError ACI.elementwise(identity, [A]; tolmarginglobalsearch=Inf)
end

@testset "Global guard recovers a separated peak" begin
    nsites = 10
    input = twopeaktt(nsites)
    guess = TCI.TensorTrain([ones(1, 4, 1) for _ in 1:nsites])
    parameters = ACI.TruncationParameters(typemax(Int), 1e-4, true)

    without_guard, = ACI.elementwise(
        identity,
        [input];
        truncationparameters=parameters,
        initial_guess=guess,
        nsearchglobalpivot=0
    )
    Random.seed!(0)
    with_guard, = ACI.elementwise(
        identity,
        [input];
        truncationparameters=parameters,
        initial_guess=guess,
        nsearchglobalpivot=30
    )

    @test abs(without_guard(fill(4, nsites)) - 2.0) > 1.0
    @test abs(with_guard(fill(1, nsites)) - 3.0) < 1e-8
    @test abs(with_guard(fill(4, nsites)) - 2.0) < 1e-8
end

@testset "Binding bond cap stops early" begin
    Random.seed!(1)
    A = ACI.randomtt(Float64, fill([2], 6), 2)
    B = ACI.randomtt(Float64, fill([2], 6), 2)
    guess = TCI.TensorTrain([ones(1, 2, 1) for _ in 1:6])
    parameters = ACI.TruncationParameters(1, 0.0, false)

    result, ranks, errors = ACI.elementwise(
        *,
        [A, B];
        max_iters=10,
        min_iters=2,
        truncationparameters=parameters,
        initial_guess=guess
    )

    @test length(ranks) == 2
    @test all(ranks .== 1)
    @test all(errors .> 0)
    @test all(TCI.linkdims(result) .<= 1)
end

@testset "Scaled and absolute tolerances" begin
    input = TCI.TensorTrain([ones(1, 2, 1), ones(1, 2, 1)])
    problem = ACI.ElementwiseProblem([input], input)
    problem.maxsamplevalue = 100.0

    @test ACI.effectivetolerance(problem, ACI.TruncationParameters(10, 1e-4, true)) == 1e-2
    @test ACI.effectivetolerance(problem, ACI.TruncationParameters(10, 1e-4, false)) == 1e-4
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
    @test_throws ArgumentError ACI.elementwise(
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

