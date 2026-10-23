function sum_rank1(a::Vector{Float64}, b::Vector{Float64})
    d = length(a)
    T1 = zeros(1, d, 2)
    T1[1, :, 1] = a
    T1[1, :, 2] = b
    T2 = zeros(2, d, 2)
    T2[1, :, 1] = a
    T2[2, :, 2] = b
    T3 = zeros(2, d, 1)
    T3[1, :, 1] = a
    T3[2, :, 1] = b
    return TCI.TensorTrain([T1, T2, T3])
end

@testset "elementwise weighting changes pivots" begin
    # f = e₁⊗e₁⊗e₁ + 10 e₂⊗e₂⊗e₂, so f.*f peaks at all-twos (1e6) vs all-ones (1).
    # Rank-1 truncation: unweighted keeps the large component; weight on all-ones only keeps that.
    a = [1.0, 0.0]
    b = [0.0, 10.0]
    F = sum_rank1(a, b)
    guess = ACI.randomtt(Float64, fill([2], 3), 2)

    trunc = ACI.TruncationParameters(1, 0.0, false)

    tt_unweighted, = ACI.elementwise(
        *, [F, F];
        initial_guess=deepcopy(guess),
        truncationparameters=trunc,
        max_iters=4,
        min_iters=2,
        weighting=nothing,
    )

    w(σ) = all(==(1), σ) ? 1.0 : 0.0
    tt_weighted, = ACI.elementwise(
        *, [F, F];
        initial_guess=deepcopy(guess),
        truncationparameters=trunc,
        max_iters=4,
        min_iters=2,
        weighting=w,
    )

    exact(σ) = F(σ)^2

    @test tt_unweighted([2, 2, 2]) != tt_weighted([2, 2, 2]) ||
          tt_unweighted([1, 1, 1]) != tt_weighted([1, 1, 1])

    @test abs(tt_weighted([1, 1, 1]) - exact([1, 1, 1])) <=
          abs(tt_unweighted([1, 1, 1]) - exact([1, 1, 1])) + 1e-10

    @test abs(tt_weighted([2, 2, 2])) < abs(tt_unweighted([2, 2, 2]))
end

@testset "zero Pi fallback updates weighting pivot tracker" begin
    R = 3
    sitedims = fill([2], R)
    Z = TCI.TensorTrain([zeros(Float64, 1, d..., 1) for d in sitedims])
    problem = ACI.ElementwiseProblem([Z], ACI.randomtt(Float64, sitedims, 2))
    errorweighting = ACI.ErrorWeighting(problem, _ -> 1.0, true)
    trunc = ACI.TruncationParameters(10, 1e-12, true)

    ACI.localupdate!(
        identity,
        problem,
        1;
        leftorthogonal=true,
        truncationparameters=trunc,
        errorweighting=errorweighting,
        environment_mode=false,
        normalize_environment=false,
    )

    @test errorweighting.pt.Iset[2] == [errorweighting.pt.Icombined_act[1]]
    @test errorweighting.pt.Jset[1] == [errorweighting.pt.Jcombined_act[1]]

    ACI.localupdate!(
        identity,
        problem,
        2;
        leftorthogonal=true,
        truncationparameters=trunc,
        errorweighting=errorweighting,
        environment_mode=false,
        normalize_environment=false,
    )

    @test errorweighting.pt.Iset[3] == [errorweighting.pt.Icombined_act[1]]
    @test errorweighting.pt.Jset[2] == [errorweighting.pt.Jcombined_act[1]]
end
