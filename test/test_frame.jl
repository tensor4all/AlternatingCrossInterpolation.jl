import OffsetArrays: Origin
import Random

Random.seed!(1234)

@testset "Frame functions" for leftorthogonal in (true, false)
    tt = ACI.randomtt(Float64, 5, 2, 3)

    envsizes = Origin(0)([(1, 1), (2, 2), (3, 3), (3, 3), (2, 2), (1, 1)])
    
    env, indices = ACI.frame(tt; leftorthogonal)
    @test size.(env) == envsizes
    env2, indices = ACI.frame(tt; indices, leftorthogonal)
    @test env2 == env
end
