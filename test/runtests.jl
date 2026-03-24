using AlternatingCrossInterpolation
using Test
using Aqua
using JET

@testset "AlternatingCrossInterpolation.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(AlternatingCrossInterpolation)
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(AlternatingCrossInterpolation; target_defined_modules = true)
    end
    # Write your tests here.
end
