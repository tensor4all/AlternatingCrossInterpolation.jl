using AlternatingCrossInterpolation
import AlternatingCrossInterpolation as ACI
using Test
using Aqua
using JET
import TensorCrossInterpolation as TCI
import QuanticsGrids as QG
using LinearAlgebra

@testset "AlternatingCrossInterpolation.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(AlternatingCrossInterpolation)
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(AlternatingCrossInterpolation; target_defined_modules = true)
    end

    include("test_frame.jl")
    include("test_elementwise.jl")
end
