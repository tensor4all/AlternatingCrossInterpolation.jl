module AlternatingCrossInterpolation

import TensorCrossInterpolation as TCI
import TensorCrossInterpolation: TensorTrain
import OffsetArrays: Origin, OffsetMatrix
import Random
import Base: iterate

export elementwise
export TruncationParameters

include("util.jl")
include("sweep.jl")
include("randomtt.jl")
include("maxvol.jl")
include("contract.jl")
include("frame.jl")
include("elementwise.jl")

end
