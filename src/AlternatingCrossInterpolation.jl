module AlternatingCrossInterpolation

import TensorCrossInterpolation as TCI
import TensorCrossInterpolation: TensorTrain
import OffsetArrays: Origin, OffsetMatrix
import Base: iterate

include("util.jl")
include("sweep.jl")
include("randomtt.jl")
include("maxvol.jl")
include("contract.jl")
include("frame.jl")
include("elementwise.jl")

end
