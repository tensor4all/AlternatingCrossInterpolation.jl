# AlternatingCrossInterpolation

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://tensor4all.github.io/AlternatingCrossInterpolation.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://tensor4all.github.io/AlternatingCrossInterpolation.jl/dev/)
[![Build Status](https://github.com/tensor4all/AlternatingCrossInterpolation.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/tensor4all/AlternatingCrossInterpolation.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

This package implements the _alternating cross interpolation_ method for efficient elementwise operations on tensor trains, such as Hadamard products.

## Installation

This package is not yet registered in the General julia registry. The easiest way to install it is to type the following in a julia REPL:
```julia
import Pkg; Pkg.add(url="https://github.com/tensor4all/AlternatingCrossInterpolation.jl.git")
```

## Usage

This package contains a single exported function, `elementwise()`, with two positional arguments: An operator to be applied to the tensor trains, and a vector containing tensor trains as `TensorCrossInterpolation.TensorTrain` objects. There are multiple keyword arguments, the most important of which is `truncationparameters`.

For example, to multiply two tensor trains `tt1` and `tt2` with a maximum bond dimension of 100, and a tolerance of ``\tau = 10^{-8}``:
```julia
truncationparameters = ACI.TruncationParameters(100, 1e-8)
ttprod, ranks, errors = ACI.elementwise(*, [tt1, tt2]; truncationparameters=truncationparameters)
```
The return values are
 - `ttprod`, the resulting tensor train, and
 - `ranks` and `errors`, the rank and error estimate in each iteration of the algorithm.

## Example
This example obtains TTs representing two Gaussians, and multiplies them with a maximum bond dimension of 15.
```julia
makegaussian(μ, σ) = x -> exp(-0.5 * ((x - μ) / σ)^2)
g1 = makegaussian(0.4, 0.15)
g2 = makegaussian(-0.4, 0.15)
grid = QG.DiscretizedGrid(R, -0.5, 0.5)
ttg1, = TCI.crossinterpolate2(Float64, q -> g1(QG.quantics_to_origcoord(grid, q)), QG.localdimensions(grid); tolerance=1e-24, maxbonddim=χ)
ttg2, = TCI.crossinterpolate2(Float64, q -> g2(QG.quantics_to_origcoord(grid, q)), QG.localdimensions(grid); tolerance=1e-24, maxbonddim=χ)
truncationparameters = ACI.TruncationParameters(15, 1e-24, false)
ttprod, = ACI.elementwise(*, TCI.tensortrain.([ttg1, ttg2]); truncationparameters)
```
More examples can be found in the repository https://github.com/rittermarc/aci-paper.
