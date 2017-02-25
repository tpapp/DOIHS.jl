module DOIHS

using ValidatedNumerics
using FastGaussQuadrature
using Distributions
using RecipesBase
using Parameters
using Distributions
using Lazy

import Base: show, size, linspace, \, zeros, ones

include("basis.jl")
include("quadrature.jl")
include("discrete_dynprog.jl")

end # module
