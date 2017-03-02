module DOIHS

using ValidatedNumerics
using FastGaussQuadrature
using Distributions
using RecipesBase
using Parameters
using Distributions
using Lazy
using NLsolve

import Base: show, size, linspace, \, *, zeros, ones

include("misc.jl")
include("basis.jl")
include("quadrature.jl")
include("iteration.jl")
include("discrete_dynprog.jl")
include("general_dynprog.jl")

end # module
