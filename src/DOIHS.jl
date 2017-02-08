module DOIHS

using ValidatedNumerics
using FastGaussQuadrature
using Distributions
using RecipesBase

import Base: show, size, linspace

export
    # basis
    AbstractBasis,
    domain,
    collocation_points,
    basis_matrix,
    ChebyshevBasis,
    IntervalAB,
    # quadrature
    Quadrature,
    quadrature

include("basis.jl")
include("quadrature.jl")

end # module
