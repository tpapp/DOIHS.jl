module DOIHS

using ValidatedNumerics
using FastGaussQuadrature
using Distributions
using RecipesBase
using Parameters
using Distributions

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
    quadrature,
    # discrete_dynprog
    DDproblem,
    value_iteration_step,
    transition_matrix,
    policy_iteration_step,
    DDsolution,
    solve_iteratively,
    simulate_transitions

include("basis.jl")
include("quadrature.jl")
include("discrete_dynprog.jl")

end # module
