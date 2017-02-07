module DOIHS

using ValidatedNumerics
import Base: show, size, linspace

export
    # basis
    AbstractBasis,
    domain,
    collocation_points,
    basis_matrix,
    ChebyshevBasis,
    IntervalAB

include("basis.jl")

end # module
