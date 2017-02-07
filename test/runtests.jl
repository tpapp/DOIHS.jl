using DOIHS
using ValidatedNumerics
using Base.Test

function approxerror(basis, f, n=100)
    α = basis_matrix(basis) \ f(collocation_points(basis))
    x = linspace(basis, n)
    y = basis_matrix(basis, x) * α
    maximum(abs(y-f(x)) for (x,y) in zip(x,y))
end

@test approxerror(ChebyshevBasis(10), exp) ≤ 1e-9

@test approxerror(IntervalAB(0..pi, ChebyshevBasis(10)), sin) ≤ 1e-7
