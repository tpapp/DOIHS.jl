# general interface

abstract AbstractBasis

"""
Return the domain of the basis.
"""
function domain end

function linspace(b::AbstractBasis, len)
    dom = domain(b)
    linspace(dom.lo, dom.hi, len)
end

"""
Return the collocation points (if any) for a basis.
"""
function collocation_points end

"""
Return the basis matrix, ie the functions in the basis evaluated at
the given points, one row for each point. Points default to
`collocation_points`.
"""
function basis_matrix end

# Chebyshev basis on [-1,1]

immutable ChebyshevBasis <: AbstractBasis
    n::Int
end

size(b::ChebyshevBasis) = (b.n,)

domain(b::ChebyshevBasis, T=Float64) = -one(T)..one(T)

function collocation_points(b::ChebyshevBasis, stretch = false)
    zs = [cos((i-0.5)*pi/b.n) for i in b.n:-1:1]
    if stretch
        scale = 1/zs[end]
        zs .= zs .* scale
    end
    zs
end

function basis_matrix{T}(b::ChebyshevBasis,
                         x::AbstractVector{T}=collocation_points(b))
    @assert all(abs(x) ≤ one(T) for x in x)
    A = Array{T}(length(x), b.n)
    for i in 1:b.n
        A[:,i] .= i == 1 ? one(T) : i == 2 ? x : (2.*x.*A[:, i-1]-A[:, i-2])
    end
    A
end

# IntervalAB

"""
Transform an inner basis to an interval.

Transforming to the inner basis is with `x ↦ muladd(scale, shift)`.
"""
immutable IntervalAB{T,S <: AbstractBasis} <: AbstractBasis
    domain::Interval{T}
    inner_basis::S
    scale::T
    shift::T
    function IntervalAB(dom, inner_basis)
        @assert isfinite(dom)
        inner_dom = domain(inner_basis)
        scale = diam(inner_dom)/diam(dom)
        new(dom, inner_basis, scale, mid(inner_dom) - scale*mid(dom))
    end
end

function IntervalAB{T,S}(domain::Interval{T}, inner_basis::S)
    IntervalAB{T,S}(domain, inner_basis)
end

_map_to_inner(b::IntervalAB, x) = muladd.(x, b.scale, b.shift)

_map_from_inner(b::IntervalAB, x) = muladd.(x, 1/b.scale, -b.shift/b.scale)

domain(b::IntervalAB) = b.domain

function collocation_points(b::IntervalAB)
    _map_from_inner(b, collocation_points(b.inner_basis))
end

basis_matrix(b::IntervalAB) = basis_matrix(b.inner_basis)

basis_matrix(b::IntervalAB, x) = basis_matrix(b.inner_basis, _map_to_inner(b, x))

function show(io::IO, b::IntervalAB)
    print(io, b.inner_basis)
    print(io, " on ")
    print(io, b.domain)
end
