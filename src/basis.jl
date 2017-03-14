######################################################################
# general interface
######################################################################

export
    # general interface
    AbstractBasis,
    domain,
    collocation_points,
    degf,
    basis_matrix,
    InterpolatedFunction,
    collocation_values,
    # bases
    ChebyshevBasis,
    IntervalAB,
    LinearInterpolation,
    ExtrapolateLevel

abstract AbstractBasis

"""
Return the domain of the basis.
"""
function domain end

"A range of `n` linearly spaced elements in the domain of `basis`."
function linspace(basis::AbstractBasis, len)
    dom = domain(basis)
    linspace(dom.lo, dom.hi, len)
end

"""
Return the collocation points (if any) for a basis.
"""
function collocation_points end

"""
Degress of freedom (number of coefficients).
"""
degf(basis::AbstractBasis) = length(collocation_points(basis))

"""
Return the basis matrix, ie the functions in the basis evaluated at
the given points, one row for each point. Points default to
`collocation_points`.
"""
function basis_matrix end

"""
Interpolated function using `basis`, with coefficients `α`.
"""
immutable InterpolatedFunction{S, T}
    basis::S
    α::T
end

*(basis::AbstractBasis, α) = InterpolatedFunction(basis, α)

@forward InterpolatedFunction.basis domain, collocation_points, degf

function show(io::IO, ipf::InterpolatedFunction)
    print(io, "Interpolated function on ", ipf.basis)
end

"""
Evaluate the approximation on `basis` with coefficients `α` at at `x`.

Notes:
1. methods should be defined for various bases.
2. the callable interface is recommended for use outside this module.
"""
function _evaluate(basis, α, x)
    first(basis_matrix(basis, [x]) * α)
end

(f::InterpolatedFunction)(x) = _evaluate(f.basis, f.α, x)

"""
Evaluate the interpolated function at the collocation points.
"""
function collocation_values(ipf::InterpolatedFunction)
    ipf.(collocation_points(ipf.basis))
end

"""
Interpolate points in `y`, assumed to be values at the collocation
points, using `basis`.
"""
function \(basis::AbstractBasis, y::AbstractVector)
    # NOTE: this is just a fallback method, not necessarily efficient.
    @assert degf(basis) == length(y)
    InterpolatedFunction(basis, basis_matrix(basis) \ y)
end

"""
Interpolate `f`, evaluated at the collocation points, using `basis`.
"""
function \(basis::AbstractBasis, f)
    basis \ f.(collocation_points(basis))
end

zeros(basis::AbstractBasis) = basis \ zeros(degf(basis))

ones(basis::AbstractBasis) = basis \ ones(degf(basis))

# plot recipe for displaying interpolated functions
@recipe function f(ipf::InterpolatedFunction, label = "function", N = 100)
    @unpack basis, α = ipf
    dom = domain(basis)
    z = collect(linspace(dom.lo, dom.hi, N))
    @series begin
        seriestype := :path
        label := label
        z, ipf.(z)
    end
    @series begin
        primary := false
        seriestype := :scatter
        collocation_points(ipf), collocation_values(ipf)
    end
end

######################################################################
# Chebyshev basis on [-1,1]
######################################################################

immutable ChebyshevBasis <: AbstractBasis
    n::Int
    stretch::Bool
    extrapolate::Bool
    function ChebyshevBasis(n; stretch=false, extrapolate=false)
        new(n, stretch, extrapolate)
    end
end

size(b::ChebyshevBasis) = (b.n,)

domain(b::ChebyshevBasis, T=Float64) = -one(T)..one(T)

function collocation_points(b::ChebyshevBasis)
    zs = [cos((i-0.5)*pi/b.n) for i in b.n:-1:1]
    if b.stretch
        scale = 1/zs[end]
        zs .= zs .* scale
    end
    zs
end

function basis_matrix{T}(b::ChebyshevBasis,
                         x::AbstractVector{T}=collocation_points(b))
    if !b.extrapolate
        @assert all(abs(x) ≤ one(T) + √eps(T) for x in x)
    end
    A = Array{T}(length(x), b.n)
    for i in 1:b.n
        A[:,i] .= i == 1 ? one(T) : i == 2 ? x : (2.*x.*A[:, i-1]-A[:, i-2])
    end
    A
end

######################################################################
# IntervalAB
######################################################################

"""
Transform an inner basis to an interval.

Transforming to the inner basis is with `x ↦ fma(scale, shift)`.
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

_map_to_inner(b::IntervalAB, x) = fma.(x, b.scale, b.shift)

_map_from_inner(b::IntervalAB, x) = fma.(x, 1/b.scale, -b.shift/b.scale)

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

######################################################################
# linear interpolation
######################################################################

immutable LinearInterpolation{T <: AbstractVector} <: AbstractBasis
    nodes::T
    function LinearInterpolation(nodes)
        @assert issorted(nodes) "Nodes need to be sorted"
        new(nodes)
    end
end

LinearInterpolation{T}(nodes::T) = LinearInterpolation{T}(nodes)

collocation_points(li::LinearInterpolation) = li.nodes

domain(li::LinearInterpolation) = li.nodes[1]..li.nodes[end]

basis_matrix(li::LinearInterpolation) = speye(length(li.nodes))

function \(li::LinearInterpolation, y::AbstractVector)
    @assert degf(li) == length(y)
    InterpolatedFunction(li, y)
end

function _evaluate(li::LinearInterpolation, α, x)
    @unpack nodes = li
    (nodes[1] ≤ x ≤ nodes[end]) || error(DomainError())
    index = searchsortedlast(nodes, x)
    @assert index ≠ 0 "internal error"
    # println("index = $index, node = $(nodes[index])")
    if x == nodes[index]
        α[index]
    else
        p = (x - nodes[index]) / (nodes[index+1] - nodes[index])
        α[index]*(1-p) + α[index+1]*p
    end
end

function collocation_values{S <: LinearInterpolation, T}(ipf::InterpolatedFunction{S,T})
    ipf.α
end

######################################################################
# ExtrapolateLevel
######################################################################

immutable ExtrapolateLevel{T} <: AbstractBasis
    inner_basis::T
end

@forward ExtrapolateLevel.inner_basis domain, collocation_points, basis_matrix, degf

function _evaluate(basis::ExtrapolateLevel, α, x)
    @unpack inner_basis = basis
    dom = domain(inner_basis)
    _evaluate(inner_basis, α, clamp(x, dom.lo, dom.hi))
end
