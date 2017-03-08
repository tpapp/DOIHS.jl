export
    Quadrature,
    quadrature,
    quadrature_normal


"""
Lightweight wrapper for the nodes and weights of a quadrature. Also
include the domain.
"""
immutable Quadrature{T <: Real}
    domain::Interval{T}
    nodes::Vector{T}
    weights::Vector{T}
end

function Base.show(io::IO, q::Quadrature)
    print(io, "Quadrature of $(length(q.nodes)) points on $(q.domain)")
end

"""
Integrate a univariate function using the quadrature.
"""
function (q::Quadrature)(f)
    sum(weight*f(node) for (node, weight) in zip(q.nodes, q.weights))
end

@recipe function f{T <: Quadrature}(q::T)
    seriestype --> :scatter
    ylim --> (0, 1.05*maximum(q.weights))
    q.nodes, q.weights
end

"""
Return a quadrature rule for calculating an integral using the density
function of distribution `d`, truncated to `domain`.

Uses Legendre quadrature on a finite interval.
"""
function quadrature(n::Int, distribution::Distribution{Univariate,Continuous},
                    domain::Interval)
    @assert isfinite(domain)
    a, b = domain.lo, domain.hi
    y_nodes, weights = gausslegendre(n)
    x_nodes = fma.(y_nodes, (b-a)/2, middle(a,b))
    Quadrature(domain, x_nodes, # FIXME: remove [] around distribution in 0.6
               normalize!(pdf.([distribution], x_nodes) .* weights, 1))
end

function quadrature{T <: Distribution{Univariate,Continuous}}(n::Int, d::Truncated{T,Continuous})
    quadrature(n, d.untruncated, d.lower..d.upper)
end

"""
Return a quadrature that integrates a function under the `Normal(μ,σ)`
distribution using `N` nodes.
"""
function quadrature_normal(N::Int, μ, σ)
    nodes, weights = gausshermite(N)
    Quadrature(-Inf..Inf, nodes*√2*σ + μ, weights/√π)
end
