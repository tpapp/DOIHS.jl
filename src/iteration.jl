export IterationOptions

"""
Options for iterative solvers.
"""
@with_kw immutable IterationOptions
    max_iter::Int = 100                    # maximum number of iterations
    tol1::Float64 = √eps()                 # 1 norm for change
    tol∞::Float64 = √eps()                 # ∞ norm for change
end

"""
Check convergence.
"""
function _converged(Δ, options::IterationOptions)
    norm(Δ, 1) ≤ options.tol1 && norm(Δ, ∞) ≤ options.tol∞
end
