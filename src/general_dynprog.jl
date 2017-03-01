export
    OptimalRHS,
    value,
    policy,
    optimize_rhs,
    DPSolution,
    value_iteration,
    value_residual,
    solve_residual,
    nonlinear_solve_value

"""
Object characterizing the optimal solution (a value and a policy) to
the right hand side of a Bellman equation at a given state.
"""
immutable OptimalRHS{TV,TP}
    value::TV
    policy::TP
end

"""
Accessor function for the value.
"""
value(o::OptimalRHS) = o.value

"""
Accessor function for the optimal policy.
"""
policy(o::OptimalRHS) = o.policy

"""
`optimize_rhs(model, value, state)` is the core function for solving
dynamic programming problems.

The user should implement this function for each type of `model` that
will be solved by the library. For a given value function `value` and
state `state`, it should return an OptimalRHS object, containing the
optimal value and the optimal policy.
"""
function optimize_rhs end

"""
Solution to a dynamic programming problem.
"""
@with_kw immutable DPSolution
    model
    value
    policy
    iterations
    converged
end

value(dp::DPSolution) = dp.value
policy(dp::DPSolution) = dp.policy

"""
Solve a dynamic programming problem by value iteration.
"""
function value_iteration(model, V, options = IterationOptions())
    @unpack max_iter = options
    basis = V.basis
    states = collocation_points(basis)
    for iteration in 1:max_iter
        VP = [optimize_rhs(model, V, s) for s in states]
        V′points = value.(VP)
        Δ = V′points - collocation_values(V)
        V′ = basis \ V′points
        converged = _converged(Δ, options)
        if converged || iteration == max_iter
            return DPSolution(model = model,
                              value = V′,
                              policy = basis \ policy.(VP),
                              iterations = iteration,
                              converged = converged)
        else
            V = V′
        end
    end
end

function value_iteration(solution::DPSolution,
                         options = IterationOptions())
    @unpack model, value = solution
    value_iteration(model, value, options)
end

"""
Calculate the residual between the right- and left-hand sides of a
Bellman equation at the given state.
"""
function value_residual(model, value, state)
    optimize_rhs(model, value, state).value - value(state)
end

function value_residual(solution::DPSolution, state)
    @unpack model, value = solution
    value_residual(model, value, state)
end

"""
Solve for `residual(f) = 0` in a linearly interpolated space of
functions.

The collocation points are determined from `initial_f`, which is also
used as a starting point.

`residual_function(model, f, s)` will be called with all `s`
collocation points, and the result is used as the residual. A
nonlinear solver is used to find `f`.

`f` and the result of `nlsolve` are returned as two values.
"""
function solve_residual(model,
                        residual_function,
                        initial_f,
                        nlsolve_options...)
    @unpack basis, α = initial_f
    states = collocation_points(basis)
    function f!(α, residual)
        f = basis * α
        residual .= [residual_function(model, f, state) for state in states]
    end
    result = nlsolve(f!, α, nlsolve_options...)
    basis * result.zero, result
end

function nonlinear_solve_value(solution::DPSolution)
    value, root = solve_residual(solution.model, value_residual, solution.value)
    policy(state) = optimize_rhs(solution.model, value, state).policy
    DPSolution(model = solution.model,
               value = value,
               policy = value.basis \ policy,
               iterations = root.iterations,
               converged = NLsolve.converged(root))
end
