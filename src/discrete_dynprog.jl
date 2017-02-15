# discrete state, discrete time dynamic programming

"""
Discrete state, discrete time dynamic programing problem with `N`
states and `M` actions.

`β` is the discount factor (infinite horizon).

`U` is an (N,M) matrix that describes the period payoffs for state `N`
and action `M`. For actions that are now allowed, use `-Inf`.

`P` is an `N`-element vector of `(M,N)` transition matrices, ie
`P[n][m,n′]` is the probability of state `n′` given state `n` and
action `m`.
"""
immutable DDproblem{Tβ <: Real, TU <: AbstractMatrix, TP <: AbstractMatrix}
    β::Tβ
    U::TU
    P::Vector{TP}
    function DDproblem(β, U, P)
        (N, M) = size(U)
        @assert length(P) == N "wrong number of elements in P"
        @assert all(size(p) == (M,N) for p in P) "non-conformable matrix in P"
        new(β, U, P)
    end
end

DDproblem{Tβ, TU, TP}(β::Tβ, U::TU, P::Vector{TP}) = DDproblem{Tβ, TU, TP}(β, U, P)

function show(io::IO, dd::DDproblem)
    (N,M) = size(dd.U)
    println(io, "Discrete state, discrete time dynamic programming problem")
    println(io, "  with $(N) states and $(M) actions, discount factor $(dd.β)")
end

"""
Value iteration step. Given value function `V`, return the new value
function and the indexes of optimal policies.
"""
function value_iteration_step(dd::DDproblem, V)
    @unpack β, U, P = dd
    optimal_Vms = [findmax(@view(U[n, :]) + β*P[n]*V) for n in indices(U, 2)]
    (first.(optimal_Vms), last.(optimal_Vms))
end

"""
Return a transition matrix between states given the optimal policies.

It will be the same type as the elements of `dd.P`, tus 
"""
function transition_matrix(dd::DDproblem, optimal_ms)
    @unpack U, P = dd
    N = size(U, 2)
    M = zeros(N, N)
    for (n,m) in enumerate(optimal_ms)
        M[n, :] = P[n][m, :]
    end
    convert(eltype(P), M)
end

"""
Policy iteration step. Given a value function `V`, return the new
value function and the indexes of optimal policies.
"""
function policy_iteration_step(dd::DDproblem, V)
    optimal_ms = value_iteration_step(dd, V)[2]
    u = [dd.U[n,m] for (n,m) in enumerate(optimal_ms)]
    M = transition_matrix(dd, optimal_ms)
    V = (I - dd.β * M) \ u
    (V, optimal_ms)
end

"""
Structure for storing the result of iterative solution techniques.

`V` is the value function.
`ms` is the index of actions (only optimal when converged).
`iterations` is the number of iterations performed.
`converged` is a boolean which is `true` when convergence is achieved.
"""
immutable DDsolution
    V
    ms
    iterations
    converged
end

function show(io::IO, dd::DDsolution)
    print(io, "DDsolution ")
    if dd.converged
        print_with_color(:green, io, "converged")
    else
        print_with_color(:red, io, "did not converge")
    end
    println(io, " after $(dd.iterations) iterations")
end

"""
Solve a problem using value and policy iteration, using initial values
`V`. Successive value functions are compared with the L¹ norm,
convergence is achieved the difference between value functions is
below `ϵ`.

For the first `value_iter` steps, use value iteration, and after that,
policy iteration.
"""
function solve_iteratively(dd::DDproblem;
                           V = zeros(size(dd.U, 1)),
                           value_iter = 20,
                           max_iter = 100,
                           ϵ = √eps(Float64))
    for iteration in 1:max_iter
        (V′, ms) = iteration ≤ value_iter ? value_iteration_step(dd, V) :
            policy_iteration_step(dd, V)
        if norm(V′-V, 1) ≤ ϵ
            return DDsolution(V′, ms, iteration, true)
        elseif iteration == max_iter
            return DDsolution(V′, ms, iteration, false)
        else
            V .= V′
        end
    end
end

"""
Given a transition matrix `M` between states, an initial state `n0`,
return a sequence of `T` simulated transitions.
"""
function simulate_transitions(M, n0, T)
    ps = [Categorical(collect(M[n, :])) for n in indices(M, 1)]
    n = n0
    [(n_prev = n; n = rand(ps[n]); n_prev) for i in 1:T]
end
