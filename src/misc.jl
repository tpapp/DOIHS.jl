export crra_utility, crra_utility_function

"""
CRRA utility function for consumption `c`, with relative risk aversion
(or IES) `σ`. σ = 1 gives log utility.
"""
function crra_utility(c::AbstractFloat, σ)
    if σ > 1
        c < zero(c) ? throw(DomainError()) : (c^(1-σ)-one(c))/(1-σ)
    elseif σ == 1
        log(c)
    else
        throw(DomainError())
    end
end

# always convert to float for corner case c==0
crra_utility(c, σ) = crra_utility(convert(AbstractFloat, c), σ)

"""
Return a *function* that maps its argument (usually consumption) with
CRRA utility. σ = 1 gives log utility.
"""
function crra_utility_function(σ)
    if σ > 1
        c -> crra_utility(c, σ)
    elseif σ == 1
        log
    else
        throw(DomainError())
    end
end
