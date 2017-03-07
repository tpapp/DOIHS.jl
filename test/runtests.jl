using DOIHS
using ValidatedNumerics
using Distributions
using Base.Test

@testset "CRRA utility" begin

    @test_throws DomainError crra_utility_function(0.1)
    @test_throws DomainError crra_utility(0.1, 0.1)
    @test_throws DomainError crra_utility(-1, 1)
    @test crra_utility(0, 1) ≡ -∞

    u = crra_utility_function(1)
    @test_throws DomainError u(-10)
    @test u(0) ≡ -∞
    @test u(1) == 0.0
    @test isa(u(1), Float64)

    u = crra_utility_function(2)
    @test_throws DomainError u(-10)
    @test u(0.0) ≡ -∞
    @test u(1) == 0.0
    @test isa(u(1), Float64)

    @test_throws DomainError crra_u′(-0.1, 0.1)
    @test crra_u′(2, 1) == 1/2
end

@testset "function approximation" begin

    function approxerror(basis, f, n=100)
        α = basis_matrix(basis) \ f(collocation_points(basis))
        x = linspace(basis, n)
        y = basis_matrix(basis, x) * α
        maximum(abs(y-f(x)) for (x,y) in zip(x,y))
    end

    @test approxerror(ChebyshevBasis(10), exp) ≤ 1e-9

    @test approxerror(IntervalAB(0..pi, ChebyshevBasis(10)), sin) ≤ 1e-7

end

@testset "linear interpolation" begin
    @test_throws Exception LinearInterpolation([2.0,1.0])
    li = LinearInterpolation(Float64[1,2,4])
    @test collocation_points(li) == [1.0,2.0,4.0]
    @test domain(li) == 1.0..4.0
    @test degf(li) == 3
    y = [5.0, 3.0, 7.0]
    lif = li \ y
    @test collocation_values(lif) == y
    @test lif(1.0) == 5.0
    @test lif(1.5) == 4.0
    @test lif(1.75) == 3.5
    @test lif(3.0) == 5.0
    @test lif(3.5) == 6.0
    @test lif(3.75) == 6.5
    @test lif(4.0) == 7.0
    @test_throws Exception lif(-1.0)
    @test_throws Exception lif(5.0)
end

@testset "extrapolate level" begin
    inner_basis = IntervalAB(1..10, ChebyshevBasis(3))
    basis = ExtrapolateLevel(inner_basis)
    @test collocation_points(inner_basis) == collocation_points(basis)
    @test degf(inner_basis) == degf(basis)
    y = [2.0, 3.0, 7.0]
    ipf = basis \ y
    @test collocation_values(ipf) ≈ y
    a = ipf(1.0)
    b = ipf(10.0)
    @test ipf(-1.0) ≈ a
    @test ipf(11.0) ≈ b
end

@testset "quadrature" begin

    d = Truncated(Normal(0,1), -1, 2)
    q = quadrature(10, d)
    m = mean(d)
    @test isapprox(q(identity), m; rtol=1e-6)
    @test isapprox(q(x->(x-m)^2), var(d); rtol=1e-6)

end
