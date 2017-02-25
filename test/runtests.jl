using DOIHS
using ValidatedNumerics
using Distributions
using Base.Test

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

@testset "quadrature" begin

    d = Truncated(Normal(0,1), -1, 2)
    q = quadrature(10, d)
    m = mean(d)
    @test isapprox(q(identity), m; rtol=1e-6)
    @test isapprox(q(x->(x-m)^2), var(d); rtol=1e-6)

end
