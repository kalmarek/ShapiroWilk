using Arblib
using Statistics

using LinearAlgebra

@testset "Shapiro-Wilk coefficients" begin
    @test ShapiroWilk.SWCoeffs(10) isa AbstractVector{<:Float64}

    OS = ShapiroWilk.NormOrderStatistic(10, prec = 64, radius=18.0)
    @test cov(OS) isa Arblib.ArbMatrix
    @test ShapiroWilk.SWCoeffs(OS) isa AbstractVector{<:Arblib.Arb}

    sw_10 = ShapiroWilk.SWCoeffs(OS)

    let approx_error = abs.(sw_10 .- ShapiroWilk.SWCoeffs(10))
        @test maximum(approx_error) < 5e-4
        @test minimum(approx_error) > 8e-5
    end

    @test Arblib.contains_zero(dot(sw_10, sw_10) - 1)
    @test Arblib.contains_zero(sum(sw_10'*cov(OS)))

    sw_20 = let N = 20, prec = 96
        OS = ShapiroWilk.NormOrderStatistic(N, prec = prec, radius=18.0)

        ShapiroWilk._precompute(OS)

        sw = ShapiroWilk.SWCoeffs(OS)

        @test Arblib.contains_zero(sum(a*a for a in sw) - 1)
        @test Arblib.contains_zero(sum(sw'*cov(OS)))

        approx_error = abs.(sw .- ShapiroWilk.SWCoeffs(N))
        @test maximum(approx_error) < 9e-4
        @test minimum(approx_error) > 3e-5
        sw
    end
end
