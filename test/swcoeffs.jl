using Arblib
using Statistics
using ShapiroWilk: OrderStatisticsNemo, OrderStatisticsArblib

using LinearAlgebra

@testset "Shapiro-Wilk coefficients" begin
    @test ShapiroWilk.SWCoeffs(10) isa AbstractVector{<:Float64}



    OS2 = OrderStatisticsArblib.NormOrderStatistic(10, prec = 64)
    @test cov(OS2) isa Arblib.ArbMatrix
    @test ShapiroWilk.SWCoeffs(OS2) isa AbstractVector{<:Arblib.Arb}

    sw_10_arblib = ShapiroWilk.SWCoeffs(OS2)

    let approx_error = abs.(sw_10_arblib .- ShapiroWilk.SWCoeffs(10))
        @test maximum(approx_error) < 5e-4
        @test minimum(approx_error) > 8e-5
    end

    @test Arblib.contains_zero(dot(sw_10_arblib, sw_10_arblib) - 1)
    @test Arblib.contains_zero(sum(sw_10_arblib'*cov(OS2)))



    sw_20_arblib = let N = 20, prec = 96
        OS = OrderStatisticsArblib.NormOrderStatistic(N, prec = prec)

        OrderStatisticsArblib._precompute(N, prec=prec, R=18.0)

        sw = ShapiroWilk.SWCoeffs(OS)

        @test Arblib.contains_zero(sum(a*a for a in sw) - 1)
        @test Arblib.contains_zero(sum(sw'*cov(OS)))

        approx_error = abs.(sw .- ShapiroWilk.SWCoeffs(N))
        @test maximum(approx_error) < 9e-4
        @test minimum(approx_error) > 3e-5
        sw
    end
end
