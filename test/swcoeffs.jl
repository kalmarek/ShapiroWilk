using Nemo
using Arblib
using Statistics
using ShapiroWilk: OrderStatisticsNemo, OrderStatisticsArblib

using LinearAlgebra

function Arblib.set!(res::Arblib.Arb, x::Nemo.arb)
    ccall(Arblib.@libarb("arb_set"), Cvoid, (Ref{Arblib.arb_struct}, Ref{Nemo.arb}), res, x)
    return res
end

@testset "Shapiro-Wilk coefficients" begin
    @test ShapiroWilk.SWCoeffs(10) isa AbstractVector{<:Float64}

    CC = Nemo.AcbField(64)
    OS1 = OrderStatisticsNemo.NormOrderStatistic(10, CC)
    @test cov(OS1) isa Nemo.MatElem
    @test ShapiroWilk.SWCoeffs(OS1) isa AbstractVector{<:Nemo.arb}

    sw_10_nemo = ShapiroWilk.SWCoeffs(OS1)

    @test Nemo.contains_zero(sum(a*a for a in sw_10_nemo) - 1)
    @test Nemo.contains_zero(sum(sw_10_nemo'*cov(OS1)))

    let approx_error = abs.(sw_10_nemo .- ShapiroWilk.SWCoeffs(10))
        @test all(approx_error .< 5e-4)
        @test all(approx_error .> 8e-5)
    end

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

    @test maximum(abs.(Arb.(sw_10_nemo) .- sw_10_arblib)) < 7e-10

    sw_20_nemo = let N = 20, prec=96

        OS = OrderStatisticsNemo.NormOrderStatistic(20, Nemo.AcbField(prec))

        OrderStatisticsNemo._precompute(N, prec=prec, R=18.0)

        sw = ShapiroWilk.SWCoeffs(OS)

        @test Nemo.contains_zero(sum(a*a for a in sw) - 1)
        @test Nemo.contains_zero(sum(sw'*cov(OS)))

        approx_error = abs.(sw .- ShapiroWilk.SWCoeffs(N))
        @test all(approx_error .< 9e-4)
        @test all(approx_error .> 3e-5)
        sw
    end

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

    @test maximum(abs.(Arb.(sw_20_nemo) .- sw_20_arblib)) < 8e-12
end
