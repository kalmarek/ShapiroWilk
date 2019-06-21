using Test
using ShapiroWilk
using ShapiroWilk.OrderStatistics
# using OrderStatistics
using ShapiroWilk.OrderStatistics.Nemo



include("alpha_beta.jl")
include("moments.jl")

function numeric_tests_order_statistics(N::Int, CC::AcbField, tol=eps(Float64),
    R=OrderStatistics.RADIUS.R)

    @time OrderStatistics.precompute_ψ(N, CC)

    @testset "Relations between α, β and expectations/moments of OS" begin

        OS = NormOrderStatistic(N, CC)

        @time test_α_ij(OS, CC, tol, R)
        @time test_β_ii(OS, CC, tol, R)
        @time test_β_ij(OS, CC, tol, R)

        @time test_moments(OS, tol)
        @time test_moments(OS, tol)
    end
end

numeric_tests_order_statistics(10, Nemo.AcbField(100))
