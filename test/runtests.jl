using Test
using ShapiroWilk
using Statistics
using Distributions

using Arblib

using ShapiroWilk.OrderStatisticsArblib
include("normordstats_arblib.jl")

let
    # import Memoize
    # empty!(Memoize.memoize_cache(ShapiroWilk.OrderStatisticsArblib.α))
    # empty!(Memoize.memoize_cache(ShapiroWilk.OrderStatisticsArblib._γ))
    # empty!(Memoize.memoize_cache(ShapiroWilk.OrderStatisticsArblib.β))
    # empty!(Memoize.memoize_cache(ShapiroWilk.OrderStatisticsArblib.ψ))

    numeric_tests_order_statistics_arblib(6, prec=69, atol=eps(Float64), R=18.0)

    OS = OrderStatisticsArblib.NormOrderStatistic(10, prec=96)
    test_sum_moments_arblib(OS, atol=2e-22, R=18.0)

    OS = OrderStatisticsArblib.NormOrderStatistic(20, prec=96)
    test_sum_moments_arblib(OS, atol=2e-16, R=18.0)
end

include("swcoeffs.jl")

if false

    # @time test_α_ij(50, 128, 4e-36, 18.0)
    # @time test_β_ii(50, 128, 2e-37, 18.0)
    # @time test_β_ij(50, 128, 5e-38, 18.0)

    OS = OrderStatisticsArblib.NormOrderStatistic(10, prec=128)
    test_sum_moments_arblib(OS, atol=3e-32, R=18.0)

    OS = OrderStatisticsArblib.NormOrderStatistic(20, prec=128)
    test_sum_moments_arblib(OS, atol=7e-27, R=18.0)

    OS = OrderStatisticsArblib.NormOrderStatistic(30, prec=128)
    test_sum_moments_arblib(OS, atol=2e-19, R=18.0)

    OS = OrderStatisticsArblib.NormOrderStatistic(40, prec=128)
    @time test_sum_moments_arblib(OS, atol=4e-12, R=18.0)

    OS = OrderStatisticsArblib.NormOrderStatistic(50, prec=128)
    @time test_sum_moments_arblib(OS, atol=6e-6, R=18.0)
end
