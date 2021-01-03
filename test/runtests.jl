using Test
using ShapiroWilk
using Statistics

using Nemo
using Arblib

using ShapiroWilk.OrderStatisticsNemo

include("normordstats_nemo.jl")
let CC = Nemo.AcbField(96)
    numeric_tests_order_statistics_nemo(10, CC, atol=1e-15, R=18.0)
end

let CC = Nemo.AcbField(128)
    OS = OrderStatisticsNemo.NormOrderStatistic(10, CC)
    @time test_sum_moments_nemo(OS, atol=3e-32, R=18.0)

    OS = OrderStatisticsNemo.NormOrderStatistic(20, CC)
    @time test_sum_moments_nemo(OS, atol=8e-27, R=18.0)
end

using ShapiroWilk.OrderStatisticsArblib
include("normordstats_arblib.jl")

let
    import Memoize
    empty!(Memoize.memoize_cache(ShapiroWilk.OrderStatisticsArblib.α))
    empty!(Memoize.memoize_cache(ShapiroWilk.OrderStatisticsArblib._γ))
    empty!(Memoize.memoize_cache(ShapiroWilk.OrderStatisticsArblib.β))
    empty!(Memoize.memoize_cache(ShapiroWilk.OrderStatisticsArblib.ψ))

    numeric_tests_order_statistics_arblib(10, prec=96, atol=1e-15, R=18.0)
end

@time test_α_ij(50, 128, 4e-36, 18.0)
@time test_β_ii(50, 128, 2e-37, 18.0)
@time test_β_ij(50, 128, 5e-38, 18.0)

let
    OS = OrderStatisticsArblib.NormOrderStatistic(10, prec=128)
    @time test_sum_moments_arblib(OS, atol=3e-32, R=18.0)

    OS = OrderStatisticsArblib.NormOrderStatistic(20, prec=128)
    @time test_sum_moments_arblib(OS, atol=7e-27, R=18.0)

    OS = OrderStatisticsArblib.NormOrderStatistic(30, prec=128)
    @time test_sum_moments_arblib(OS, atol=2e-19, R=18.0)

    # OS = OrderStatisticsArblib.NormOrderStatistic(40, prec=128)
    # @time test_sum_moments_arblib(OS, atol=4e-12, R=18.0)

    # OS = OrderStatisticsArblib.NormOrderStatistic(50, prec=128)
    # @time test_sum_moments_arblib(OS, atol=6e-6, R=18.0)
end

include("swcoeffs.jl")
