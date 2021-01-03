using Test
using ShapiroWilk
using Statistics

CC = Nemo.AcbField(96)
include("normordstats_nemo.jl")
numeric_tests_order_statistics_nemo(10, CC, atol=1e-15, R=18.0)

let
    empty!(Memoize.memoize_cache(ShapiroWilk.OrderStatisticsArblib.α))
    empty!(Memoize.memoize_cache(ShapiroWilk.OrderStatisticsArblib._γ))
    empty!(Memoize.memoize_cache(ShapiroWilk.OrderStatisticsArblib.β))
    empty!(Memoize.memoize_cache(ShapiroWilk.OrderStatisticsArblib.ψ))

    include("normordstats_arblib.jl")
    numeric_tests_order_statistics_arblib(10, prec=96, atol=1e-15, R=18.0)
end

test_sum_moments(10, prec=128, atol=3e-32, R=18.0)
test_sum_moments(20, prec=128, atol=7e-27, R=18.0)



include("swcoeffs.jl")
