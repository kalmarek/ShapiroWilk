using Test
using ShapiroWilk
using Statistics

CC = Nemo.AcbField(96)
include("normordstats_nemo.jl")
numeric_tests_order_statistics_nemo(10, CC, atol=1e-15, R=18.0)





