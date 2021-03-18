using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using ShapiroWilk
using Distributions
using Statistics
using LinearAlgebra
using Random
using BoxCoxTrans
using LsqFit

using JLD

import ShapiroWilk.NormOrderStatistic

function read_sw_coeffs(n::Integer)::ShapiroWilk.SWCoeffs{Float64}
    fname_re = Regex("log/sw_coeffs_(?<n>$n)_(?<prec>\\d+).jld")
    m = nothing
    for fn in readdir(joinpath(@__DIR__, "..", "log"), join=true, sort=true)
        m = match(fname_re, fn)
        !isnothing(m) && break
    end

    @assert !isnothing(m) "Failed parsing $fname_re"
    @debug "Found procomputed SW-coefficients with precision=$(m[:prec])"
    sw = load(m.match, "sw")::Vector{Float64}
    return ShapiroWilk.SWCoeffs(n, sw)
end

const sw_coeffs = Dict{Int, SWCoeffs{Float64}}(n=>read_sw_coeffs(n) for n in 5:189)

for n = 4:4
    os = ShapiroWilk.NormOrderStatistic(n, prec=96, radius=16.0)
    sw = ShapiroWilk.SWCoeffs(os)
    swfl = ShapiroWilk.SWCoeffs(n, Float64.(sw))
    sw_coeffs[n] = swfl
end


function sample_Wstatistic(sw::ShapiroWilk.SWCoeffs{T}, sample_size::Integer) where T
    Wstats = zeros(T, sample_size)
    tmp = similar(Wstats, length(sw))
    for i in 1:sample_size
        randn!(tmp)
        sort!(tmp)
        Wstats[i] = ShapiroWilk.Wstatistic(tmp, sw)
    end
    return Wstats
end

let qs= 0.0005:0.001:0.9995, n_range = 4:11
    cdfs = map(n_range) do n
        W = sample_Wstatistic(sw_coeffs[n], 10_000_000)
        quantile(W, qs)
    end

    writedlm(joinpath("src", "W_cdf_$(n_range).csv"), cdfs)
end

#=
# checking values at a given level

cdfs = map(4:11) do n
    qs= 0.0005:0.001:0.9995
    W = sample_Wstatistic(sw_coeffs[n], 10_000_000)
    (n => CumulativeDistribution(quantile(W, qs), qs))
    # cdf = quantile(W, qs)
    # plot!(cdf, qs, label="$n")
end |> Dict

let n = 4, cdf = cdfs[n], level=0.1
    W = sample_Wstatistic(sw_coeffs[n], 100_000)
    c = map(W) do w
        cdf(w) <= level
    end
    sum(c)/length(c)
end
=#
