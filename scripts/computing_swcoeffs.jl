using ShapiroWilk
using Distributions
using Arblib
using JLD

import ShapiroWilk.OrderStatisticsArblib
import ShapiroWilk.OrderStatisticsArblib.NormOrderStatistic

function precision_sum_moments(OS::NormOrderStatistic; R)

    OrderStatisticsArblib._precompute_ψ(OS.n, prec=precision(OS), R=R)

    expected_tol = maximum(1:OS.n-1) do i
        res = sum(Distributions.expectation(OS, i, j, radius=R) for j = 1:OS.n) - 1
        Arblib.contains_zero(res) || @warn "sum of expected values is not guaranteed to contain 1: $(res + 1)"
        abs(res)
    end

    moment_tol = let OS = OS, R=R
        res = sum(Distributions.moment(OS, i, pow = 2, radius=R) for i = 1:OS.n) - OS.n
        Arblib.contains_zero(res) || @warn "sum of second moments does not contain $(OS.n): $(res+OS.n)"
        abs(res)
    end

    return max(expected_tol, moment_tol)
end

n = 5
prec = 64
results = Vector{Float64}[]
R = 18.0

isdir("log") || mkdir("log")

while n <= 60
    # @info "computing SW coefficients for n=$n"
    computed = false
    swfl = Vector{Float64}(undef, n)

    while !computed
        os = NormOrderStatistic(n, prec = prec)
        @info "n=$n, prec=$prec"
        @time tol = precision_sum_moments(os, R=R)
        @info "Accuracy for moments: $tol"
        sw = ShapiroWilk.SWCoeffs(os)
        if all(x->Arblib.radref(x) < eps(Float64), sw)
            swfl .= Float64.(sw)
            computed = true
        else
            global prec += 32
        end
    end

    push!(results, swfl)
    save("log/sw_coeffs_$(n)_$(prec).jld", "sw", swfl)
    global n +=1
end


# precision_sum_moments(NormOrderStatistic(20, prec=96), R=18.0)
# precision_sum_moments(NormOrderStatistic(30, prec=128), R=18.0)
# precision_sum_moments(NormOrderStatistic(40, prec=160), R=18.0)
# precision_sum_moments(NormOrderStatistic(50, prec=160), R=18.0)
# precision_sum_moments(NormOrderStatistic(60, prec=192), R=18.0)