using ShapiroWilk
using Distributions
using Arblib
using JLD

import ShapiroWilk.NormOrderStatistic

function precision_sum_moments(OS::NormOrderStatistic)

    ShapiroWilk._precompute(OS)
    tests_pass = true

    expected_tol = maximum(1:OS.n-1) do i
        res = sum(Distributions.expectation(OS, i, j) for j = 1:OS.n) - 1
        if !Arblib.contains_zero(res)
            # @warn "sum of expected values is not guaranteed to contain 1: $(res + 1)"
            tests_pass = false
        end
        abs(res)
    end

    moment_tol = let OS = OS
        res = sum(Distributions.moment(OS, i, pow = 2) for i = 1:OS.n) - OS.n
        if !Arblib.contains_zero(res)
            # @warn "sum of second moments does not contain $(OS.n): $(res+OS.n)"
            tests_pass = false
        end
        abs(res)
    end

    return tests_pass, max(expected_tol, moment_tol)
end

n = 5
prec = 64
results = Vector{Float64}[]
R = 16.0

isdir("log") || mkdir("log")
files_computed = readdir("log")

while n <= 30
    re = Regex("sw_coeffs_$(n)_(?<prec>\\d+)\\.jld")
    if any(f -> match(re, f) !== nothing, files_computed)
        @info "Found precomputed values for n=$n, skipping"
        global n +=1
        continue
    else
        re_prev = Regex("sw_coeffs_$(n-1)_(?<prec>\\d+)\\.jld")
        for f in reverse(files_computed)
            m = match(re_prev, f)
            if m !== nothing
                global prec = max(prec, parse(Int, m[:prec]))
                break
            end
        end
        computed = false
        swfl = Vector{Float64}(undef, n)
    end

    while !computed
        os = NormOrderStatistic(n, prec = prec, radius=R)
        @info "computing SW coefficients for n=$n, prec=$prec, radius=$R"
        @time test_pass, tol = precision_sum_moments(os)
        while !test_pass
            global R *= 2
            @warn "Moments tests fail, doubling the radius..."
            os = NormOrderStatistic(n, prec = prec, radius=R)
            @time test_pass, tol = precision_sum_moments(os)
        end
        @info "Accuracy for moments: $tol"

        sw = ShapiroWilk.SWCoeffs(os)
        if all(x->Arblib.radref(x) < eps(Float64), sw)
            swfl .= Float64.(sw)
            computed = true
        else
            @warn "SWCoefficients lost accuracy, dubling the precision..."
            global prec *= 2
        end
    end

    push!(results, swfl)
    save("log/sw_coeffs_$(n)_$(prec).jld", "sw", swfl)
    global n +=1
end
