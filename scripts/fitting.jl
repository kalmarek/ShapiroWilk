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

include(joinpath(@__DIR__, "plotting.jl"))

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

for n = 3:4
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

function BoxCox_λµσ(n, sampleW)
    W = one(eltype(sampleW)) .- sampleW
    λ, _ = BoxCoxTrans.lambda(W)
    W .= BoxCoxTrans.transform.(W, λ)
    µ, σ = mean(W), std(W)
    return (λ, µ, σ)
end

λµσ = let
    if isfile("log/λµσ.jld")
        load("log/λµσ.jld", "λµσ")
    else
        λµσ = fill((λ=NaN, µ=NaN, σ=NaN), 189)
        Threads.@threads for n in 7:189
            W = sample_Wstatistic(sw_coeffs[n], 1_000_000)
            (λ,µ,σ), time = @timed BoxCox_λµσ(n, W)

            Wt = Wt = (BoxCoxTrans.transform.(1.0 .- W, λ) .- µ)./σ

            let Wt=Wt, qs = qs = 0.005:0.005:0.995
                res = quantile(Wt, qs) .- quantile.(Normal(0,1), qs)
                @info "BoxCox (((1-W)^t - 1)/t - µ)/σ (n=$n, $(time)s)" mse=mean(abs2, res) max=maximum(abs, res) (λ, µ, σ)
            end

            λµσ[n] = (λ=λ, µ=µ, σ=σ)
        end
        save("log/λµσ.jld", "λµσ", λµσ)
        λµσ
    end
end

cubic_model(t, p) =
    ((a₀, a₁, a₂, a₃) = p; @. a₀ + a₁*t + a₂*t^2 + a₃*t^3)

log_model(t, p) = evalpoly.(log.(t), Ref(p))
invlog_model(t, p) = cubic_model(inv.(log.(t)), p)


p_λ = let seed=1, fit_range = 12:140
    Random.seed!(seed)
    p0=rand(4)
    @info "Box-Cox exponent fitting:"
    m = fit_and_plot(
        [x.λ for x in λμσ] .+ 1, 7:189, fit_range, invlog_model,
        p0=p0,
        label="Box-Cox exponent")
    savefig("log/λ_$(fit_range).png")
    m
end

p_µ = let seed=1, fit_range = 12:140
    Random.seed!(seed)
    @info "Mean of the fitted Normal:"
    p0=rand(4)
    m = fit_and_plot(
        [x.µ for x in λμσ], 7:189, fit_range, log_model,
        p0=p0,
        label="mean")
    savefig("log/µ_$(fit_range).png")
    m
end

p_σ = let seed=1, fit_range = 12:140
    Random.seed!(seed)
    @info "Std of the fitted Normal:"
    p0=rand(4)
    m = fit_and_plot(
        [x.σ for x in λμσ], 7:189, fit_range, log_model,
        p0=p0,
        label="standard deviation")
    savefig("log/σ_$(fit_range).png")
    m
end





let n = 20
    W = sample_Wstatistic(sw_coeffs[n], 1_000_000)
    @time λ, µ, σ = BoxCox_λµσ(n, W)

    Wt = Wt = (BoxCoxTrans.transform.(1.0 .- W, λ) .- µ)./σ

    let Wt=Wt, qs = qs = 0.005:0.005:0.995
        res = quantile(Wt, qs) .- quantile.(Normal(0,1), qs)
        @info "BoxCox (((1-W)^t - 1)/t - µ)/σ (n=$n)" mse=mean(abs2, res) max=maximum(abs, res) (λ, µ, σ)
    end

    plot(normpdf, width=2)

    histogram!(Wt, normalized=true, opacity=0.6)
end
