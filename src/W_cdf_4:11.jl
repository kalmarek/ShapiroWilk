using DelimitedFiles

struct CumulativeDistribution{T <: Real}
    x::Vector{T}
    y::StepRangeLen{T, Base.TwicePrecision{T}, Base.TwicePrecision{T}}
end

function (f::CumulativeDistribution{T})(arg::Real) where T
    arg < first(f.x) && return zero(T)
    arg > last(f.x) && return one(T)
    k = findfirst(>=(arg), f.x)
    @assert k !== nothing

    x₋, x₊, y₋, y₊ = f.x[k], f.x[k+1], f.y[k], f.y[k+1]
    t = (arg - x₋)/(x₊ - x₋)
    return (one(t) - t) * y₋ + t * y₊
end

qs = 0.0005:0.001:0.9995

vals = readdlm(joinpath(@__DIR__, "W_cdf_4:11.csv"))
