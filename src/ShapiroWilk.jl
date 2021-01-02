module ShapiroWilk

using Statistics
using StatsFuns

struct SWCoeffs{T} <: AbstractVector{T}
    N::Int
    A::Vector{T}
end

Base.size(SWc::SWCoeffs) = (SWc.N,)
Base.IndexStyle(::Type{SWCoeffs}) = IndexLinear()

function Base.getindex(SWc::SWCoeffs, i::Int)
    if i <= lastindex(SWc.A)
        return SWc.A[i]
    elseif i <= length(SWc)
        if isodd(SWc.N) && i == div(SWc.N, 2) + 1
            return 0.0
        else
            return -SWc.A[SWc.N+1-i]
        end
    else
        throw(BoundsError(SWc, i))
    end
end

function expectation end

function SWCoeffs(OS)
    m = expectation(OS)
    minvV = m'*inv(cov(OS))
    A = minvV./sqrt(first(minvV*minvV'))
    return SWCoeffs(OS.n, -A[1,1:div(OS.n,2)])
end

include("normordstats.jl")
include("royston.jl")

end # of module ShapiroWilk
