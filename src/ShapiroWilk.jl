module ShapiroWilk

import LinearAlgebra: dot
import Statistics
import Distributions

using Arblib
import Memoize: @memoize
import LRUCache: LRU

export SWCoeffs, Wstatistic

include("orderstatistics.jl")

include("swcoeffs.jl")
include("royston.jl")

include("normordstats_arblib.jl")
include("normalizing_transform_12:Inf.jl")

_W_cdf_4_11 = let
    include("W_cdf_4:11.jl")
    Dict(n => CumulativeDistribution(vals[n-3, :], qs) for n in 4:11)
end


function Wstatistic(X, A::SWCoeffs)

    @assert issorted(X)

    if last(X) - first(X) < length(X) * eps(eltype(X))
        throw("Data seems to be constant!")
    end

    µ = Statistics.mean(X)
    S² = sum(x->abs2(x-µ), X)

    return dot(A, X)^2 / S²
end

function pvalue(w::Real, A::ShapiroWilk.SWCoeffs, N1=A.N)

    A.N > N1 && throw("Shapiro-Wilk test for censored samples is not implemented yet.")

    if A.N == 3 # exact integration by Shapiro&Wilk 1965
        return 6/pi*asin(sqrt(w)) - 2
    elseif 4 ≤ A.N ≤ 11
        # linear interpolation over 1000 quantiles of MC simulation (10_000_000 samples)
        _W_cdf_4_11[A.N](w)
    else # Parrish 1992: BoxCox normalizing transform
        λ, µ, σ = λμσ(A.N)

        # BoxCox:
        w = λ ≈ 0 ? log(1 - w) : ((1 - w) ^ λ) / λ
        # N(0,1) shift:
        z = (w - µ)/σ
        return normccdf(z)
    end
end

end # of module ShapiroWilk
