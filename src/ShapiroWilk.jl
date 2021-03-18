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

function Wstatistic(X, A::SWCoeffs)

    @assert issorted(X)

    if last(X) - first(X) < length(X) * eps(eltype(X))
        throw("Data seems to be constant!")
    end

    µ = Statistics.mean(X)
    S² = sum(x->abs2(x-µ), X)

    return dot(A, X)^2 / S²
end

end # of module ShapiroWilk
