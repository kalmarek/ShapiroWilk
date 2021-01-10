module ShapiroWilk

import LinearAlgebra: dot
import Statistics

export SWCoeffs, Wstatistic

include("orderstatistics.jl")

include("swcoeffs.jl")
include("royston.jl")

include("normordstats_nemo.jl")
include("normordstats_arblib.jl")

function Wstatistic(X, A::SWCoeffs)

    @assert issorted(X)

    if last(X) - first(X) < length(X)*eps(eltype(X))
        throw("Data seems to be constant!")
    end

    AX = dot(A,X)
    S² = sum(abs2, X.-mean(X))

    return AX^2/S²
end

end # of module ShapiroWilk
