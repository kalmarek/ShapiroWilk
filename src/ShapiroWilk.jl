module ShapiroWilk

import Statistics: cov
import StatsFuns: norminvcdf


function expectation end

include("swcoeffs.jl")
include("royston.jl")

include("normordstats_nemo.jl")

end # of module ShapiroWilk
