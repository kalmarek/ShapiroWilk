module ShapiroWilk

import Statistics: cov
import StatsFuns: norminvcdf

export SWCoeffs, expectation, moment

function expectation end
function moment end

include("swcoeffs.jl")
include("royston.jl")

include("normordstats_nemo.jl")
include("normordstats_arblib.jl")

end # of module ShapiroWilk
