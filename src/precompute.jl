###############################################################################
#   Poor man's caching

const global _cache = Dict{Symbol,Dict}()

function dropcache!(cache = _cache)
    for k in keys(cache)
        delete!(cache, k)
    end
    return cache
end

function setvalue!(cache, f, args, val)
    sf = Symbol(f)
    if !(haskey(cache, sf))::Bool
        cache[sf] = Dict{typeof(args),typeof(val)}()
    end
    return cache[sf][args] = val
end

setcache!(f, args, val) = setvalue!(_cache, f, args, val)

function getval!(f, ::Type{returnT}, args...) where {returnT}
    sf = Symbol(f)
    if !(haskey(_cache, sf))::Bool
        _cache[sf] = Dict{typeof(args),returnT}()
    end
    # @info "Computing $f with" args
    g() = f(args...)
    return get!(g, _cache[sf], args)::returnT
end

###############################################################################
#   Precomutation of ψ

ijs_ψ(n::Int) = [(i, j) for i = 2:div(n, 2) for j = i:n-i]

function _precompute_ψ(N::Integer, F::AcbField; R)
    args = [(i, j, F(R)) for (i, j) in ijs_ψ(N)]

    ψ_vals = Vector{acb}(undef, length(args))

    Threads.@threads for i = 1:length(args)
        arg = args[i]
        ψ_vals[i] = ψ_int(arg...)
    end
    vals = Dict(zip(args, ψ_vals))
    for (k, v) in vals
        setcache!(ψ_int, k, v)
    end
end

Base.hash(a::Nemo.acb, h::UInt = UInt(0)) = hash(precision(parent(a)), h)
