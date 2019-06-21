ijs_ψ(n::Int) = [(i,j) for i in 2:div(n, 2) for j in i:n-i]

function precompute_ψ(N::Integer, F::AcbField; R=OrderStatistics.RADIUS.R)
    ijs = ijs_ψ(N)
    args = [(i,j, F(R)) for (i,j) in ijs]

    ψ_vals = Vector{acb}(undef, length(args))

    Threads.@threads for i in 1:length(args)
        arg = args[i]
        ψ_vals[i] = OrderStatistics.ψ_int(arg...)
    end
    vals = Dict(zip(args, ψ_vals))
    for (k,v) in vals
        OrderStatistics.setcache!(OrderStatistics.ψ_int, k, v)
    end
end
