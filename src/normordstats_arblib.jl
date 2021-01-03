module OrderStatisticsArblib

import Statistics
using Arblib

using Memoize
using LRUCache

import ..expectation, ..moment

Arblib.fac!(a::Arblib.ArbLike, n::Signed) = Arblib.fac!(a, UInt(n))
Arblib.fac!(a::Arblib.Mag, n::Signed) = Arblib.fac!(a, UInt(n))

struct Factorials <: AbstractVector{ArbRef}
    cache::ArbVector

    function Factorials(n::Integer; prec = 256)
        cache = ArbVector(n + 1, prec = prec)
        for i = 1:n+1
            Arblib.fac!(Arblib.ref(cache, i), i - 1)
        end
        return new(cache)
    end
end

Base.length(f::Factorials) = length(f.cache) - 1
Base.size(f::Factorials) = (length(f),)
Base.IndexStyle(::Type{Factorials}) = IndexLinear()

Base.@propagate_inbounds function Base.getindex(f::Factorials, i::Integer)
    @boundscheck 0 <= i <= length(f)
    return Arblib.ref(f.cache, i + 1)
end

struct NormOrderStatistic <: AbstractVector{Arb}
    n::Int
    factorial::Factorials
    E::ArbVector

    function NormOrderStatistic(n::Int; prec = 256)

        OS = new(n, Factorials(n, prec = prec), ArbVector(n, prec = prec))
        π_ = Arb(π, prec = prec)
        X = Arblib.rsqrt!(Arb(prec = prec), π_)

        w = Arb(2, prec = prec)
        Arblib.sqrt!(w, w)
        Arblib.atan!(w, w) # w = atan(sqrt(2))

        if n == 2
            OS.E[1] = -X
        elseif OS.n == 3
            OS.E[1] = -3X / 2 # -α(3:3), after eq (3.9.8)
        elseif OS.n == 4
            α44 = (6X / π_) * w # atan(sqrt(Arb(2, prec = prec)))
            OS.E[1] = -α44 # -α(4:4), after eq (3.9.10)
            OS.E[2] = -(4 * 3X / 2 - 3α44) # -(4α(3:3) - 3*α(4:4))
        elseif OS.n == 5
            I₃1 = 3 / (2π_) * w - 1 / 4 # I₃(1), eq (3.9.11)
            OS.E[1] = -10X * I₃1 # -α(5:5), eq (3.9.11)
            α44 = (6X / π_) * w
            OS.E[2] = -(5α44 - 4(10X * I₃1)) # -(5α(4:4) - 4E(5:5))
        else
            for i = 1:div(OS.n, 2)
                OS.E[i] = moment(OS, i, pow = 1)
            end
        end

        if iseven(n)
            OS.E[div(n, 2)+1:end] = -reverse(OS.E[1:div(n, 2)])
        else
            OS.E[div(OS.n, 2)+2:end] = -reverse(OS.E[1:div(OS.n, 2)])
        end
        return OS
    end
end

Base.size(OS::NormOrderStatistic) = (OS.n,)
Base.IndexStyle(::Type{NormOrderStatistic}) = IndexLinear()
Base.getindex(OS::NormOrderStatistic, i::Integer) where {T} = OS.E[i]

Base.precision(OS::NormOrderStatistic) = precision(OS.E)

###############################################################################
#   Exact expected values and moments of normal order statistics

erfc!(res::Arblib.AcbOrRef, x::Arblib.AcbOrRef; prec = precision(res)) =
    Arblib.hypgeom_erfc!(res, x, prec = prec)

function normcdf!(res::Arblib.AcbOrRef, x::Arblib.AcbOrRef; tmp = zero(res))
    tmp[] = 2
    Arblib.rsqrt!(tmp, tmp)
    Arblib.neg!(tmp, tmp)
    Arblib.mul!(tmp, x, tmp)
    erfc!(res, tmp)
    Arblib.div!(res, res, 2)
    return res
end

function normccdf!(res::Arblib.AcbOrRef, x::Arblib.AcbOrRef; tmp = zero(res))
    normcdf!(res, x; tmp = tmp)
    Arblib.one!(tmp)
    Arblib.sub!(res, tmp, res)
    return res
end

function normpdf!(res::Arblib.AcbOrRef, x::Arblib.AcbOrRef; tmp = zero(res))
    Arblib.const_pi!(tmp)
    Arblib.mul!(tmp, tmp, 2)
    Arblib.rsqrt!(res, tmp) # res = 1/sqrt(2π)

    Arblib.pow!(tmp, x, 2)
    Arblib.div!(tmp, tmp, -2)
    Arblib.exp!(tmp, tmp) # tmp = exp((x^2)/-2)

    Arblib.mul!(res, res, tmp)
    return res
end

function _ncdf_i_mul_nccdf_j!(
    res::Arblib.AcbOrRef,
    x::Arblib.AcbOrRef,
    i,
    j;
    tmp1 = zero(res),
    tmp2 = zero(res),
)

    Arblib.one!(res)
    i == 0 && j == 0 && return res

    normcdf!(tmp1, x, tmp = tmp2)
    tmp2[] = tmp1 # tmp2 = normcdf(x)

    if i > 0
        Arblib.pow!(tmp1, tmp1, i)
        Arblib.mul!(res, res, tmp1) # res *= normcdf(x)^i
    end

    if j > 0
        Arblib.one!(tmp1)
        Arblib.sub!(tmp1, tmp1, tmp2)
        Arblib.pow!(tmp1, tmp1, j)
        Arblib.mul!(res, res, tmp1) # res *= normccdf(x)^j
    end
    return res
end

function I!(res::Arblib.AcbOrRef, x, i, j; tmp1 = zero(res), tmp2 = zero(res))
    # normpdf(x) * normcdf(x)^i * normccdf(x)^j

    _ncdf_i_mul_nccdf_j!(res, x, i, j, tmp1 = tmp1, tmp2 = tmp2)

    normpdf!(tmp2, x, tmp = tmp1)
    Arblib.mul!(res, res, tmp2)

    return res
end

function moment(OS::NormOrderStatistic, i::Integer; pow=1, radius=Acb(18.0, prec=precision(OS)))
    n = OS.n
    @assert 1 <= i <= n
    C = Arb(OS.factorial[n], prec = precision(OS))
    Arblib.div!(C, C, OS.factorial[i-1])
    Arblib.div!(C, C, OS.factorial[n-i])

    res = Acb(prec = precision(OS))
    let tmp1 = Acb(prec = precision(OS)), tmp2 = Acb(prec = precision(OS))
        if isone(pow)
            Arblib.integrate!(
                (y, x) -> (I!(y, x, i - 1, n - i, tmp1 = tmp1, tmp2 = tmp2);
                Arblib.mul!(y, y, x)),
                res,
                -radius,
                radius,
            )
        else
            Arblib.integrate!(
                (y, x) -> (I!(y, x, i - 1, n - i, tmp1 = tmp1, tmp2 = tmp2);
                Arblib.pow!(tmp1, x, pow);
                Arblib.mul!(y, y, tmp1)),
                res,
                -radius,
                radius,
            )
        end
    end

    return Arblib.realref(Arblib.mul!(res, res, C))
end

expectation(OS::NormOrderStatistic) = copy(OS.E)
expectation(OS::NormOrderStatistic, i::Integer) = OS[i]

###############################################################################
#
#   Exact expected products of normal order statistics
#
# after H.J. Godwin
# Some Low Moments of Order Statistics
# Ann. Math. Statist.
# Volume 20, Number 2 (1949), 279-285.
# doi:10.1214/aoms/1177730036
#
# Radius of integration = 12.2 in
# Rudolph S. Parrish
# Computing variances and covariances of normal order statistics
# Communications in Statistics - Simulation and Computation
# Volume 21, 1992 - Issue 1
# doi:10.1080/03610919208813009
#
###############################################################################

function α_int!(res::Arblib.AcbOrRef, i::Integer, j::Integer, r)
    Arblib.integrate!((y, x) -> (_ncdf_i_mul_nccdf_j!(y, x, i, j);
    Arblib.mul!(y, y, x)), res, -r, r)
    return res
end

function β_int!(res::Arblib.AcbOrRef, i::Integer, j::Integer, r)
    Arblib.integrate!((y, x) -> (I!(y, x, i, j);
    Arblib.mul!(y, y, x);
    Arblib.mul!(y, y, x)), res, -r, r)
    return res
end

function _ψ_integrand!(
    res::Arblib.AcbOrRef,
    j::Integer,
    x::Arblib.AcbOrRef,
    r;
    tmp = zero(res),
)
    let tmp = tmp
        Arblib.integrate!((y, x) -> (normcdf!(y, x, tmp = tmp);
        Arblib.pow!(y, y, j)), res, r, x)
    end
    return res
end

function ψ_int!(res::Arblib.AcbOrRef, i::Integer, j::Integer, r)
    let tmp1 = zero(res),
        tmp2 = zero(res),
        R = Acb(r, prec = precision(res)),
        mR = -R

        Arblib.integrate!(
            (y, x) -> (normcdf!(y, x, tmp = tmp1);
            Arblib.pow!(y, y, i);
            _ψ_integrand!(tmp1, j, -x, mR, tmp = tmp2);
            Arblib.mul!(y, y, tmp1)),
            res,
            mR,
            R,
        )
    end
    return res
end

function α!(res::Arblib.AcbOrRef, i::Integer, j::Integer, r)
    j < i && return Arblib.neg!(res, α!(res, j, i, r))
    return α_int!(res, i, j, r)
end

function β!(res::Arblib.AcbOrRef, i::Integer, j::Integer, r)
    j < i && return β!(res, j, i, r)
    return β_int!(res, i, j, r)
end

function ψ!(res::Arblib.AcbOrRef, i::Integer, j::Integer, r)
    j < i && return ψ!(res, j, i, r)

    if i == 1
        res[] = j + 1
        Arblib.inv!(res, res)

        tmp = zero(res)
        α!(tmp, j, 1, r)

        Arblib.sub!(res, res, tmp)
        return res
    end

    return ψ_int!(res, i, j, r)
end

@memoize () -> LRU{Tuple{Int,Int,Int,Float64},Arb}(maxsize = 10000) α(
    prec::Int,
    i::Int,
    j::Int,
    r::Float64,
) = real(α!(Acb(prec = prec), i, j, r))
@memoize () -> LRU{Tuple{Int,Int,Int,Float64},Arb}(maxsize = 10000) β(
    prec::Int,
    i::Int,
    j::Int,
    r::Float64,
) = real(β!(Acb(prec = prec), i, j, r))
@memoize () -> LRU{Tuple{Int,Int,Int,Float64},Arb}(maxsize = 10000) ψ(
    prec::Int,
    i::Int,
    j::Int,
    r::Float64,
) = real(ψ!(Acb(prec = prec), i, j, r))

@memoize () -> LRU{Tuple{Int,Int,Int,Float64},Arb}(maxsize = 10000) function _γ(
    prec::Int,
    i::Int,
    j::Int,
    r::Float64,
)
    @assert i >= 0
    @assert j >= 0
    @assert i <= j

    @debug "memoizing γ with prec=$prec, i=$i, j=$j, r=$r"
    return (α(prec, i, j, r) + i * β(prec, i - 1, j, r) - ψ(prec, i, j, r)) / (i * j)
end

γ(prec::Integer, i::Integer, j::Integer, r) =
    (j < i ? _γ(prec, j, i, r) : _γ(prec, i, j, r))

function K!(res::Arblib.ArbOrRef, facs::Factorials, n::Integer, i::Integer, j::Integer)
    #n!/((i-1)!*(n-j)!*(j-i-1)!)
    res[] = Arb(facs[n])
    Arblib.div!(res, res, facs[i-1])
    Arblib.div!(res, res, facs[n-j])
    Arblib.div!(res, res, facs[j-i-1])
    return res
end

function expectation(OS::NormOrderStatistic, i::Integer, j::Integer; radius = 18.0)
    if i == j
        return moment(OS, i, pow = 2, radius=radius)
    elseif i > j
        return expectation(OS, j, i, radius=radius)
    elseif i + j > OS.n + 1
        return expectation(OS, OS.n - j + 1, OS.n - i + 1, radius=radius)
    else
        n = OS.n
        res = Arb(prec = precision(OS))
        tmp = zero(res)
        denom = Arb(prec = precision(OS))

        for r = 0:j-i-1
            for s = 0:j-i-1-r
                tmp[] = γ(precision(OS), i + r, n - j + s + 1, radius)

                Arblib.mul!(tmp, tmp, OS.factorial[j-i-1])
                isodd(r + s) && Arblib.neg!(tmp, tmp)

                Arblib.mul!(denom, OS.factorial[r], OS.factorial[s])
                Arblib.mul!(denom, denom, OS.factorial[j-i-1-r-s])
                Arblib.div!(tmp, tmp, denom)

                # res += (-1)^(r+s)*γ(i+r, n-j+s+1)*(j-i-1)!/( r!*s!*(j-i-1-r-s)! )

                Arblib.add!(res, res, tmp)
            end
        end

        K!(tmp, OS.factorial, n, i, j)
        Arblib.mul!(res, res, tmp)
        return res
    end
end

################################################################################
#   Exact variances and covariances

function Statistics.var(OS::NormOrderStatistic, i::Integer)
    # return expectation(OS, i, i) - expectation(OS, i)^2
    Ei² = expectation(OS, i)
    Arblib.pow!(Ei², Ei², UInt(2))
    Eii = expectation(OS, i, i)
    return Arblib.sub!(Eii, Eii, Ei²)
end

function Statistics.cov(OS::NormOrderStatistic, i::Integer, j::Integer)
    # return expectation(OS,i,j) - expectation(OS,i)*expectation(OS,j)
    i == j && return Statistics.var(OS, i)
    EiEj = expectation(OS, i)
    Arblib.mul!(EiEj, EiEj, expectation(OS, j))
    Eij = expectation(OS, i, j)
    return Arblib.sub!(Eij, Eij, EiEj)
end

function Statistics.cov(OS::NormOrderStatistic)
    V = ArbMatrix(OS.n, OS.n, prec = precision(OS))
    for i = 1:OS.n
        V[i, i] = Statistics.cov(OS, i, i)
        for j = i+1:OS.n
            V[i, j] = Statistics.cov(OS, i, j)
            V[j, i] = Arblib.ref(V, i, j)
        end
    end
    return V
end

################################################################################
#   Precomputation routines

ijs_ψ(n::Int) = [(i, j) for i = 2:div(n, 2) for j = i:n-i]

function _precompute_ψ(n; prec, R)
    pairs = ijs_ψ(n)
    tasks = [Threads.@spawn ψ(prec, i, j, R) for (i, j) in pairs]
    w = fetch.(tasks)

    return pairs, w
end

end
