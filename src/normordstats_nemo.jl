module OrderStatisticsNemo

using Statistics
using StatsFuns
import Distributions

using Nemo

import ..OrderStatistic

###############################################################################
#
#   NormOrderStatistic Type
#
###############################################################################

nf(k) = Nemo.gamma(k+1)

struct Factorials{T} <: AbstractVector{T}
    cache::Vector{T}
end
Base.size(f::Factorials) = size(f.cache)
Base.IndexStyle(::Type{<:Factorials}) = IndexLinear()
Base.@propagate_inbounds function Base.getindex(f::Factorials, i::Integer)
    iszero(i) && return one(first(f.cache))
    return f.cache[i]
end

mutable struct NormOrderStatistic <: OrderStatistic{Nemo.arb}
    n::Int
    facs::Factorials{acb}
    E::Vector{acb}

    function NormOrderStatistic(n::Int, F)

        OS = new(n, Factorials([nf(F(k)) for k in 1:n]))

        OS.E = fill(zero(F), OS.n)
        # expected values of order statistics
        # Exact Formulas:
        # N. Balakrishnan, A. Clifford Cohen
        # Order Statistics & Inference: Estimation Methods
        # Section 3.9
        π = Nemo.const_pi(F)
        X = sqrt(1/π)
        if OS.n == 2
            OS.E[1] = -X # -α(2:2), after eq (3.9.4)
        elseif OS.n == 3
            OS.E[1] = -3X/2 # -α(3:3), after eq (3.9.8)
        elseif OS.n == 4
            α44 = (6X/π)*atan(sqrt(F(2)))
            OS.E[1] = -α44 # -α(4:4), after eq (3.9.10)
            OS.E[2] = -(4*3X/2 - 3α44) # -(4α(3:3) - 3*α(4:4))
        elseif OS.n == 5
            I₃1 = 3/(2π) * atan(sqrt(F(2))) - F(1)/4 # I₃(1), eq (3.9.11)
            OS.E[1] = -10X * I₃1 # -α(5:5), eq (3.9.11)
            α44 = (6X/π)*atan(sqrt(F(2)))
            OS.E[2] = -(5α44 - 4(10X*I₃1)) # -(5α(4:4) - 4E(5:5))
        else
            for i in 1:div(OS.n,2)
                OS.E[i] = F(Distributions.moment(OS, i, pow = 1))
            end
        end

        if iseven(OS.n)
            OS.E[div(OS.n,2)+1: end] = -reverse(OS.E[1:div(OS.n,2)])
        else
            OS.E[div(OS.n, 2)+2:end] = -reverse(OS.E[1:div(OS.n,2)])
        end
        return OS
    end
end

Base.size(OS::NormOrderStatistic) = (OS.n,)
Base.IndexStyle(::Type{<:NormOrderStatistic}) = IndexLinear()
Base.getindex(OS::NormOrderStatistic, i::Integer) = real(OS.E[i])

Base.precision(OS::NormOrderStatistic) = precision(parent(first(OS)))
Nemo.base_ring(OS::NormOrderStatistic) = Nemo.AcbField(precision(OS))

###############################################################################
#
#   Exact expected values of normal order statistics
#
###############################################################################

StatsFuns.normcdf(x::acb) = (F = parent(x); Nemo.erfc(-x/sqrt(F(2)))/2)
StatsFuns.normccdf(x::acb) = 1 - normcdf(x)
StatsFuns.normpdf(x::acb) = (F = parent(x); 1/sqrt(2const_pi(F))*exp(-x^2/2))

I(x, i, j) = normcdf(x)^i * normccdf(x)^j * normpdf(x)

function Distributions.moment(
    OS::NormOrderStatistic,
    i::Int;
    pow = 1,
    radius = 18.0,
)
    n = OS.n
    @assert 1<= i <= n
    C = OS.facs[n]/OS.facs[i-1]/OS.facs[n-i]
    F = Nemo.base_ring(OS)
    return real(C*Nemo.integrate(F, x -> x^pow * I(x, i-1, n-i), -radius, radius))
end

Distributions.expectation(OS::NormOrderStatistic) = real.(OS.E)
Distributions.expectation(OS::NormOrderStatistic, i::Int) = OS[i]

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

function α_int(i::Int, j::Int, r::acb)
    res = Nemo.integrate(parent(r), x -> x * normcdf(x)^i * normccdf(x)^j, -r, r)
    return res
end

function β_int(i::Int, j::Int, r::acb)
    res = Nemo.integrate(parent(r), x -> x^2 * I(x, i, j), -r, r)
    return res
end

function integrand(j::Int, x::acb, r::acb)
    return Nemo.integrate(parent(r), y -> normcdf(y)^j, -r, -x)
end

function ψ_int(i::Int, j::Int, r::acb)
    res = Nemo.integrate(parent(r),
        x -> normcdf(x)^i * integrand(j, x, r), -r, r)
    return res
end

function α(F::AcbField, i::Int, j::Int, r)
    j < i && return -α(F, j, i, r)
    args = (i,j, F(r))
    # returnT = first(Base.return_types(α_int, typeof(args)))
    return getval!(α_int, acb, args...)
end

function β(F::AcbField, i::Int, j::Int, r)
    j < i && return β(F, j, i, r)
    args = (i,j, F(r))
    # returnT = first(Base.return_types(β_int, typeof(args)))
    return getval!(β_int, acb, args...)
end

function ψ(F::AcbField, i::Int, j::Int, r)
    j < i && return ψ(F, j, i, r)
    i == 1 && return inv(F(j+1)) - α(F, j, 1, r)
    args = (i,j, F(r))
    # returnT = first(Base.return_types(ψ_int, typeof(args)))::Type
    return getval!(ψ_int, acb, args...)
end

function γ(F, i::Int, j::Int, r)

    j > i && return γ(F, j, i, r)

    res = zero(F)
    res = Nemo.add!(res, res, i*β(F,i-1,j,r))
    res = Nemo.add!(res, res,   α(F,i,  j,r))
    res = Nemo.sub!(res, res,   ψ(F,i,  j,r))
    res = Nemo.div!(res, Nemo.div!(res, res, F(i)), F(j)) # res = res/(i*j)

    return F(abs(res))
end

function K(facs::Factorials, n::Integer, i::Integer, j::Integer)
    #n!/((i-1)!*(n-j)!*(j-i-1)!)
    return facs[n]/facs[i-1]/facs[n-j]/facs[j-i-1]
end

function Distributions.expectation(
    OS::NormOrderStatistic,
    i::Int,
    j::Int;
    radius = 18.0,
)
    if i == j
        return Distributions.moment(OS, i, pow = 2, radius = radius)
    elseif i > j
        return Distributions.expectation(OS, j, i, radius = radius)
    elseif i + j > OS.n + 1
        return Distributions.expectation(
            OS,
            OS.n - j + 1,
            OS.n - i + 1,
            radius = radius,
        )
    else
        n = OS.n
        C = K(OS.facs, n, i, j)
        F = Nemo.base_ring(OS)
        S = zero(F)
        num = zero(F)
        denom = one(F)
        tmp = zero(F)

        NUM = OS.facs[j-i-1]
        NEGATIVE_NUM = -NUM

        for r in 0:j-i-1
            for s in 0:j-i-1-r
                num = (isodd(r+s) ? NEGATIVE_NUM : NUM)
                denom = Nemo.mul!(denom, OS.facs[r], OS.facs[s])
                denom = Nemo.mul!(denom, denom, OS.facs[j-i-1-r-s])
                tmp = Nemo.div!(tmp, num, denom)
                S = Nemo.addmul!(S, tmp, γ(F, i+r, n-j+s+1, radius), tmp)
            end
        end
        return real(Nemo.mul!(tmp, S, C))
    end
end

###############################################################################
#
#   Exact variances and covariances
#
###############################################################################

function Statistics.cov(OS::NormOrderStatistic)
    F = ArbField(precision(OS))
    V = matrix(F, zeros(OS.n, OS.n))
    for i = 1:OS.n
        V[i, i] = F(cov(OS, i, i))
        for j = i+1:OS.n
            V[i, j] = F(cov(OS, i, j))
            V[j, i] = V[i, j]
        end
    end
    return V # V is a Nemo.MatElem
end

include("precompute.jl")


### needed for SWCoeffs

function Base.:*(m::AbstractArray, n::Nemo.MatElem)
    @boundscheck size(m, 2) == size(n, 1) || throw(ArgumentError("Incompatible sizes for matrix multiplication: $(join(size(m), "×")) and $(join(size(n), "×"))"))
    res = similar(m, size(m, 1), size(n, 2))
    @inbounds for c in 1:size(res, 2)
        for r in 1:size(res, 1)
            res[r, c] = sum(m[r, i]*n[i, c] for i in 1:size(m, 2))
        end
    end
    return res
end

function Base.:*(m::Nemo.MatElem, n::AbstractArray)
    @boundscheck size(m, 2) == size(n, 1) || throw(ArgumentError("Incompatible sizes for matrix multiplication: $(join(size(m), "×")) and $(join(size(n), "×"))"))
    res = similar(n, size(m, 1), size(n, 2))
    @inbounds for c in 1:size(res, 2)
        for r in 1:size(res, 1)
            res[r, c] = sum(m[r, i]*n[i, c] for i in 1:size(m, 2))
        end
    end
    return res
end

end # of module OrderStatistics
