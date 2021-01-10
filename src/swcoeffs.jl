import Distributions: expectation
import StatsFuns: norminvcdf
import Statistics: cov

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

function SWCoeffs(N::Integer)
    if N < 3
        throw(ArgumentError("N must be greater than or equal to 3: got $N instead."))
    elseif N == 3 # exact
        return SWCoeffs(N, [sqrt(2.0)/2.0])
    else
        # Weisberg&Bingham 1975 statistic; store only positive half of m:
        # it is (anti-)symmetric; hence '2' factor below
        m = [-norminvcdf((i - 3/8)/(N + 1/4)) for i in 1:div(N,2)]
        mᵀm = 2sum(abs2, m)

        x = 1/sqrt(N)

        a₁ = m[1]/sqrt(mᵀm) + _C1(x) # aₙ = cₙ + (...)

        if N ≤ 5
            # renormalize and correct the first coefficient
            ϕ = (mᵀm - 2m[1]^2)/(1 - 2a₁^2)
            m .= m/sqrt(ϕ) # A, but reusing m to save allocs
            m[1] = a₁
        else
            # renormalize and correct the first two coefficients
            a₂ = m[2]/sqrt(mᵀm) + _C2(x) # aₙ₋₁ = cₙ₋₁ + (...)
            ϕ = (mᵀm - 2m[1]^2 - 2m[2]^2)/(1 - 2a₁^2 - 2a₂^2)
            m .= m/sqrt(ϕ) # A, but reusing m to save allocs
            m[1], m[2] = a₁, a₂
        end

        return SWCoeffs(N, m)
    end
end

function SWCoeffs(OS)
    m = expectation(OS)
    mΣ⁻¹ = m'*inv(cov(OS))
    A = mΣ⁻¹./sqrt(first(mΣ⁻¹*mΣ⁻¹'))
    return SWCoeffs(OS.n, -A[1,1:div(OS.n,2)])
end
