const ROYSTON_COEFFS = Dict{String, Vector{Float64}}(
"C1" => [0.0E0, 0.221157E0, -0.147981E0, -0.207119E1, 0.4434685E1, -0.2706056E1],
"C2" => [0.0E0, 0.42981E-1, -0.293762E0, -0.1752461E1, 0.5682633E1, -0.3582633E1],
"C3" => [0.5440E0, -0.39978E0, 0.25054E-1, -0.6714E-3],
"C4" => [0.13822E1, -0.77857E0, 0.62767E-1, -0.20322E-2],
"C5" => [-0.15861E1, -0.31082E0, -0.83751E-1, 0.38915E-2],
"C6" => [-0.4803E0, -0.82676E-1, 0.30302E-2],
"C7" => [0.164E0, 0.533E0],
"C8" => [0.1736E0, 0.315E0],
"C9" => [0.256E0, -0.635E-2],
"G"  => [-0.2273E1, 0.459E0]
)

for (s,c) in ROYSTON_COEFFS
    @eval $(Symbol("_"*s))(x) = Base.Math.@horner(x, $(c...))
end

function SWCoeffs(N::Int)
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
