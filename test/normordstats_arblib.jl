function test_α_ij(OS::ShapiroWilk.NormOrderStatistic; atol)
    n, prec, R = OS.n, precision(OS), ShapiroWilk.integration_radius(OS)

    @testset "α_ij (n=$n, prec=$prec, atol=$atol, R=$R)" begin
        α = ShapiroWilk.α
        for i = 1:n
            for j = 1:n
                alpha_ij_residual =
                    α(prec, i, j, R) - (α(prec, i, j + 1, R) + α(prec, i + 1, j, R))
                @test Arblib.contains_zero(alpha_ij_residual)
                @test alpha_ij_residual < atol
            end
        end
    end
end

function test_β_ii(OS::ShapiroWilk.NormOrderStatistic; atol)
    n, prec, R = OS.n, precision(OS), ShapiroWilk.integration_radius(OS)

    @testset "β_ii (n=$n, prec=$prec, atol=$atol, R=$R)" begin
        α = ShapiroWilk.α
        β = ShapiroWilk.β
        for i = 1:n
            beta_ii_residual =
                β(prec, i, i, R) -
                (i * β(prec, i - 1, i - 1, R) / (4i + 2) - 2α(prec, i + 1, i, R) / (2i + 1))
            @test Arblib.contains_zero(beta_ii_residual)
            @test real(beta_ii_residual) < atol
        end
    end
end

function test_β_ij(OS::ShapiroWilk.NormOrderStatistic; atol)
    n, prec, R = OS.n, precision(OS), ShapiroWilk.integration_radius(OS)
    @testset "β_ij (n=$n, prec=$prec, atol=$atol, R=$R)" begin
        β = ShapiroWilk.β
        for i = 1:n
            for j = 1:n
                beta_ij_residual =
                    β(prec, i, j, R) - (β(prec, i, j + 1, R) + β(prec, i + 1, j, R))
                @test Arblib.contains_zero(beta_ij_residual)
                # @info (i,j) beta_ij_residual
                @test real(beta_ij_residual) < atol
            end
        end
    end
end

function test_sum_moments_arblib(OS; atol)
    @time ShapiroWilk._precompute(OS)

    @testset "Sums of products and moments: Arblib (n=$(OS.n))" begin
        for i = 1:OS.n-1
            res = sum(Distributions.expectation(OS, i, j) for j = 1:OS.n)
            @info res
            @test Arblib.contains_zero(res - 1)
            @test res - 1 < atol
        end

        res = sum(Distributions.moment(OS, i, pow = 2) for i = 1:OS.n)
        @test Arblib.contains_zero(res - OS.n)
        @test res - OS.n < atol
    end
end

function numeric_tests_order_statistics_arblib(OS::ShapiroWilk.NormOrderStatistic; atol)
    n, prec, R = OS.n, precision(OS), ShapiroWilk.integration_radius(OS)

    @testset "Relations between α, β and expectations/moments of OS using Arblib (n=$n)" begin

        @time test_α_ij(OS, atol=atol)
        @time test_β_ii(OS, atol=atol)
        @time test_β_ij(OS, atol=atol)

        @time test_sum_moments_arblib(OS, atol=atol)
    end
end



let
    # import Memoize
    # empty!(Memoize.memoize_cache(ShapiroWilk.α))
    # empty!(Memoize.memoize_cache(ShapiroWilk.β))
    # empty!(Memoize.memoize_cache(ShapiroWilk.ψ))
    # empty!(Memoize.memoize_cache(ShapiroWilk._γ))

    OS = ShapiroWilk.NormOrderStatistic(6, prec=69, radius=18.0)
    numeric_tests_order_statistics_arblib(OS, atol=eps(Float64))

    OS = ShapiroWilk.NormOrderStatistic(10, prec=96, radius=18.0)
    test_sum_moments_arblib(OS, atol=2e-22)

    OS = ShapiroWilk.NormOrderStatistic(20, prec=96, radius=18.0)
    test_sum_moments_arblib(OS, atol=2e-16)
end

if false

    # @time test_α_ij(50, 128, 4e-36, 18.0)
    # @time test_β_ii(50, 128, 2e-37, 18.0)
    # @time test_β_ij(50, 128, 5e-38, 18.0)

    OS = ShapiroWilk.NormOrderStatistic(10, prec=128, radius=18.0)
    test_sum_moments_arblib(OS, atol=3e-32)

    OS = ShapiroWilk.NormOrderStatistic(20, prec=128, radius=18.0)
    test_sum_moments_arblib(OS, atol=7e-27)

    OS = ShapiroWilk.NormOrderStatistic(30, prec=128, radius=18.0)
    test_sum_moments_arblib(OS, atol=2e-19)

    OS = ShapiroWilk.NormOrderStatistic(40, prec=128, radius=18.0)
    @time test_sum_moments_arblib(OS, atol=4e-12)

    OS = ShapiroWilk.NormOrderStatistic(50, prec=128, radius=18.0)
    @time test_sum_moments_arblib(OS, atol=6e-6)
end
