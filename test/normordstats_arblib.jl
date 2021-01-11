function test_α_ij(n, prec, atol, R)
    @testset "α_ij (n=$n, prec=$prec, atol=$atol, R=$R)" begin
        α = OrderStatisticsArblib.α
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

function test_β_ii(n, prec, atol, R)
    @testset "β_ii (n=$n, prec=$prec, atol=$atol, R=$R)" begin
        α = OrderStatisticsArblib.α
        β = OrderStatisticsArblib.β
        for i = 1:n
            beta_ii_residual =
                β(prec, i, i, R) -
                (i * β(prec, i - 1, i - 1, R) / (4i + 2) - 2α(prec, i + 1, i, R) / (2i + 1))
            @test Arblib.contains_zero(beta_ii_residual)
            @test real(beta_ii_residual) < atol
        end
    end
end

function test_β_ij(n, prec, atol, R)
    @testset "β_ij (n=$n, prec=$prec, atol=$atol, R=$R)" begin
        β = OrderStatisticsArblib.β
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

function test_sum_moments_arblib(OS; atol, R)

    @time OrderStatisticsArblib._precompute(OS.n, prec=precision(OS), R=R)

    @testset "Sums of products and moments: Arblib (n=$(OS.n))" begin
        for i = 1:OS.n-1
            res = sum(Distributions.expectation(OS, i, j, radius=R) for j = 1:OS.n)
            @info res
            @test Arblib.contains_zero(res - 1)
            @test res - 1 < atol
        end

        res = sum(Distributions.moment(OS, i, pow = 2, radius=R) for i = 1:OS.n)
        @test Arblib.contains_zero(res - OS.n)
        @test res - OS.n < atol
    end
end

function numeric_tests_order_statistics_arblib(n::Integer; prec, atol, R)
    @testset "Relations between α, β and expectations/moments of OS using Arblib (n=$n)" begin

        @time test_α_ij(n, prec, atol, R)
        @time test_β_ii(n, prec, atol, R)
        @time test_β_ij(n, prec, atol, R)

        OS = OrderStatisticsArblib.NormOrderStatistic(n, prec=prec)
        @time test_sum_moments_arblib(OS, atol=atol, R=R)
    end
end
