function test_α_ij(n, prec, atol, R)
    CC = AcbField(prec)
    @testset "α_ij" begin
        α = OrderStatisticsNemo.α
        for i in 1:n
            for j in 1:n
                alpha_ij_residual = α(CC, i, j, R) - ( α(CC, i, j+1, R) + α(CC, i+1, j, R) )
                @test Nemo.contains_zero(alpha_ij_residual)
                @test real(alpha_ij_residual) < atol
            end
        end
    end
end

function test_β_ii(n, prec, atol, R)
    CC = AcbField(prec)
    @testset "β_ii" begin
        α = OrderStatisticsNemo.α
        β = OrderStatisticsNemo.β
        for i in 1:n
            beta_ii_residual = β(CC, i, i, R) - ( i*β(CC, i-1, i-1, R)/(4i + 2) -  2α(CC, i+1, i, R)/(2i + 1))
            @test Nemo.contains_zero(beta_ii_residual)
            @test real(beta_ii_residual) < atol
        end
    end
end
function test_β_ij(n, prec, atol, R)
    CC = AcbField(prec)
    @testset "β_ij" begin
        β = OrderStatisticsNemo.β
        for i in 1:n
            for j in 1:n
                beta_ij_residual = β(CC, i, j, R) - ( β(CC, i, j+1, R) + β(CC, i+1, j, R) )
                @test Nemo.contains_zero(beta_ij_residual)
                @test real(beta_ij_residual) < atol
            end
        end
    end
end

function test_sum_moments_nemo(OS; atol, R)

    @time OrderStatisticsNemo._precompute(OS.n, prec=precision(OS), R=R)

    @testset "Sums of products and moments: Nemo (n = $(OS.n))" begin
        for i in 1:OS.n-1
            res = sum(Distributions.expectation(OS, i, j, radius=R) for j in 1:OS.n)
            @info res
            @test Nemo.contains_zero(res - 1)
            @test res - 1 < atol
        end

        res = sum(Distributions.moment(OS, i, pow=2, radius=R) for i in 1:OS.n)
        @test Nemo.contains_zero(res - OS.n)
        @test res - OS.n < atol
    end
end

function numeric_tests_order_statistics_nemo(N, CC; atol, R)

    @testset "Relations between α, β and expectations/moments of OS using Nemo (n=$N)" begin
        prec=precision(CC)
        @time test_α_ij(N, prec, atol, R)
        @time test_β_ii(N, prec, atol, R)
        @time test_β_ij(N, prec, atol, R)

        OS = OrderStatisticsNemo.NormOrderStatistic(N, CC)
        @time test_sum_moments_nemo(OS; atol=atol, R=R)
    end
end
