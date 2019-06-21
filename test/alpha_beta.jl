using ShapiroWilk.OrderStatistics: α, β

function test_α_ij(OS, CC, tol, R)
    @testset "α_ij" begin
        for i in 1:OS.n
            for j in 1:OS.n
                alpha_ij_residual = α(CC, i, j, R) - ( α(CC, i, j+1, R) + α(CC, i+1, j, R) )
                check, int = unique_integer(alpha_ij_residual)
                @test check && int == 0
                @test real(alpha_ij_residual) < tol
            end
        end
    end
end

function test_β_ii(OS, CC, tol, R)
    @testset "β_ii" begin
        for i in 1:OS.n
            beta_ii_residual = β(CC, i, i, R) - ( i*β(CC, i-1, i-1, R)/(4i + 2) -  2α(CC, i+1, i, R)/(2i + 1))
            check, int = unique_integer(beta_ii_residual)
            @test check && int == 0
            @test real(beta_ii_residual) < tol
        end
    end
end
function test_β_ij(OS, CC, tol, R)
    @testset "β_ij" begin
        for i in 1:OS.n
            for j in 1:OS.n
                beta_ij_residual = β(CC, i, j, R) - ( β(CC, i, j+1, R) + β(CC, i+1, j, R) )
                check, int = unique_integer(beta_ij_residual)
                @test check && int == 0
                @test real(beta_ij_residual) < tol
            end
        end
    end
end
