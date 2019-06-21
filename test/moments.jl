function test_moments(OS, tol)

    @testset "sum of products and moments" begin
        for i in 1:OS.n-1
            res = sum(expectation(OS, i, j) for j in 1:OS.n)
            check, int = unique_integer(res)
            @test check && int == 1
            @test real(res) - int < tol
        end

        res = sum(moment(OS, i, pow=2) for i in 1:OS.n)
        check, int = unique_integer(res)
        @test check && int == OS.n
        @test real(res) - int < tol
    end
end
