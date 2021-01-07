abstract type OrderStatistic{T} <: AbstractVector{T} end

Statistics.var(OS::OrderStatistic, i::Int) =
    expectation(OS,i,i) - expectation(OS,i)^2

Statistics.cov(OS::OrderStatistic, i::Int, j::Int) =
    expectation(OS,i,j) - expectation(OS,i)*expectation(OS,j)

Statistics.cov(OS::OrderStatistic) =
    [cov(OS, i, j) for i in 1:length(OS), j in 1:length(OS)]
