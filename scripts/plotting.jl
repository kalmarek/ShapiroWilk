using Plots

function plot_data(data, full_range, fit_range; label::String)
    test_range = setdiff(full_range, fit_range)
    plt = scatter!(fit_range, data[fit_range], opacity=0.6, label="fit $label")
    scatter!(test_range, data[test_range], opcity=0.6, label="test $label")
    return plt
end

function plot_residuals(predict, y, full_range, fit_range)
    test_range = setdiff(full_range, fit_range)
    res_fit  = (predict[fit_range] .- y[fit_range])./predict[fit_range]
    res_fit = [abs(p) < 0.005 ? NaN : r for (r,p) in zip(res_fit, predict[fit_range])]

    res_test = (predict[test_range] .- y[test_range])./predict[test_range]

    µ = round(mean(res_fit),digits=3)
    σ = round(std(res_fit), digits=3)
    plt = scatter!(fit_range, res_fit,
        opcity=0.2,
        label="fit residuals (µ, σ) = $(µ), $(σ))"
        )

    µ = round(mean(res_test),digits=3)
    σ = round(std(res_test), digits=3)
    scatter!(test_range, res_test,
        opcity=0.2,
        label="test residuals (µ, σ) = $(µ), $(σ))"
        )
    return plt
end

function fit_and_plot(y, full_range, fit_range, model; p0, lb=fill(-Inf, length(p0)), ub=fill(Inf, length(p0)), seed=1, label)
    Random.seed!(seed)

    fit = curve_fit(
        model,
        fit_range,
        y[fit_range],
        p0,
        lower=lb,
        upper=ub,
    )

    m = coef(fit)
    if true
        sigma = stderror(fit)
        margin_of_error = margin_error(fit, 0.05)
        confidence_inter = confidence_interval(fit, 0.05)
        # @info "inverse fit for Box-Cox exponent parameters:" m sigma margin_of_error confidence_inter
        @info "Model parameters:\n $m\n $sigma\n $confidence_inter"
    end

    data_plot = let data_plot = plot()
        plot_data(y, full_range, fit_range, label=label)
        # plot!(full_range, t->model(t, m), label="$model fit", width=2, legend=:top)
        plot!(7:250, t->model(t, m), label="$model fit", width=2, legend=:top)
    end

    res_plot = let res_plot = plot(legend=:bottom)
        predict = [model(t, m) for t in 1:length(y)]
        @info "predict residual: $(norm(predict[full_range] - y[full_range], 2))"
        plot_residuals(predict, y, full_range, fit_range)
        res_plot
    end

    plt = plot(data_plot, res_plot,
        layout=layout = grid(2, 1, heights=[0.7, 0.3]),
        size=(1200,1200),
    )
    display(plt)
    return m
end
