function λμσ(t)
    p_λ = (
        0.9460808392209046,
        -0.8862810594801697,
        5.8863742363252625,
        -5.530086126624541,
    )
    p_µ = (
        -0.8536605859604649,
        -0.1401260293039975,
        -0.18073245949192288,
        0.006758576358325964,
    )
    p_σ = (
        0.10313485732628469,
        0.10179855958129908,
        -0.0030421608024849664,
        -9.70205679092708e-5,
    )

    x = log(t)
    λ = evalpoly(inv(x), p_λ)
    µ = evalpoly(x, p_µ)
    σ = evalpoly(x, p_σ)

    return (λ, µ, σ)
end
