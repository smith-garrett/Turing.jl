using Turing

@model function gdemo(x)
    m ~ Normal(0, 1)
    x ~ Normal(m, 1)
    return m
end

sample(gdemo([0.2]), ABC(1e-3), 1000)
