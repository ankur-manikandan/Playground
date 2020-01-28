using Plots

xGrid = 0:0.1:5
G(x) = 1/2*x^2-2*x
g(x)=x-2

plot(xGrid, G.(xGrid), label="G(x)", color=:green)
plot!(xGrid, g.(xGrid), label="g(x)", color=:blue)
title!("Plot of G(x)= \frac{1}{2}xˆ2-2x and it’s derivative")
