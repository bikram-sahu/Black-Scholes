**@author: Bikram (https://github.com/bikram-sahu/Black-Scholes)**

# The Black-Scholes option pricing formula for European options.

*The project is aimed at building intuition on Option Pricing and how it depends on various parameters. The plots come with interactive widgets.*

This contains:

  1. Analytical solution of Black-Scholes equation and sensitivity analysis of Option price to different parameters.
  2. Option Greeks: Delta, Gamma, Vega, Theta
  3. Numerical Solution to Balck-Scholes
	  * Monte Carlo
	  * Finite Difference


**The BSM equation:**

$$
\frac{\partial V}{\partial t}+\frac{1}{2} \sigma^{2} S^{2} \frac{\partial^{2} V}{\partial S^{2}}+r S \frac{\partial V}{\partial S}-r V=0
$$

This is the partial differential equation, which governs the price of an option.

**Analytical solution of BSM equation:**

The value of a call option $c\left(S_{0}, t\right)$ can be written in the simplified was a follows
$$
c\left(S_{0}, t\right)=S_{0} N\left(d_{1}\right)-K e^{r(T-t)} N\left(d_{2}\right)
$$
with,
$$
d_{1}=\frac{\log \left(S_{0} / K\right)+\left(r+\frac{1}{2} \sigma^{2}\right)(T-t)}{\sigma \sqrt{T-t}}
$$

$$
d_{2}=\frac{\log \left(S_{0} / K\right)+\left(r-\frac{1}{2} \sigma^{2}\right)(T-t)}{\sigma \sqrt{T-t}}=d_{1}-\sigma \sqrt{T-t}
$$
and $N$ is the CDF of standard normal distribution.
