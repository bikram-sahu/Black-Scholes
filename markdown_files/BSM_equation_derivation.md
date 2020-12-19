## Deriving BSM equation

We first derive the Black-Scholes-Merton equation for an option on an underlying security. Consider a portfolio II of a long option with value $V$ and $\Delta$ amount of short stock,

$$
\Pi=V-\Delta S
$$
Using the assumption that during small time the number of share hold doesn't change i.e. $d \Delta=0$ we get,
$$
d \Pi=d V-\Delta d S
$$
Underlying stock follows geometric brownian motion given by
$$
d S=\mu S d t+\sigma S d W
$$
We assume that both $\mu$ and $\sigma$ are constant. Change in the portfolio value $d \Pi$ can be calculated using Ito's lemma given by
$$
d F=\frac{\partial F}{\partial S} d S+\frac{\partial F}{\partial t} d t+\frac{1}{2} \sigma^{2} S^{2} \frac{\partial^{2} F}{\partial S^{2}} d t
$$
Using this to calculate $d V,$ we get
$$
d V=\frac{\partial V}{\partial S} d S+\frac{\partial V}{\partial t} d t+\frac{1}{2} \sigma^{2} S^{2} \frac{\partial^{2} V}{\partial S^{2}} d t
$$

Substituting Eqn. 3 we get,
$$

d V=\frac{\partial V}{\partial S} \mu S d t+\frac{\partial V}{\partial S} \sigma S d W+\frac{\partial V}{\partial t} d t+\frac{1}{2} \sigma^{2} S^{2} \frac{\partial^{2} V}{\partial S^{2}} d t
$$

$$
d V=\left(\frac{\partial V}{\partial S} \mu S+\frac{\partial V}{\partial t}+\frac{1}{2} \sigma^{2} S^{2} \frac{\partial^{2} V}{\partial S^{2}}\right) d t+\frac{\partial V}{\partial S} \sigma S d W $$

$$
\Delta d S=\Delta \mu S d t+\Delta \sigma S d W
$$

Substituting above equations into Eqn. 2 we get,
$$
d \Pi=\left(\frac{\partial V}{\partial S} \mu S+\frac{\partial V}{\partial t}+\frac{1}{2} \sigma^{2} S^{2} \frac{\partial^{2} V}{\partial S^{2}}-\Delta \mu S\right) d t+\left(\frac{\partial V}{\partial S} \sigma S-\Delta \sigma S\right) d W
$$
since we want to hedge the risk in the portfolio, we need to eliminate the stochastic term $d W$ by setting the coefficient of it equal to $0 .$ This gives us the relation,
$$
\Delta=\frac{\partial V}{\partial S}
$$
We need $\Delta$ amount of shares to hedge the risk. Resubstituring the Eqn. 7 we get
$$
d \Pi=\left(\frac{\partial V}{\partial t}+\frac{1}{2} \sigma^{2} S^{2} \frac{\partial^{2} V}{\partial S^{2}}\right) d t
$$
since this 'hedged portfolio' is riskless and the drift of the stock $\mu$ is eliminated by delta hedging condition, it should grow at the risk free rate $r,$ hence the change in the portfolio must be equal to
$$
d \Pi=r \Pi d t
$$

This leads to the partial differential equation
$$
\frac{\partial V}{\partial t}+\frac{1}{2} \sigma^{2} S^{2} \frac{\partial^{2} V}{\partial S^{2}}=r \Pi
$$
Substituting the value of $\Pi$ from eqn. (1) and $\Delta$ from eqn. (7) we get
$$
\frac{\partial V}{\partial t}+\frac{1}{2} \sigma^{2} S^{2} \frac{\partial^{2} V}{\partial S^{2}}+r S \frac{\partial V}{\partial S}-r V=0
$$
This is the partial differential equation, which governs the price of an option. Note that this is not a stochastic differential equation! The stochastic part is eliminated by delta hedging. Which leaves us with a deterministic partial differential equation which predicts the price of an option deterministically! We made following assumptions while deriving the above equation.
