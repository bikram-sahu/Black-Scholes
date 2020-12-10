# Monte Carlo Method

We use the following alorithm to price an option using Monte Carlo,

1. Simulate the stock price under the risk neutral measure for the the desired time.
2. Calculate the payoff
3. Repeat step 1 and 2 for many times
4. Calculate the average payoff
5. Take the present value of the average payoff

To perform the first step we assume the following time evolution for the stock price

$$
d S_{t}=r S_{t} d t+\sigma S_{t} d Z_{t}
$$

Which can be descretized using Euler method as follows

$$
S_{t}=S_{t-\Delta t} \exp \left(\left(r-\frac{1}{2} \sigma^{2}\right) \Delta t+\sigma \sqrt{\Delta t} z_{t}\right)
$$

$$
\log S_{t}=\log S_{t-\Delta t}+\left(r-\frac{1}{2} \sigma^{2}\right) \Delta t+\sigma \sqrt{\Delta t} z_{t}
$$


The payoff depends upon the type of option in hand. Given the payoff as a function of stock price, the price of an option is calculated using

$$
\text { option price }=e^{-r(T-t)} \tilde{E}[\text { Payoff }(S)]
$$

