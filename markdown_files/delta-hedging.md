
Let us now view the option price as a function of $S$ and $\sigma$ only.
A simple application of Taylor's Theorem says
r'''
C(S+\Delta S, \sigma+\Delta \sigma) & \approx C(S, \sigma)+\Delta S \frac{\partial C}{\partial S}+\frac{1}{2}(\Delta S)^{2} \frac{\partial^{2} C}{\partial S^{2}}+\Delta \sigma \frac{\partial C}{\partial \sigma}
'''
=C(S, \sigma)+\Delta S \delta+\frac{1}{2}(\Delta S)^{2} \Gamma+\Delta \sigma \text { vega. }

We therefore obtain
$$
\mathrm{P} \& \mathrm{~L} & \approx \delta \Delta S+\frac{\Gamma}{2}(\Delta S)^{2}+\text { vega } \Delta \sigma \\
=\text { delta } \mathrm{P} \& \mathrm{~L}+\text { gamma } \mathrm{P} \& \mathrm{~L}+\text { vega } \mathrm{P} \& \mathrm{~L}
$$
