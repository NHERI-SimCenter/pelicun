# Damage state probability validation test

Here we test whether we get the correct damage state probabilities for a single component with two damage states.
For such a component, assuming the EDP demand and the fragility curve capacities are all lognormal, there is a closed-form solution for the probability of each damage state.
We utilize those equations to ensure that the probabilities obtained from our Monte-Carlo sample are in line with our expectations.

## Equations

If $\mathrm{Y} \sim \textrm{LogNormal}(\delta, \beta)$, then  $\mathrm{X} = \log(\mathrm{Y}) \sim \textrm{Normal}(\mu, \sigma)$ with $\mu = \log(\delta)$ and $\sigma = \beta$.

```math
\begin{align*}
\mathrm{P}(\mathrm{DS}=0) &= 1 - \Phi\left(\frac{\log(\delta_D) - \log(\delta_{C1})}{\sqrt{\beta_{D}^2 + \beta_{C1}^2}}\right), \\
\mathrm{P}(\mathrm{DS}=1) &= \Phi\left(\frac{\log(\delta_D) - \log(\delta_{C1})}{\sqrt{\beta_D^2 + \beta_{C1}^2}}\right) - \Phi\left(\frac{\log(\delta_{D}) - \log(\delta_{C2})}{\sqrt{\beta_D^2 + \beta_{C2}^2}}\right), \\
\mathrm{P}(\mathrm{DS}=2) &= \Phi\left(\frac{\log(\delta_D) - \log(\delta_{C2})}{\sqrt{\beta_D^2 + \beta_{C2}^2}}\right), \\
\end{align*}
```
where $\Phi$ is the cumulative distribution function of the standard normal distribution, $\delta_{C1}$, $\delta_{C2}$, $\beta_{C1}$, $\beta_{C2}$ are the medians and dispersions of the fragility curve capacities, and $\delta_{D}$, $\beta_{D}$ is the median and dispersion of the EDP demand.

The equations inherently assume that the capacity RVs for the damage states are perfectly correlated, which is the case for sequential damage states.
