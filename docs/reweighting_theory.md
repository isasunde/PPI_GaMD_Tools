### Energetic reweighting
In conventional MD simulations, the distribution across a reaction coordinate, A, is sampled from the Boltzmann distribution. The GaMD boost smooths this distribution, giving a biased distribution $p^*(A)$ [1]:
$$p(A)=w\cdot p^*(A)$$

To recover the unbiased distribution, the sampled reaction coordinate is discretized into $M$ bins equally distributed along the reaction coordinate. The biased probability of bin $j$, denoted $p^*(A_j)$, is obtained from the number of simulation frames falling within that bin. The unbiased probability of each bin can then be estimated by reweighting the biased distribution according to the boost potential sampled within the same bin:
$$
    p(A_j) = p^*(A_j) \frac{\left\langle e^{\beta \Delta U} \right\rangle_j}{\sum_{i=1}^M p^*(A_i) \left\langle e^{\beta \Delta U} \right\rangle_i},\quad j=1,\dots,M
$$
where $\beta = \frac{1}{k_B T}$ and $\langle\exp{\beta\Delta U}\rangle_j$ is the ensemble averaged Boltzmann factor of $\Delta U$ of frames in bin $j$. However,  applying this Boltzmann factor directly will lead to high energetic noise based on the expected variance of the system. To minimize this, the factor is instead computed using a cumulant expansion of the 2nd order [2]:
$$
    \left\langle e^{\beta \Delta U} \right\rangle = \exp\left(\sum^\infty_{k=1}\frac{\beta^k}{k!}C_k\right) \approx \exp\left(\beta C_1 + \frac{\beta^2}{2} C_2\right)
$$

The reweighted free energy, also termed PMF is then obtained as:
$$
\begin{split}
    F(A)&= -k_BT\ln{p(A)}\\
    &=F^*(A)-\exp\left(C_1 + \frac{\beta}{2} C_2\right)+F_c
\end{split}
$$
where $C_1$ and $C_2$ are the conditional first and second cumulants of the boost potential in bin A, $F_c$ is an additive constant, and $F^*(A)$ is the modified free energy found from the simulation:
$$
    F^*(A)=-k_BT\ln{p^*(A)}
$$  

The quality of reweighting is therefore related to the extent to which the applied boost $\Delta U$ follows a Gaussian distribution. This can be quantified through the degree of anharmonicity:
$$
\begin{split}
    \gamma &=S_{\max}-S_{\Delta U}\\
    &= \frac{1}{2}\ln{2\pi \mathrm{e}\sigma_{\Delta U}^2}+\int_0^\infty p(\Delta U)\ln{p(\Delta U}) \mathrm{d}\Delta U
\end{split}
$$
where $S_{\max}$ is the max entropy of $\Delta U$. If $\gamma=0$ then $\Delta U$ follows a perfect Gaussian distribution given enough time to sample it, while as $\gamma$ increases, the accuracy of reweighting decreases [1].

### References
[1] Miao, Y. & McCammon, J. A. Gaussian Accelerated Molecular Dynamics: Theory, Implementation, and Applications. Annual reports in computational chemistry 13, 231. ISSN: 18755232 (2017).

[2] Miao, Y. et al. Improved Reweighting of Accelerated Molecular Dynamics Simulations for Free Energy Calculation. Journal of Chemical Theory and Computation 10, 2677–2689. ISSN: 15499626 (July 2014).