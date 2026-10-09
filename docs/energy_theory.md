### Binding energetics
From the resulting potential of mean force (PMF), the standard binding free energy can be calculated by integrating the Boltzmann-weighted PMF over the bound region of the coordinate space:
$$
\Delta G^\circ = -\Delta W_{3D} - RT \ln \left(\frac{V_b}{V_0}\right)
$$

where $V_b$ is the Boltzmann-weighted bound volume, $V_0$ is the standard-state volume, and $\Delta W_{3D}$ represents the depth of the PMF relative to the unbound region [3]. This expression assumes that the PMF is calculated over coordinates that define a physically meaningful volume, such as the Cartesian displacement of the ligand protein relative to the receptor protein in the x, y, and z directions. In this case, the integrated quantity has units of volume and can therefore be compared to the standard-state volume.

The bound volume is calculated as:
$$
V_b = \int_b e^{-\beta W(\mathbf{r})} d\mathbf{r}
$$

where $W(\mathbf{r})$ is the reweighted PMF at position $\mathbf{r}$, $\beta = 1/k_BT$, and $d\mathbf{r}$ is the volume element of the PMF coordinate space. 
In practice, the PMF is not continuous but discretized into bins along each coordinate. For a three-dimensional Cartesian PMF, each bin represents a small rectangular volume element in the coordinate space. If the bin widths along the three coordinates are $\Delta x$, $\Delta y$, and $\Delta z$, the volume of one bin is:
$$
\Delta V = \Delta x \Delta y \Delta z.
$$

As such, the bound volume can be approximated by:

$$
V_b= \int_b e^{-\beta W(\mathbf{r})} d\mathbf{r}\approx \sum_{\mathbf{r}_i \in b} e^{-\beta W(\mathbf{r}_i)} \Delta V
$$
where $\Delta V$ is the volume of each bin [1,2].

The PMF depth term, $\Delta W_{3D}$, accounts for the energetic offset of the unbound region and is calculated from the average Boltzmann weight over the unbound region:
$$
\begin{split}
\Delta W_{3D}
&= -RT \ln \left(
\frac{\int_u e^{-\beta W(\mathbf{r})} d\mathbf{r}}
{\int_u d\mathbf{r}}
\right) \\
&= -k_BT \ln \left(\frac{V_u}{V_{u,0}}\right)
\end{split}
$$

where $V_u$ is the Boltzmann-weighted unbound volume and $V_{u,0}$ is the geometric volume of the sampled unbound region. The unbound volume correction is included to account for the finite region sampled in the PMF and to reference the bound-state integral to the standard state. The resulting $\Delta G^\circ$ therefore depends on both the energetic preference for the bound state and the configurational volume available to the ligand protein in the bound region [1].

### References
[1] Miao, Y. & McCammon, J. A. Gaussian Accelerated Molecular Dynamics: Theory, Implementation, and Applications. Annual reports in computational chemistry 13, 231. ISSN: 18755232 (2017).

[2] Miao, Y. et al. Improved Reweighting of Accelerated Molecular Dynamics Simulations for Free Energy Calculation. Journal of Chemical Theory and Computation 10, 2677–2689. ISSN: 15499626 (July 2014).

[3] Doudou, S., Burton, N. A. & Henchman, R. H. Standard Free Energy of Binding from a One-Dimensional Potential of Mean Force. Journal of Chemical Theory and Computation 5, 909–918. ISSN: 15499618 (Apr. 2009).