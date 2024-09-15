<div style="text-align: center;">
    <strong style="font-size: 2em;">Math Overview</strong>
</div>

Disc galaxies have a negative exponential surface brightness profile as described by Equation 1, where '$a$' is the scale length of the disc. These galaxies are often reasonably smooth in the azimuthal coordinate '$\phi$', with low-order multiplicity '$m$' and can be described by Fourier Series. 

$$\Sigma(R) \propto \exp\left(-\frac{R}{a}\right)$$ 
**(1)**

In the radial '$R$' coordinate, Laguerre basis functions, as shown in Equation 2, are utilised as the exponential term closely approximates to the exponential profile of a typical disc galaxy (Equation 1). Here, in Equation 2, '$L_n^1$' is the associated Laguerre polynomial of order '$\alpha=1$' and degree '$n$', and '$a$' is the expansion scale length.

$$
G_n(R)=\frac{1}{a\sqrt{n+1}}~\exp\left(-\frac{R}{a}\right)~L_n^1\left(\frac{2R}{a}\right) \tag{2}
$$

Hence, Fourier-Laguerre basis functions can accurately model the surface brightness profile of discs in 2 dimensions. $G_n(R)\cos{m\phi}$ and $G_n(R)\sin{m\phi}$, known as Fourier-Laguerre functions, comprise of coefficients $\hat{c}_{mn}$ and $\hat{s}_{mn}$, that are naturally determined by the projection of the surface density, $\Sigma(R,\phi)$, into the basis functions. These coefficients are calculated as shown in Equation 3, where $(R_{xy},\phi_{xy})$ is the polar position of pixel $(x,y)$. It's worth noting here that discretion summations are used over continuous integrals as our model is accumulating discrete pixels in $(x,y)$. 

$$
\hat{c}_{mn} = \frac{1}{2\pi}\sum_x \sum_y \Sigma(R_{xy}, \phi_{xy})\cos\left(m\phi_{xy}\right) G_n(R_i) \\
\hat{s}_{mn} = \frac{1}{2\pi}\sum_x \sum_y \Sigma(R_{xy}, \phi_{xy})\sin\left(m\phi_{xy}\right) G_n(R_i) \tag{3}
$$
