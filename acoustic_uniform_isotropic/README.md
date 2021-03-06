# Acoustic Wave Equation through a uniform, isotropic medium

The equation for this case is 

$$
\begin{equation}
\frac{1}{c^2} \frac{\partial^2 u}{\partial t^2} - \nabla . \nabla u = s
\end{equation}
$$

We base this notebook on https://github.com/devitocodes/devito/blob/master/examples/seismic/tutorials/01\_modelling.ipynb but implement in 3D. As such, there are no plots. Sad indeed. The only thing tried so far is the subs keyword param in devito.Operator().

## C-Code 

- GeneratedCode1.c: C code generated by Operator() without any optimization flags
- GeneratedCode2.c: C code generated by "op = Operator(stencil + src\_term + rec\_term, subs=model.spacing\_map)"

## Insights
It is straightforward to set up our problem and export the propagator. We don't actually intend to perform FWI using Devito, we just want to extract the forward and adjoint propagators. In this notebook, we only look at the C-Code for the forward model in our simple case. We move on now to the acoustic isotropic wave equation with attenuation. 
