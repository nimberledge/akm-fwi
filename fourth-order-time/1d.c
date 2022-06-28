// pde = model.m * u.dt2 + model.damp * u.dt - u.dx2
u[t2][x + 2] = (r3*damp[x + 1]*u[t0][x + 2] + r4*r6 + r4*u[t0][x + 1] + r4*u[t0][x + 3] + r5*(-r2*r6 - r2*u[t1][x + 2]))
                /(r2*r5 + r3*damp[x + 1]);

// pde = model.m * u.dt2 + model.damp * u.dt - u.dx2
u[t2][x + 2] = (r3*damp[x + 1]*u[t0][x + 2] + r4*r6 + r4*u[t0][x + 1] + r4*u[t0][x + 3] + r5*(-r2*r6 - r2*u[t1][x + 2]))
                /(r2*r5 + r3*damp[x + 1]);

// pde = model.m * u.dt2 + model.damp*u.dt - 1/model.b * (model.b * u.dxc).dxc
u[t2][x + 2] = (r4*damp[x + 1]*u[t0][x + 2] + r5*(-r3*(-2.0F*u[t0][x + 2]) - r3*u[t1][x + 2]) + 
                (2.5e-1F*(-r6*(-u[t0][x] + u[t0][x + 2])*b[x + 1] + r6*(-u[t0][x + 2] + u[t0][x + 4])*b[x + 3]))/b[x + 2])
                /(r3*r5 + r4*damp[x + 1]);



// pde = model.m * u.dt2 + model.damp*u.dt - 1/model.b * (model.b * u.dx).dx
u[t2][x + 2] = (r4*damp[x + 1]*u[t0][x + 2] + r5*(-r3*(-2.0F*u[t0][x + 2]) - r3*u[t1][x + 2]) + 
                (r2*(-(r2*(-u[t0][x + 2]) + r2*u[t0][x + 3])*b[x + 2]) + r2*(r2*(-u[t0][x + 3]) + r2*u[t0][x + 4])*b[x + 3])/b[x + 2])
                /(r3*r5 + r4*damp[x + 1]);

// pde = model.m * u.dt2 + model.damp*u.dt - 1/model.b * (model.b * u.dxc).dxc 
// + s**2 / 12 * 1/model.b * (model.b * (1/model.m * 1/model.b * (model.b * u.dx).dxc).dxc).dxc
u[t2][x + 2] = (r4*damp[x + 1]*u[t0][x + 2] + r6*(r3*(2.0F*u[t0][x + 2] - u[t1][x + 2])) + r7*(-1.04166667e-2F*s*s*r2*r2*r2*((-r10*r5*r7
                + ((vp[x + 4]*vp[x + 4])*(-r8*b[x + 3] + (r2*(-u[t0][x + 5]) + r2*u[t0][x + 6])*b[x + 5]))/b[x + 4])*b[x + 3]
                - (r10*r5*r7 + (-vp[x]*vp[x]*(r9*b[x + 1] - (r2*u[t0][x] + r2*(-u[t0][x - 1]))*b[x - 1]))/b[x])*b[x + 1])
                + (r2*r2)*(2.5e-1F*(-(-u[t0][x] + u[t0][x + 2])*b[x + 1] + (-u[t0][x + 2] + u[t0][x + 4])*b[x + 3]))))
                /(r3*r6 + r4*damp[x + 1]);

// pde = model.m * u.dt2 + model.damp*u.dt - 1/model.b * (model.b * u.dx).dx + s**2 / 12 * 1/model.b * (model.b * (1/model.m * 1/model.b * (model.b * u.dx).dx).dx).dx
u[t2][x + 2] = (r4*damp[x + 1]*u[t0][x + 2] + r7*(-r3*(-2.0F*u[t0][x + 2]) - r3*u[t1][x + 2])
                + (-1.0F/12.0F*s*s*(r2*(-r5[x]) + r2*r5[x + 1]) + r2*(-(r2*(-u[t0][x + 2]) + r2*u[t0][x + 3])*b[x + 2])
                + r2*(r2*(-u[t0][x + 3]) + r2*u[t0][x + 4])*b[x + 3])/b[x + 2])
                /(r3*r7 + r4*damp[x + 1]);

// pde = model.m * u.dt2 + model.damp * u.dt - u.dx2 - 1/model.b * (model.b.dxc * u.dxc)
u[t2][x + 2] = (r4*damp[x + 1]*u[t0][x + 2] + r5*r7 + r5*u[t0][x + 1] + r5*u[t0][x + 3] + r6*(-r2*r7 - r2*u[t1][x + 2])
                + (2.5e-1F*(r3*r3)*(-b[x + 1] + b[x + 3])*(-u[t0][x + 1] + u[t0][x + 3]))/b[x + 2])
                /(r2*r6 + r4*damp[x + 1]);

// pde = model.m * u.dt2 + model.damp * u.dt - 1/model.b * (model.b * u.dxc).dxc
u[t2][x + 2] = (r4*damp[x + 1]*u[t0][x + 2] + r5*(-r3*(-2.0F*u[t0][x + 2]) - r3*u[t1][x + 2])
                + (2.5e-1F*(-r6*(-u[t0][x] + u[t0][x + 2])*b[x + 1] + r6*(-u[t0][x + 2] + u[t0][x + 4])*b[x + 3]))/b[x + 2])
                /(r3*r5 + r4*damp[x + 1])


// pde = model.m * u.dt2 + model.damp*u.dt - 1/model.b * (model.b * u.grad()).div() 
// Base Devito
u[t2][x + 2] = (r4*damp[x + 1]*u[t0][x + 2] + r5*(-r3*(-2.0F*u[t0][x + 2]) - r3*u[t1][x + 2])
                + (r2*(-(r2*(-u[t0][x + 2]) + r2*u[t0][x + 3])*b[x + 2]) + r2*(r2*(-u[t0][x + 3]) + r2*u[t0][x + 4])*b[x + 3])/b[x + 2])
                /(r3*r5 + r4*damp[x + 1]);
// This seems to do (x+2. x+3) and (x+3, x+4)

// pde = model.m * u.dt2 + model.damp*u.dt - 1/model.b * (model.b * u.grad()).div() 
// Changed to make all grad / div dxc
u[t2][x + 2] = (r4*damp[x + 1]*u[t0][x + 2] + r5*(-r3*(-2.0F*u[t0][x + 2]) - r3*u[t1][x + 2])
                + (5.0e-1F*(-r6*(-u[t0][x + 1] + u[t0][x + 3])*b[x + 2] + r6*(-u[t0][x + 2] + u[t0][x + 4])*b[x + 3]))/b[x + 2])
                /(r3*r5 + r4*damp[x + 1]); 
// This seems to do (x+1, x+3) and (x+2, x+4) which is strange

// model.m * u.dt2 + model.damp*u.dt - 1/model.b * (model.b * u.dxc).div()
// Changed to make all grad / div dxc
u[t2][x + 2] = (r4*damp[x + 1]*u[t0][x + 2] + r5*(-r3*(-2.0F*u[t0][x + 2]) - r3*u[t1][x + 2]) + (2.5e-1F*(-r6*(-u[t0][x] + u[t0][x + 2])*b[x + 1]
                + r6*(-u[t0][x + 2] + u[t0][x + 4])*b[x + 3]))/b[x + 2])
                /(r3*r5 + r4*damp[x + 1]);
// this does (x, x+2), (x+2, x+4)

// pde = model.m * u.dt2 + model.damp*u.dt - 1/model.b * (model.b * u.dxc).div()
// div and grad unchanged
u[t2][x + 2] = (r4*damp[x + 1]*u[t0][x + 2] + r5*(-r3*(-2.0F*u[t0][x + 2]) - r3*u[t1][x + 2])
                + (5.0e-1F*(-r6*(-u[t0][x + 1] + u[t0][x + 3])*b[x + 2] + r6*(-u[t0][x + 2] + u[t0][x + 4])*b[x + 3]))/b[x + 2])
                /(r3*r5 + r4*damp[x + 1]);
// does (x+1, x+3), (x+2, x+4)

// pde = model.m * u.dt2 + model.damp*u.dt - 1/model.b * (model.b * u.dxc).div()
// div and grad changed
u[t2][x + 2] = (r4*damp[x + 1]*u[t0][x + 2] + r5*(-r3*(-2.0F*u[t0][x + 2]) - r3*u[t1][x + 2])
                + (2.5e-1F*(-r6*(-u[t0][x] + u[t0][x + 2])*b[x + 1] + r6*(-u[t0][x + 2] + u[t0][x + 4])*b[x + 3]))/b[x + 2])
                /(r3*r5 + r4*damp[x + 1]);

// pde = model.m * u.dt2 + model.damp*u.dt - u.dx2 - 0.5*(u.dx * model.b.dx + u.dxl * model.b.dxl)
u[t2][x + 2] = ((r3*r3)*(5.0e-1F*(5.0e-1F*b[x] - 2.0F*b[x + 1] + 1.5F*b[x + 2])*(5.0e-1F*u[t0][x] - 2.0F*u[t0][x + 1]+ 1.5F*u[t0][x + 2]))
                + r4*damp[x + 1]*u[t0][x + 2] + r5*r7 + r5*u[t0][x + 1] + r5*u[t0][x + 3] + r6*(-r2*r7 - r2*u[t1][x + 2])
                + 5.0e-1F*(r3*(-b[x + 2]) + r3*b[x + 3])*(r3*(-u[t0][x + 2]) + r3*u[t0][x + 3]))
                /(r2*r6 + r4*damp[x + 1]);

// pde = model.m * u.dt2 + model.damp*u.dt - 0.5*(u.dx * model.b.dx + u.dxl * model.b.dxl)
u[t2][x + 2] = ((r3*r3)*(5.0e-1F*(5.0e-1F*b[x] - 2.0F*b[x + 1] + 1.5F*b[x + 2])*(5.0e-1F*u[t0][x] - 2.0F*u[t0][x + 1] + 1.5F*u[t0][x + 2]))
                + r4*damp[x + 1]*u[t0][x + 2] + r5*(-r2*(-2.0F*u[t0][x + 2])
                - r2*u[t1][x + 2]) + 5.0e-1F*(r3*(-b[x + 2]) + r3*b[x + 3])*(r3*(-u[t0][x + 2]) + r3*u[t0][x + 3]))
                /(r2*r5 + r4*damp[x + 1]);

// pde = model.m * u.dt2 + model.damp*u.dt - 0.5*(u.dxr * model.b.dxr + u.dxl * model.b.dxl)
u[t2][x + 2] = ((r3*r3)*(5.0e-1F*((5.0e-1F*b[x] - 2.0F*b[x + 1] + 1.5F*b[x + 2])*(5.0e-1F*u[t0][x] - 2.0F*u[t0][x + 1] + 1.5F*u[t0][x + 2])
                + (-1.5F*b[x + 2] + 2.0F*b[x + 3] - 5.0e-1F*b[x + 4])*(-1.5F*u[t0][x + 2] + 2.0F*u[t0][x + 3] - 5.0e-1F*u[t0][x + 4])))
                + r4*damp[x + 1]*u[t0][x + 2] + r5*(-r2*(-2.0F*u[t0][x + 2]) - r2*u[t1][x + 2]))
                /(r2*r5 + r4*damp[x + 1]);


// pde = model.m * u.dt2 - 1/model.b * (model.b * u.dxr).dxl
u[t2][x + 2] = r4*(dt*dt)*(((r2*r2)*(5.0e-1F*(-1.5F*u[t0][x] + 2.0F*u[t0][x + 1] - 5.0e-1F*u[t0][x + 2])*b[x] - 2.0F*(-1.5F*u[t0][x + 1] + 
				2.0F*u[t0][x + 2] - 5.0e-1F*u[t0][x + 3])*b[x + 1] + 1.5F*(-1.5F*u[t0][x + 2] + 2.0F*u[t0][x + 3] - 5.0e-1F*u[t0][x + 4])*b[x + 2]))/b[x + 2]
		 		+ (-r3*(-2.0F*u[t0][x + 2]) - r3*u[t1][x + 2])/r4);

// pde = model.m * u.dt2 - 1/model.b * (model.b * u.dx2 + model.b.dxc * u.dxc)
u[t2][x + 2] = r5*(dt*dt)*((2.5e-1F*(r2*r2)*(-b[x + 1] + b[x + 3])*(-u[t0][x + 1] + u[t0][x + 3]) + 
				(r3*r6 + r3*u[t0][x + 1] + r3*u[t0][x + 3])*b[x + 2])/b[x + 2]
			    + (-r4*r6 - r4*u[t1][x + 2])/r5);

// pde = model.m * u.dt2 - 1/model.b * (model.b * u.dxc).dxc
u[t2][x + 2] = r4*(dt*dt)*((2.5e-1F*(-r5*(-u[t0][x] + u[t0][x + 2])*b[x + 1] + r5*(-u[t0][x + 2] + u[t0][x + 4])*b[x + 3]))/b[x + 2]
			   + (-r3*(-2.0F*u[t0][x + 2]) - r3*u[t1][x + 2])/r4);
