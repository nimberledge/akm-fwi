// pde = u.dt2 - u.dx(x0=x+x.spacing/2)
u[t2][x + 2][y + 2] = (dt*dt)*(-r4*(-2.0F*u[t0][x + 2][y + 2]) - 
                    r4*u[t1][x + 2][y + 2] + 
                    2.0e-2F*(-u[t0][x + 2][y + 2] + u[t0][x + 3][y + 2]));


// pde = u.dt2 - u.dx(x0=x-x.spacing/2)
u[t2][x + 2][y + 2] = (dt*dt)*(-r4*(-2.0F*u[t0][x + 2][y + 2]) - 
                    r4*u[t1][x + 2][y + 2] + 
                    2.0e-2F*(-u[t0][x + 1][y + 2] + u[t0][x + 2][y + 2]));

// pde = u.dt2 - u.dx
u[t2][x + 2][y + 2] = (dt*dt)*(-r4*(-2.0F*u[t0][x + 2][y + 2]) - 
                    r4*u[t1][x + 2][y + 2] + 
                    2.0e-2F*(-u[t0][x + 2][y + 2] + u[t0][x + 3][y + 2]));


// pde = u.dt2 - u.dx(x0=x-x.spacing)
u[t2][x + 2][y + 2] = (dt*dt)*(-r4*(-2.0F*u[t0][x + 2][y + 2]) - 
                    r4*u[t1][x + 2][y + 2] + 
                    2.0e-2F*(-u[t0][x + 1][y + 2] + u[t0][x + 2][y + 2]));

// pde = u.dt2 - 0.5 * (u.dx(x0=x-x.spacing/2) + u.dx(x0=x+x.spacing/2))
u[t2][x + 2][y + 2] = (dt*dt)*(-r4*(-2.0F*u[t0][x + 2][y + 2]) - 
                    r4*u[t1][x + 2][y + 2] + 
                    9.99999977648258e-3F*(-u[t0][x + 1][y + 2] + u[t0][x + 3][y + 2]));



// pde = u.dt2 - u.dx.dx
u[t2][x + 2][y + 2] = (dt*dt)*(-r4*(-2.0F*u[t0][x + 2][y + 2]) - 
                    r4*u[t1][x + 2][y + 2] + 
                    4.0e-4F*(u[t0][x + 2][y + 2] + u[t0][x + 4][y + 2]) - 
                    8.0e-4F*u[t0][x + 3][y + 2]);

// pde = u.dt2 - u.dx2
u[t2][x + 2][y + 2] = (dt*dt)*(-r4*(-2.0F*u[t0][x + 2][y + 2]) - 
                    r4*u[t1][x + 2][y + 2] + 
                    4.0e-4F*(u[t0][x + 1][y + 2] + u[t0][x + 3][y + 2]) - 
                    7.9999998e-4F*u[t0][x + 2][y + 2]);


// pde = u.dt2 - u.dx(x0=x+x.spacing/2).dx(x0=x-x.spacing/2)
u[t2][x + 2][y + 2] = (dt*dt)*(-r4*(-2.0F*u[t0][x + 2][y + 2]) - 
                    r4*u[t1][x + 2][y + 2] + 
                    4.0e-4F*(u[t0][x + 1][y + 2] + u[t0][x + 3][y + 2]) - 
                    8.0e-4F*u[t0][x + 2][y + 2]);