// pde = model.m * u.dt2 -   1/model.b *(model.b.dx*u.dx+model.b*u.dx2+model.b.dy*u.dy+model.b*u.dy2) + model.damp * u.dt
u[t2][x + 2][y + 2] = (r5*damp[x + 1][y + 1]*u[t0][x + 2][y + 2] + 
                      r6*(-r4*(-2.0F*u[t0][x + 2][y + 2]) - r4*u[t1][x + 2][y + 2]) 
                      + (
                          (r7 + 4.0e-4F*(u[t0][x + 1][y + 2] + u[t0][x + 3][y + 2]))*b[x + 2][y + 2] 
                        + (r7 + 4.0e-4F*(u[t0][x + 2][y + 1] + u[t0][x + 2][y + 3]))*b[x + 2][y + 2]
                        + 4.0e-4F*(
                                (r8 + b[x + 2][y + 3])*(r9 + u[t0][x + 2][y + 3]) + 
                                (r8 + b[x + 3][y + 2])*(r9 + u[t0][x + 3][y + 2]))
                        ) / b[x + 2][y + 2]) 
                        / (r4*r6 + r5*damp[x + 1][y + 1]);
// In the above, we have model.b.dx calculated as b_dx[x+2][y+2] = b[x+3][y+2] - b[x+2][y+2]
// Also u_dx[current][x+2][y+2] = u[current][x+3][y+2] - u[current][x+2][y+2]
// This isn't ideal, also it's taking damp[x+1][y+1] to update index [x+2][y+2] (?)

// Approach 2
// b_dx = model.b.dx(x0=x-x.spacing/2)
// b_dy = model.b.dy(x0=y-y.spacing/2)
// u_dx = u.dx(x0=x-x.spacing/2)
// u_dy = u.dy(x0=y-y.spacing/2)
u[t2][x + 2][y + 2] = (r5*damp[x + 1][y + 1]*u[t0][x + 2][y + 2] + 
                      r6*(-r4*(-2.0F*u[t0][x + 2][y + 2]) - r4*u[t1][x + 2][y + 2])
                      + (
                          (r7 + 4.0e-4F*(u[t0][x + 1][y + 2] + u[t0][x + 3][y + 2]))*b[x + 2][y + 2]
                        + (r7 + 4.0e-4F*(u[t0][x + 2][y + 1] + u[t0][x + 2][y + 3]))*b[x + 2][y + 2]
                        + 4.0e-4F*(
                            (-b[x + 1][y + 2] + b[x + 2][y + 2])*(-u[t0][x + 1][y + 2] + u[t0][x + 2][y + 2]) + 
                            (-b[x + 2][y + 1] + b[x + 2][y + 2])*(-u[t0][x + 2][y + 1] + u[t0][x + 2][y + 2])
                            )
                        ) /b[x + 2][y + 2])
                        / (r4*r6 + r5*damp[x + 1][y + 1]);

// In the above, we have model.b.dx calculated as b_dx[x+2][y+2] = b[x+2][y+2] - b[x+1][y+2]
// Also u_dx[current][x+2][y+2] = u[current][x+2][y+2] - u[current][x+1][y+2]
// This isn't ideal, also it's taking damp[x+1][y+1] to update index [x+2][y+2] (?)

// Approach 3
// b_dx = model.b.dx(x0=x-x.spacing)
// b_dy = model.b.dy(x0=y-y.spacing)
// u_dx = u.dx(x0=x-x.spacing)
// u_dy = u.dy(x0=y-y.spacing)
u[t2][x + 2][y + 2] = (r5*damp[x + 1][y + 1]*u[t0][x + 2][y + 2] + 
                      r6*(-r4*(-2.0F*u[t0][x + 2][y + 2]) - r4*u[t1][x + 2][y + 2])
                       + (
                           (r7 + 4.0e-4F*(u[t0][x + 1][y + 2] + u[t0][x + 3][y + 2]))*b[x + 2][y + 2] + 
                           (r7 + 4.0e-4F*(u[t0][x + 2][y + 1] + u[t0][x + 2][y + 3]))*b[x + 2][y + 2] + 
                           4.0e-4F*(
                               (-b[x + 1][y + 2] + b[x + 2][y + 2])*(-u[t0][x + 1][y + 2] + u[t0][x + 2][y + 2]) + 
                               (-b[x + 2][y + 1] + b[x + 2][y + 2])*(-u[t0][x + 2][y + 1] + u[t0][x + 2][y + 2])
                               )
                         ) / b[x + 2][y + 2])
                         / (r4*r6 + r5*damp[x + 1][y + 1]);

// Approach 4
// b_dx = 0.5 * (model.b.dx(x0=x-x.spacing) + model.b.dx(x0=x+x.spacing))
// b_dy = model.b.dy(x0=y-y.spacing)
// u_dx = u.dx(x0=x-x.spacing)
// u_dy = u.dy(x0=y-y.spacing)
u[t2][x + 2][y + 2] = (r5*damp[x + 1][y + 1]*u[t0][x + 2][y + 2] + 
                      r6*(-r4*(-2.0F*u[t0][x + 2][y + 2]) - r4*u[t1][x + 2][y + 2])
                    + (
                        (r7 + 4.0e-4F*(u[t0][x + 1][y + 2] + u[t0][x + 3][y + 2]))*b[x + 2][y + 2] + 
                        (r7 + 4.0e-4F*(u[t0][x + 2][y + 1] + u[t0][x + 2][y + 3]))*b[x + 2][y + 2] + 
                        4.0e-4F*(-b[x + 2][y + 1] + b[x + 2][y + 2])*(-u[t0][x + 2][y + 1] + u[t0][x + 2][y + 2])
                        + 1.99999991059303e-4F*(-u[t0][x + 1][y + 2] + u[t0][x + 2][y + 2])*
                        (-b[x + 1][y + 2] + b[x + 2][y + 2] - b[x + 3][y + 2] + b[x + 4][y + 2])
                      )
                      /b[x + 2][y + 2])
                      /(r4*r6 + r5*damp[x + 1][y + 1]);

// Approach 5
// b_dx = 0.5 * (model.b.dx(x0=x-x.spacing/2) + model.b.dx(x0=x+x.spacing/2))
// b_dy = model.b.dy(x0=y-y.spacing)
// u_dx = 0.5 * (u.dx(x0=x-x.spacing/2) + u.dx(x0=x+x.spacing/2))
// u_dy = u.dy(x0=y-y.spacing)
u[t2][x + 2][y + 2] = (r5*damp[x + 1][y + 1]*u[t0][x + 2][y + 2] + 
                      r6*(-r4*(-2.0F*u[t0][x + 2][y + 2]) - r4*u[t1][x + 2][y + 2])
                       + (
                           (r7 + 4.0e-4F*(u[t0][x + 1][y + 2] + u[t0][x + 3][y + 2]))*b[x + 2][y + 2] + 
                           (r7 + 4.0e-4F*(u[t0][x + 2][y + 1] + u[t0][x + 2][y + 3]))*b[x + 2][y + 2] + 
                           9.99999955296517e-5F*(-b[x + 1][y + 2] + b[x + 3][y + 2])*(-u[t0][x + 1][y + 2] + u[t0][x + 3][y + 2]) + 
                           4.0e-4F*(-b[x + 2][y + 1] + b[x + 2][y + 2])*(-u[t0][x + 2][y + 1] + u[t0][x + 2][y + 2])
                         )
                         /b[x + 2][y + 2])
                         /(r4*r6 + r5*damp[x + 1][y + 1]);

// Approach 6
// b_dx = 0.5 * (model.b.dx(x0=x-x.spacing/2) + model.b.dx(x0=x+x.spacing/2))
// b_dy = 0.5 * (model.b.dy(x0=y-y.spacing/2) + model.b.dy(x0=y+y.spacing/2))
// u_dx = 0.5 * (u.dx(x0=x-x.spacing/2) + u.dx(x0=x+x.spacing/2))
// u_dy = 0.5 * (u.dy(x0=y-y.spacing/2) + u.dy(x0=y+y.spacing/2))



// pde = (model.m * u.dt2 - 1/model.b *(b_dx*u_dx
//                                     +model.b*u.dx2
//                                     +b_dy*u_dy
//                                     +model.b*u.dy2) + model.damp * u.dt)

u[t2][x + 2][y + 2] = (r5*damp[x + 1][y + 1]*u[t0][x + 2][y + 2] + 
                      r6*(-r4*(-2.0F*u[t0][x + 2][y + 2]) - r4*u[t1][x + 2][y + 2])
                       + (
                           (r7 + 4.0e-4F*(u[t0][x + 1][y + 2] + u[t0][x + 3][y + 2]))*b[x + 2][y + 2] + 
                           (r7 + 4.0e-4F*(u[t0][x + 2][y + 1] + u[t0][x + 2][y + 3]))*b[x + 2][y + 2] + 
                           9.99999955296517e-5F*(
                               (-b[x + 1][y + 2] + b[x + 3][y + 2])*(-u[t0][x + 1][y + 2] + u[t0][x + 3][y + 2]) + 
                               (-b[x + 2][y + 1] + b[x + 2][y + 3])*(-u[t0][x + 2][y + 1] + u[t0][x + 2][y + 3])
                                                )
                         )
                         /b[x + 2][y + 2])
                         /(r4*r6 + r5*damp[x + 1][y + 1]);