#!/usr/bin/env bash

real[] sho(real t, real[] y, real[] theta, real[] x_r, real[] x_i) {
 real dy1_dt = y[2];
 real dy2_dt = -y[1] - theta[1]*y[2];
 return {dy1_dt, dy2_ty};
}

# integrate_ode_bdf - backward distribution formulae, stiff

# integrate_ode_rk45 - backward, non-stiff
# real[] integrate_ode
