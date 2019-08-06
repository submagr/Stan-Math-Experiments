from scipy.integrate import ode
import numpy as np


def dyn(y, t, theta0, theta1):
    return [y[1], (-1)*(theta0/theta1)*np.sin(y[0])]


y0 = np.array([np.pi/4, 0.])
theta0 = (9.8, 1.)
ts = [0., 0.2, 0.3]
y1 = odeint(dyn, y0, ts, args=theta0)
print(y1)
