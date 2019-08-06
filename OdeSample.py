import numpy as np
from scipy.integrate import odeint


def pend(y, t, b, c):
    theta, omega = y
    dydt = [omega, -c*np.sin(theta)]
    return dydt

b = 0.25
c = 5.0
y0 = [0.71552836, -0.69021702]
t = [0.2, 0]
sol = odeint(pend, y0, t, args=(b,c))
print(sol)
