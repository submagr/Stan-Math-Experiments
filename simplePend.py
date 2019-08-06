import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def pend(y, t, c):
    theta, omega = y
    return [omega, -c*np.sin(theta)]

y0 = [np.pi/2-0.1, 0.0]
c = 5.0

t = np.linspace(0, 10, 101)
sol = odeint(pend, y0, t, args=(c,))

plt.plot(t, sol[:, 0], label="theta(t)")
plt.plot(t, sol[:, 1], label="omega(t)")
plt.legend()
plt.xlabel('t')
plt.show()