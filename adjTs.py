import numpy as np


def pend_exact(ts, theta):
    g = theta[0]
    l = theta[1]
    y = np.empty((2, len(ts)))
    y[0, :] = (-np.pi/8)*np.sqrt(g/l)*(ts/l)*np.sin(np.sqrt(g/l)*ts),
    y[1, :] = (-np.pi/4)*np.sqrt(g/l)*np.sin(np.sqrt(g/l)*ts)
    return y


def dgs(ts, theta):
    g = theta[0]
    l = theta[1]
    return (-np.pi/8.)*(ts/g)*(np.sqrt(g/l))*np.sin(np.sqrt(g/l)*ts)

def dls(ts, theta):
    g = theta[0]
    l = theta[1]
    return (np.pi/8.)*(ts/l)*(np.sqrt(g/l))*np.in(np.sqrt(g/l)*ts)

if __name__ == "__main__":
    # print actual pendulum solutions at timesteps, given initial states
    # print actual differential of pendulum parameters at all the timesteps

    y0 = [np.pi/4, 0.]
    theta0 = [9.8, 1.]
    ts = np.arange(0, 10, 0.05)

    # Exact states at time steps
    ys = pend_exact(ts, theta0)

    # Exact gradient wrt to parameters at those steps
    dgs = pend_grad_g(ts, theta0[0])
    dls = pend_grad_l(ts, theta0[1])

