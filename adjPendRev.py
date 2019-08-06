from math import sin,cos
import numpy as np


def pend_dyn(y, t, theta):
    return np.array([y[1], -(theta[0]/theta[1])*sin(y[0])])


def pend_dyn_back(y, t, theta):
    return -1*pend_dyn(y, t, theta)


def grad_f_y(y, theta):
    return np.array([
        [0, -1],
        [(theta[0]/theta[1])*cos(y[0]), 0]
    ])


def grad_f_theta(y, theta):
    return np.array([
        [0, 0],
        [(1/theta[1])*sin(y[0]), (theta[0]/theta[1]**2)*sin(y[0])]
    ])


def grad_f_t():
    return np.array([
        0,
        0
    ])


def pend_aug_dyn(s, t, theta):
    y = s[0]
    ay = s[1]

    ret_y = pend_dyn_back(y, t, theta)
    ret_ay = -np.transpose(ay).dot(grad_f_y(y, theta))
    ret_atheta = -np.transpose(ay).dot(grad_f_theta(y, theta))
    ret_at = -np.transpose(ay).dot(grad_f_t())
    return [ret_y, ret_ay, ret_atheta, ret_at]


def integrate_euler(dyn, y0, theta0, t0, t1):
    tstep = 0.001
    t = t0
    while t < t1:
        yp = dyn(y0, t, theta0)
        for i in range(len(y0)):
            y0[i] += tstep*yp[i]
        t += tstep
    return y0


def forward(y0, theta0, t0, t1):
    y1 = integrate_euler(pend_dyn, y0, theta0, t0, t1)

    fy1 = pend_dyn(y1, t1, theta0)
    dl_dy1 = np.identity(y1.shape[0])
    dl_dt1 = np.transpose(dl_dy1).dot(fy1)

    dl_dtheta = np.zeros((y1.shape[0], len(theta0)))

    s0 = [y1, dl_dy1, dl_dtheta, -dl_dt1]
    s1 = integrate_euler(pend_aug_dyn, s0, theta0, 0, t1-t0)

    y0p, dl_dy0, dl_dtheta, dl_dt0 = s1

    print(y0p)
    print(dl_dt0)
    print(dl_dt1)
    print(dl_dtheta)
    return [dl_dy0, dl_dtheta, dl_dt1]


if __name__ == "__main__":
    y0 = np.array([np.pi/4, 0])
    print(y0)
    theta0 = np.array([9.8, 1])
    forward(y0, theta0, 0, 0.2)
