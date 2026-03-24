from numba import jit
import numpy as np

def rk4_v1(f, u0, t, args=()):
    u = np.array([u0]*t.size, dtype=float)
    for i in range(t.size-1):
        h = t[i+1] - t[i]
        k1 = h*f(u[i], t[i], *args)
        k2 = h*f(u[i] + k1/2, t[i] + h/2, *args)
        k3 = h*f(u[i] + k2/2, t[i] + h/2, *args)
        k4 = h*f(u[i] + k3, t[i] + h, *args)
        u[i+1] = u[i] + (k1 + 2*(k2 + k3) + k4)/6
    return u

@jit
def rk4_v2(f, u0, t, args=()):
    n = t.size
    u0 = np.asarray(u0)
    u = np.empty((n, *u0.shape), dtype=np.float64)
    u[0] = u0

    for i in range(n-1):
        h = t[i+1] - t[i]
        k1 = h * np.asarray(f(u[i], t[i], *args))
        k2 = h * np.asarray(f(u[i] + k1/2, t[i] + h/2, *args))
        k3 = h * np.asarray(f(u[i] + k2/2, t[i] + h/2, *args))
        k4 = h * np.asarray(f(u[i] + k3, t[i] + h, *args))
        u[i+1] = u[i] + (k1 + 2*k2 + 2*k3 + k4) / 6
    return u
