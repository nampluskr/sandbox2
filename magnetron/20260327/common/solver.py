import numpy as np
from scipy.integrate import odeint


E_CHARGE = -1.602e-19   # [C]
E_MASS   = 9.109e-31    # [kg]

def lorentz(t, u, Efield, Bfield):
    pos = u[:3]             # [x, y, z]
    vel = u[3:]             # [vx, vy, vz]
    E = Efield.E(*pos)
    B = Bfield.B(*pos)
    acc = (E_CHARGE / E_MASS) * (E + np.cross(vel, B))
    return np.concatenate([vel, acc])   # du/dt = [v, a] = [v, (q/m)(E + v x B)]


# @njit
def rk4(eqn, u0, t, args=()):
    u0 = np.asarray(u0, dtype=np.float64)
    u = np.empty((t.size, *u0.shape), dtype=np.float64)
    u[0] = u0

    for i in range(t.size - 1):
        h = t[i+1] - t[i]
        k1 = h * eqn(t[i], u[i], *args)
        k2 = h * eqn(t[i] + h*0.5, u[i] + k1*0.5, *args)
        k3 = h * eqn(t[i] + h*0.5, u[i] + k2*0.5, *args)
        k4 = h * eqn(t[i] + h, u[i] + k3, *args)
        u[i+1] = u[i] + (k1 + 2*k2 + 2*k3 + k4) / 6.0
    return u


def trace_all(pos0, vel0, t, args=()):
    u0 = np.concatenate([pos0, vel0])
    sol = rk4(lorentz, u0, t, args)
    pos, vel = sol[:, :3], sol[:, 3:]
    return pos, vel


def trace_next(pos0, vel0, t0, dt, args=()):
    u0 = np.concatenate([pos0, vel0])
    t = np.array([t0, t0 + dt])
    sol = rk4(lorentz, u0, t, args)
    pos, vel = sol[-1, :3], sol[-1, 3:]
    return pos, vel


def trace(pos0, vel0, t, args=()):
    pos_list = [pos0]
    vel_list = [vel0]

    pos, vel = pos0, vel0
    for i in range(t.size - 1):
        dt = t[i+1] - t[i]
        pos, vel = trace_next(pos, vel, t[i], dt, args)
        pos_list.append(pos)
        vel_list.append(vel)
    return np.array(pos_list), np.array(vel_list)

