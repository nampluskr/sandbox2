import numpy as np
from numba import njit, float64


E_CHARGE = 1.60217662e-19
E_MASS   = 9.10938356e-31


# @njit
def lorentz(t, u, E_func, B_func):
    x, y, z, vx, vy, vz = u
    E = E_func.E(x, y, z)
    B = B_func.B(x, y, z)
    vel = np.array([vx, vy, vz])
    vxB = np.cross(vel, B)
    acc = -(E_CHARGE / E_MASS) * (E + vxB)
    return np.array([vx, vy, vz, acc[0], acc[1], acc[2]])

# @njit
def rk4_step(f, u, t, dt, E_func, B_func):
    k1 = dt * f(t, u, E_func, B_func)
    k2 = dt * f(t + 0.5*dt, u + 0.5*k1, E_func, B_func)
    k3 = dt * f(t + 0.5*dt, u + 0.5*k2, E_func, B_func)
    k4 = dt * f(t + dt, u + k3, E_func, B_func)
    return u + (k1 + 2*k2 + 2*k3 + k4) / 6.0

# @njit
def rk4(f, u0, t, args=()):
    n_steps = t.size
    u0 = np.asarray(u0)
    u_dim = u0.shape[0]
    u = np.empty((n_steps, u_dim), dtype=np.float64)
    u[0] = u0
    for i in range(n_steps - 1):
        dt = t[i+1] - t[i]
        k1 = dt * f(t[i], u[i], *args)
        k2 = dt * f(t[i] + 0.5*dt, u[i] + 0.5*k1, *args)
        k3 = dt * f(t[i] + 0.5*dt, u[i] + 0.5*k2, *args)
        k4 = dt * f(t[i] + dt, u[i] + k3, *args)
        u[i+1] = u[i] + (k1 + 2*k2 + 2*k3 + k4) / 6.0
    return u


def make_E_func(target):
    @njit
    def E_func(x, y, z):
        return target.E(x, y, z)
    return E_func


def make_B_func(magnet):
    @njit
    def B_func(x, y, z):
        return magnet.B(x, y, z)
    return B_func


@njit
def is_valid(pos, vel):
    x, y, z = pos
    if z < 0.0: return False
    if z > 0.1: return False
    if abs(x) > 0.2 or abs(y) > 0.2: return False
    return True


# @njit
def trace_single(u0, t, E_func, B_func):
    result_list = [u0.copy()]
    current = u0
    for i in range(len(t) - 1):
        dt = t[i+1] - t[i]
        u_next = rk4_step(lorentz, current, t[i], dt, E_func, B_func)
        pos_next = u_next[:3]
        vel_next = u_next[3:]

        # if not is_valid(pos_next, vel_next):
        #     break

        result_list.append(u_next)
        current = u_next

    n = len(result_list)
    result = np.empty((n, 6), dtype=np.float64)
    for i in range(n):
        result[i] = result_list[i]
    return result


def trace_all(pos0_list, vel0_list, t, target, magnet):
    E_func = make_E_func(target)
    B_func = make_B_func(magnet)

    pos_list = []
    vel_list = []
    for pos0, vel0 in zip(pos0_list, vel0_list):
        u0 = np.concatenate([pos0, vel0])
        traj = trace_single(u0, t, E_func, B_func)
        pos_list.append(traj[:, :3])
        vel_list.append(traj[:, 3:])
    return pos_list, vel_list
