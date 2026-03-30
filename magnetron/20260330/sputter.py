import numpy as np
from numba import njit, float64
from numba.types import Tuple
from numba import prange
from numba.typed import List

# =============================================================================
# 1. Magnetic fields (magnets)
# =============================================================================

@njit(float64(float64, float64, float64, float64), cache=True)
def _f1(x, y, z, h):
    z_h = z - h
    n = np.sqrt(x*x + y*y + z_h*z_h)
    denom = n + y
    if denom == 0.0 or n == 0.0:
        return 0.0
    ratio = (n - y) / denom
    if ratio <= 0.0:
        return 0.0
    return np.log(ratio)

@njit(float64(float64, float64, float64, float64), cache=True)
def _f2(x, y, z, h):
    z_h = z - h
    n = np.sqrt(x*x + y*y + z_h*z_h)
    if n == 0.0:
        return 0.0
    if y == 0.0:
        sign = 1.0 if x * z_h >= 0 else -1.0
        return sign * (np.pi / 2.0)
    ratio = (x / y) * (z_h / n)
    return np.arctan(ratio)

@njit(Tuple((float64, float64, float64))(float64, float64, float64, float64[:]), cache=True)
def _B_single(x, y, z, params):
    x0, y0, z0, dx, dy, dz, scale = params
    x, y, z = x - x0, y - y0, z - z0

    bx = (
        _f1(dx - x, y, z, dz) + _f1(dx - x, dy - y, z, dz)
        - _f1(x, y, z, dz) - _f1(x, dy - y, z, dz)
        - _f1(dx - x, y, z, 0.0) - _f1(dx - x, dy - y, z, 0.0)
        + _f1(x, y, z, 0.0) + _f1(x, dy - y, z, 0.0)
    )
    Bx_val = -scale * bx * 0.5 / 4.0 / np.pi

    by = (
        _f1(dy - y, x, z, dz) + _f1(dy - y, dx - x, z, dz)
        - _f1(y, x, z, dz) - _f1(y, dx - x, z, dz)
        - _f1(dy - y, x, z, 0.0) - _f1(dy - y, dx - x, z, 0.0)
        + _f1(y, x, z, 0.0) + _f1(y, dx - x, z, 0.0)
    )
    By_val = -scale * by * 0.5 / 4.0 / np.pi

    bz = (
        _f2(y, dx - x, z, dz) + _f2(dy - y, dx - x, z, dz)
        + _f2(x, dy - y, z, dz) + _f2(dx - x, dy - y, z, dz)
        + _f2(dy - y, x, z, dz) + _f2(y, x, z, dz)
        + _f2(dx - x, y, z, dz) + _f2(x, y, z, dz)
        - _f2(y, dx - x, z, 0.0) - _f2(dy - y, dx - x, z, 0.0)
        - _f2(x, dy - y, z, 0.0) - _f2(dx - x, dy - y, z, 0.0)
        - _f2(dy - y, x, z, 0.0) - _f2(y, x, z, 0.0)
        - _f2(dx - x, y, z, 0.0) - _f2(x, y, z, 0.0)
    )
    Bz_val = -scale * bz / 4.0 / np.pi

    return Bx_val, By_val, Bz_val

@njit(float64[:, :](float64[:, :], float64[:]), cache=True, parallel=True)
def _compute_B_array(points, params):
    N = points.shape[0]
    result = np.empty((N, 3), dtype=float64)
    for i in prange(N):
        x, y, z = points[i, 0], points[i, 1], points[i, 2]
        Bx_val, By_val, Bz_val = _B_single(x, y, z, params)
        result[i, 0] = Bx_val
        result[i, 1] = By_val
        result[i, 2] = Bz_val
    return result


class Magnet:
    def __init__(self, x0, y0, z0, dx, dy, dz, scale):
        self.params = np.array([x0, y0, z0, dx, dy, dz, scale], dtype=np.float64)

    def B(self, x, y, z):
        x, y, z = np.broadcast_arrays(x, y, z)
        if np.isscalar(x):
            Bx, By, Bz = _B_single(x, y, z, self.params)
            return np.array([Bx, By, Bz])

        x_flat = np.asarray(x, dtype=np.float64).ravel()
        y_flat = np.asarray(y, dtype=np.float64).ravel()
        z_flat = np.asarray(z, dtype=np.float64).ravel()
        points = np.column_stack((x_flat, y_flat, z_flat))
        B_array = _compute_B_array(points, self.params)
        return B_array.reshape(*x.shape, 3)

    def Bx(self, x, y, z):
        return self.B(x, y, z)[..., 0]

    def By(self, x, y, z):
        return self.B(x, y, z)[..., 1]

    def Bz(self, x, y, z):
        return self.B(x, y, z)[..., 2]

    def lorentz_force(self, x, y, z, vx, vy, vz, q=1.0):
        x, y, z = np.broadcast_arrays(x, y, z)
        vx, vy, vz = np.broadcast_arrays(vx, vy, vz)
        if np.isscalar(x):
            Bx, By, Bz = _B_single(x, y, z, self.params)
            fx = q * (vy * Bz - vz * By)
            fy = q * (vz * Bx - vx * Bz)
            fz = q * (vx * By - vy * Bx)
            return np.array([fx, fy, fz])

        pos = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
        vel = np.column_stack((vx.ravel(), vy.ravel(), vz.ravel()))
        B_array = _compute_B_array(pos, self.params)
        v = vel
        B = B_array
        F = np.empty_like(B)
        for i in range(F.shape[0]):
            F[i, 0] = q * (v[i, 1]*B[i, 2] - v[i, 2]*B[i, 1])
            F[i, 1] = q * (v[i, 2]*B[i, 0] - v[i, 0]*B[i, 2])
            F[i, 2] = q * (v[i, 0]*B[i, 1] - v[i, 1]*B[i, 0])
        return F.reshape(*x.shape, 3)


@njit
def _compute_total_B_scalar(x, y, z, params_list):
    Bx = 0.0
    By = 0.0
    Bz = 0.0
    for params in params_list:
        Bx_i, By_i, Bz_i = _B_single(x, y, z, params)
        Bx += Bx_i
        By += By_i
        Bz += Bz_i
    return Bx, By, Bz

@njit(parallel=True)
def _compute_total_B_array(points, params_list):
    N = points.shape[0]
    result = np.empty((N, 3), dtype=float64)
    for i in prange(N):
        x, y, z = points[i, 0], points[i, 1], points[i, 2]
        Bx, By, Bz = _compute_total_B_scalar(x, y, z, params_list)
        result[i, 0] = Bx
        result[i, 1] = By
        result[i, 2] = Bz
    return result


class MagnetPack:
    def __init__(self, *magnets):
        if len(magnets) == 0:
            raise ValueError("적어도 하나의 Magnet이 필요합니다.")

        self.magnets = magnets
        self._params_list = List()
        for m in magnets:
            self._params_list.append(m.params)

    def B(self, x, y, z):
        x, y, z = np.broadcast_arrays(x, y, z)

        if np.isscalar(x):
            Bx, By, Bz = _compute_total_B_scalar(x, y, z, self._params_list)
            return np.array([Bx, By, Bz])

        x_flat = x.ravel().astype(np.float64)
        y_flat = y.ravel().astype(np.float64)
        z_flat = z.ravel().astype(np.float64)
        points = np.column_stack((x_flat, y_flat, z_flat))

        B_total = _compute_total_B_array(points, self._params_list)
        return B_total.reshape(*x.shape, 3)

    def Bx(self, x, y, z):
        return self.B(x, y, z)[..., 0]

    def By(self, x, y, z):
        return self.B(x, y, z)[..., 1]

    def Bz(self, x, y, z):
        return self.B(x, y, z)[..., 2]

    def lorentz_force(self, x, y, z, vx, vy, vz, q=1.0):
        x, y, z = np.broadcast_arrays(x, y, z)
        vx, vy, vz = np.broadcast_arrays(vx, vy, vz)

        B_total = self.B(x, y, z)
        v = np.stack([vx, vy, vz], axis=-1)

        # F = q * (v × B)
        F = q * np.cross(v, B_total)
        return F

# =============================================================================
# 2. Electric field (Target)
# =============================================================================

@njit(float64[:, :](float64[:], float64[:], float64[:], float64, float64), cache=True, parallel=True)
def _compute_E_array(x, y, z, voltage, sheath):
    N = x.size
    result = np.empty((N, 3), dtype=float64)
    for i in prange(N):
        zi = z[i]
        ez = 0.0
        if zi < sheath:
            ez = 2.0 * voltage * (sheath - zi) / (sheath * sheath)
            ez = min(ez, 0.0)
        result[i, 0] = 0.0  # Ex
        result[i, 1] = 0.0  # Ey
        result[i, 2] = ez   # Ez
    return result

@njit(float64[:](float64[:], float64[:], float64[:], float64, float64), cache=True, parallel=True)
def _compute_potential_array(x, y, z, voltage, sheath):
    N = x.size
    phi = np.empty(N, dtype=float64)
    for i in prange(N):
        zi = z[i]
        if zi < sheath:
            phi[i] = -0.5 * voltage * (zi - sheath)**2 / (sheath * sheath)
        else:
            phi[i] = 0.0
    return phi

class Target:
    def __init__(self, voltage, sheath):
        self.voltage = float(voltage)
        self.sheath = float(sheath)

    def E(self, x, y, z):
        x, y, z = np.broadcast_arrays(x, y, z)
        if np.isscalar(x):
            ez = 0.0
            if z < self.sheath:
                ez = min(2.0 * self.voltage * (self.sheath - z) / self.sheath**2, 0.0)
            return np.array([0.0, 0.0, ez])

        x_flat = np.asarray(x, dtype=np.float64).ravel()
        y_flat = np.asarray(y, dtype=np.float64).ravel()
        z_flat = np.asarray(z, dtype=np.float64).ravel()
        E_array = _compute_E_array(x_flat, y_flat, z_flat, self.voltage, self.sheath)
        return E_array.reshape(*x.shape, 3)

    def Ex(self, x, y, z):
        return self.E(x, y, z)[..., 0]

    def Ey(self, x, y, z):
        return self.E(x, y, z)[..., 1]

    def Ez(self, x, y, z):
        return self.E(x, y, z)[..., 2]

    def potential(self, x, y, z):
        x, y, z = np.broadcast_arrays(x, y, z)
        if np.isscalar(x):
            return -0.5 * self.voltage * (z - self.sheath)**2 / self.sheath**2 if z < self.sheath else 0.0

        x_flat = np.asarray(x, dtype=np.float64).ravel()
        y_flat = np.asarray(y, dtype=np.float64).ravel()
        z_flat = np.asarray(z, dtype=np.float64).ravel()
        phi = _compute_potential_array(x_flat, y_flat, z_flat, self.voltage, self.sheath)
        return phi.reshape(x.shape)

    def lorentz_force(self, x, y, z, q=1.0):
        E_vec = self.E(x, y, z)
        return q * E_vec
