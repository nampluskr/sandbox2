import numpy as np
from itertools import product
from numba import njit

@njit
def _compute_fields(x, y, z, corners, scale):
    n = x.size
    bx, by, bz = np.zeros(n), np.zeros(n), np.zeros(n)
    eps = 1e-16

    for idx in range(n):
        xi, yi, zi = x[idx], y[idx], z[idx]
        bx_, by_, bz_ = 0.0, 0.0, 0.0

        for i in range(8):
            sign = corners[i, 0]
            dx = xi - corners[i, 1]
            dy = yi - corners[i, 2]
            dz = zi - corners[i, 3]
            r = np.sqrt(dx*dx + dy*dy + dz*dz)

            bx_ += sign * np.log(dy + r + eps)
            by_ += sign * np.log(dx + r + eps)
            bz_ += sign * np.arctan2(dx*dy, dz * r + eps)

        bx[idx] = -bx_ * scale
        by[idx] = -by_ * scale
        bz[idx] =  bz_ * scale
    return bx, by, bz


class Magnet:
    def __init__(self, corner, size, Br=1.4):
        corner = np.array(corner, dtype=np.float64)
        size = np.array(size, dtype=np.float64)

        self.center = corner + size / 2
        self.x0, self.y0, self.z0 = self.center
        self.dx, self.dy, self.dz = size
        self.scale = Br / 4.0 / np.pi
        self.corners = np.empty((8, 4), dtype=np.float64)

        for idx, (i, j, k) in enumerate(product((-1, 1), (-1, 1), (-1, 1))):
            self.corners[idx, 0] = i * j * k
            self.corners[idx, 1] = self.x0 + i * self.dx / 2
            self.corners[idx, 2] = self.y0 + j * self.dy / 2
            self.corners[idx, 3] = self.z0 + k * self.dz / 2

    def Bx(self, x, y, z):
        x, y, z = np.broadcast_arrays(x, y, z)
        fields = _compute_fields(x.ravel(), y.ravel(), z.ravel(), self.corners, self.scale)
        return fields[0].reshape(x.shape)

    def By(self, x, y, z):
        x, y, z = np.broadcast_arrays(x, y, z)
        fields = _compute_fields(x.ravel(), y.ravel(), z.ravel(), self.corners, self.scale)
        return fields[1].reshape(x.shape)

    def Bz(self, x, y, z):
        x, y, z = np.broadcast_arrays(x, y, z)
        fields = _compute_fields(x.ravel(), y.ravel(), z.ravel(), self.corners, self.scale)
        return fields[2].reshape(x.shape)

    def B(self, x, y, z):
        x, y, z = np.broadcast_arrays(x, y, z)
        fields = _compute_fields(x.ravel(), y.ravel(), z.ravel(), self.corners, self.scale)
        return np.stack(fields, axis=-1).reshape(*x.shape, 3)


class MagnetPack:
    def __init__(self, *magnets):
        self.magnets = magnets

    def B(self, x, y, z):
        x, y, z = np.broadcast_arrays(x, y, z)
        B = np.zeros((*x.shape, 3))
        for magnet in self.magnets:
            B += magnet.B(x, y, z)
        return B

    def Bx(self, x, y, z):
        return self.B(x, y, z)[..., 0]

    def By(self, x, y, z):
        return self.B(x, y, z)[..., 1]

    def Bz(self, x, y, z):
        return self.B(x, y, z)[..., 2]
    

class Target:
    def __init__(self, voltage, sheath):
        self.voltage = voltage
        self.sheath = sheath

    def Ex(self, x, y, z):
        return np.zeros_like(z)

    def Ey(self, x, y, z):
        return np.zeros_like(z)

    def Ez(self, x, y, z):
        return np.minimum(2 * self.voltage * (self.sheath - z) / self.sheath**2, 0.0)

    def E(self, x, y, z):
        x, y, z = np.broadcast_arrays(x, y, z)
        return np.stack([self.Ex(x, y, z), self.Ey(x, y, z), self.Ez(x, y, z)], axis=-1)
    
    def potential(self, x, y, z):
        if z < self.sheath:
            return -0.5 * self.voltage*(z - self.sheath)**2 / self.sheath**2
        else:
            return 0.0
