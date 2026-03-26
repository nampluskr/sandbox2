import numpy as np
from itertools import product
from numba import njit


class MagnetV1:
    def __init__(self, corner, size, Br=1.4):
        self.x0, self.y0, self.z0 = corner
        self.dx, self.dy, self.dz = size
        self.scale = Br / 4 / np.pi
        self.eps = 1e-16

    def Bx(self, x, y, z):
        x, y, z = np.broadcast_arrays(x, y, z)
        Bx = np.zeros_like(x, dtype=float)
        for i, j, k in product((-1, 1), (-1, 1), (-1, 1)):
            X = x - (self.x0 + (i + 1) * self.dx / 2)
            Y = y - (self.y0 + (j + 1) * self.dy / 2)
            Z = z - (self.z0 + (k + 1) * self.dz / 2)
            R = np.sqrt(X**2 + Y**2 + Z**2)
            Bx += i * j * k * np.log(Y + R + self.eps)
        return -Bx * self.scale

    def By(self, x, y, z):
        x, y, z = np.broadcast_arrays(x, y, z)
        By = np.zeros_like(x, dtype=float)
        for i, j, k in product((-1, 1), (-1, 1), (-1, 1)):
            X = x - (self.x0 + (i + 1) * self.dx / 2)
            Y = y - (self.y0 + (j + 1) * self.dy / 2)
            Z = z - (self.z0 + (k + 1) * self.dz / 2)
            R = np.sqrt(X**2 + Y**2 + Z**2)
            By += i * j * k * np.log(X + R + self.eps)
        return -By * self.scale

    def Bz(self, x, y, z):
        x, y, z = np.broadcast_arrays(x, y, z)
        Bz = np.zeros_like(x, dtype=float)
        for i, j, k in product((-1, 1), (-1, 1), (-1, 1)):
            X = x - (self.x0 + (i + 1) * self.dx / 2)
            Y = y - (self.y0 + (j + 1) * self.dy / 2)
            Z = z - (self.z0 + (k + 1) * self.dz / 2)
            R = np.sqrt(X**2 + Y**2 + Z**2)
            Bz += i * j * k * np.arctan2(X * Y, Z * R + self.eps)
        return Bz * self.scale

    def B(self, x, y, z):
        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
        Bx = self.Bx(x, y, z)
        By = self.By(x, y, z)
        Bz = self.Bz(x, y, z)
        return np.stack([Bx, By, Bz], axis=-1)


class MagnetV2:
    def __init__(self, corner, size, Br=1.4):
        corner = np.array(corner, dtype=np.float64)
        size = np.array(size, dtype=np.float64)
        self.center = corner + size / 2
        self.cx, self.cy, self.cz = self.center
        self.dx, self.dy, self.dz = size

        corners = []
        signs = []
        for i, j, k in product((-1, 1), (-1, 1), (-1, 1)):
            x_ = self.cx + i * self.dx / 2
            y_ = self.cy + j * self.dy / 2
            z_ = self.cz + k * self.dz / 2
            corners.append((x_, y_, z_))
            signs.append(i * j * k)

        self.corners = np.array(corners, dtype=np.float64)  # (8, 3)
        self.signs = np.array(signs, dtype=np.float64)      # (8,)
        self.scale = Br / 4 / np.pi
        self.eps = 1e-16

    def Bx(self, x, y, z):
        x, y, z = np.broadcast_arrays(x, y, z)
        bx = np.zeros_like(x, dtype=float)
        for (x_, y_, z_), sign in zip(self.corners, self.signs):
            r = np.sqrt((x - x_)**2 + (y - y_)**2 + (z - z_)**2)
            bx += sign * np.log(y - y_ + r + self.eps)
        return -bx * self.scale

    def By(self, x, y, z):
        x, y, z = np.broadcast_arrays(x, y, z)
        by = np.zeros_like(x, dtype=float)
        for (x_, y_, z_), sign in zip(self.corners, self.signs):
            r = np.sqrt((x - x_)**2 + (y - y_)**2 + (z - z_)**2)
            by += sign * np.log(x - x_ + r + self.eps)
        return -by * self.scale

    def Bz(self, x, y, z):
        x, y, z = np.broadcast_arrays(x, y, z)
        bz = np.zeros_like(x, dtype=float)
        for (x_, y_, z_), sign in zip(self.corners, self.signs):
            r = np.sqrt((x - x_)**2 + (y - y_)**2 + (z - z_)**2)
            bz += sign * np.arctan2((x - x_) * (y - y_), (z - z_) * r + self.eps)
        return bz * self.scale

    def B(self, x, y, z):
        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
        return np.stack([self.Bx(x, y, z), self.By(x, y, z), self.Bz(x, y, z)], axis=-1)


class MagnetPackV1:
    def __init__(self, *magnets):
        self.magnets = magnets

    def Bx(self, x, y, z):
        return sum(magnet.Bx(x, y, z) for magnet in self.magnets)

    def By(self, x, y, z):
        return sum(magnet.By(x, y, z) for magnet in self.magnets)

    def Bz(self, x, y, z):
        return sum(magnet.Bz(x, y, z) for magnet in self.magnets)

    def B(self, x, y, z):
        return np.stack([self.Bx(x, y, z), self.By(x, y, z), self.Bz(x, y, z)], axis=-1)


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


def lorentz(t, u, Efield, Bfield):
    E_CHARGE = -1.602e-19   # [C]
    E_MASS   = 9.109e-31    # [kg]
    x, y, z = u[:3].T       # [x, y, z]
    vel = u[3:]             # [vx, vy, vz]
    E = Efield.E(x, y, z)
    B = Bfield.B(x, y, z)
    acc = (E_CHARGE / E_MASS) * (E + np.cross(vel, B))
    return np.concatenate([vel, acc])   # du/dt = [v, a] = [v, (q/m)(E + v × B)]


# @njit
def rk4(eqn, u0, t, args=()):
    n = t.size
    u0 = np.asarray(u0, dtype=np.float64)
    u = np.empty((n, *u0.shape), dtype=np.float64)
    u[0] = u0

    for i in range(n - 1):
        h = t[i+1] - t[i]
        k1 = h * eqn(t[i], u[i], *args)
        k2 = h * eqn(t[i] + h*0.5, u[i] + k1*0.5, *args)
        k3 = h * eqn(t[i] + h*0.5, u[i] + k2*0.5, *args)
        k4 = h * eqn(t[i] + h, u[i] + k3, *args)
        u[i+1] = u[i] + (k1 + 2*k2 + 2*k3 + k4) / 6.0
    return u


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from scipy.integrate import odeint

    magnet0 = Magnet((-0.005, -0.005, -0.005), (0.01, 0.01, 0.01), 1.4)

    magnet1 = Magnet((-0.175, -0.010, 0.0), (0.35, 0.02, 0.01),  1.4)    # center
    magnet2 = Magnet((-0.200, -0.025, 0.0), (0.01, 0.05, 0.01), -1.4)    # bottom
    magnet3 = Magnet((-0.200,  0.025, 0.0), (0.40, 0.01, 0.01), -1.4)    # right
    magnet4 = Magnet(( 0.190, -0.025, 0.0), (0.01, 0.05, 0.01), -1.4)    # top
    magnet5 = Magnet((-0.200, -0.035, 0.0), (0.40, 0.01, 0.01), -1.4)    # left
    magnets = MagnetPack(magnet1, magnet2, magnet3, magnet4, magnet5)

    magnets_info = [
        [(-0.175, -0.010), 0.35, 0.02],
        [(-0.200, -0.025), 0.01, 0.05],
        [(-0.200,  0.025), 0.40, 0.01],
        [( 0.190, -0.025), 0.01, 0.05],
        [(-0.200, -0.035), 0.40, 0.01],
    ]

    if 0:   # Figure 2-2. Bx along 2cm above the Magnet
        x = np.linspace(-0.05, 0.05, 101)
        Bx = magnet0.Bx(x, 0, 0.02) * 10000

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x, Bx, 'g', lw=2)
        ax.axhline(y=0, c='k', lw=1.5, ls="--")

        ax.set_title("Bx along 2cm above the Magnet", size=15)
        ax.set_xlabel("x [m]", fontsize=12)
        ax.set_ylabel("Bx [G]", fontsize=12)

        ax.grid()
        fig.tight_layout()
        plt.show()

    if 0:   # Figure 2-4. Magnetic field vectors.
        y = np.linspace(-0.05, 0.05, 101)
        z = np.linspace(-0.05, 0.05, 101)
        Y, Z = np.meshgrid(y, z)

        B = magnet0.B(0, Y, Z) * 10000
        By, Bz = B[..., 1], B[..., 2]
        normB  = np.sqrt(By**2 + Bz**2)

        fig, ax = plt.subplots(figsize=(6, 6))
        cmap = ax.streamplot(Y, Z, By, Bz, cmap = "coolwarm", color = np.clip(normB, 50, 500), linewidth=1, density=[1, 2])
        cbar = fig.colorbar(cmap.lines, ax=ax, shrink=0.6)
        cbar.ax.set_ylabel("|B| [G]", fontsize=12)
        ax.add_patch(Rectangle((-0.005, -0.005), 0.01, 0.01, facecolor='gray', alpha=0.5))

        ax.set_title("Magnetic Vector Field", fontsize=15)
        ax.set_xlabel("y [m]", fontsize=12); ax.set_xlim(y.min(), y.max())
        ax.set_ylabel("z [m]", fontsize=12); ax.set_ylim(z.min(), z.max())
        ax.set_aspect("equal")

        ax.grid()
        fig.tight_layout()
        plt.show()

    if 0:   # Figure 2-14. Magnetic field 2cm above magnet array.
        x = np.linspace(0.05, 0.21, 501)
        y = np.linspace(-0.04, 0.04, 301)
        X, Y = np.meshgrid(x, y)

        B = magnets.B(X, Y, 0.02) * 10000
        Bx, By, Bz = B[..., 0], B[..., 1], B[..., 2]
        normB  = np.sqrt(Bx**2 + By**2)

        fig, ax = plt.subplots(figsize=(8,6))
        ax.contourf(X, Y, normB, cmap='coolwarm')
        ax.contourf(X, Y, Bz, levels=np.linspace(-10, 10, 3), colors='k')
        ax.streamplot(X, Y, Bx, By, cmap='jet', color= np.clip(normB, 200, 1500), linewidth=1, density=1.5)

        for (x0, y0), dx, dy in magnets_info:
            ax.add_patch(Rectangle((x0, y0), dx, dy, facecolor='gray', alpha=0.5))

        ax.set_title("Parallel Magnetic Field @z=2cm", fontsize=15)
        ax.set_xlabel("x [m]", fontsize=11); ax.set_xlim(x.min(), x.max())
        ax.set_ylabel("y [m]", fontsize=11); ax.set_ylim(y.min(), y.max())

        ax.set_aspect("equal")
        fig.tight_layout()
        plt.show()

    if 0:   # Figure 4-3. Linear electric field in sheath
        target = Target(voltage=-300, sheath=0.001)
        z = np.linspace(0, 0.002, 101)
        Ez = target.Ez(0, 0, z)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(z * 1000, Ez, 'g', lw=2)
        ax.set_xlabel('z (mm)')
        ax.set_ylabel('$E_z$ (V/m)')
        ax.set_title('Electric Field in Sheath')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 2)

        fig.tight_layout()
        plt.show()
        
    if 1:   # Figure 4-4 / Figure 4-5
        target = Target(-300, 0.001)
        magnet1 = Magnet((-0.2, -0.02, -0.02), (0.4, 0.01, 0.01), 1.4)
        magnet2 = Magnet((-0.2,  0.01, -0.02), (0.4, 0.01, 0.01), -1.4)
        magnets = MagnetPack(magnet1, magnet2)
        
        t = np.arange(0, 100, 0.01) * 1.0E-9        # [ns]  Tracing time
        x0, y0, z0  = (0.0, -0.007, 0.0)            # [m]   Initial position
        vx0, vy0, vz0 = (0., 0., 0.)                # [m/s] Initial velocity
        u0 = np.array([x0, y0, z0, vx0, vy0, vz0])
        
        u = rk4(lorentz, u0, t, args=(target, magnets))
        # u = odeint(lorentz, u0, t, args=(target, magnets), tfirst=True)
        ex, ey, ez, evx, evy, evz = u.T
        
        xlim = [0.0, 0.1]
        ylim = [-0.01, 0.01]
        zlim = [0.0, 0.0025]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5))
        ax1.plot(ex, ey, lw=1.5)

        ax1.set_title("(Fig. 4.4) Electron Trajectory in X-Y Plane", fontsize=15)
        ax1.set_xlabel("$x$ [m]", fontsize=11); ax1.set_xlim(xlim)
        ax1.set_ylabel("$y$ [m]", fontsize=11); ax1.set_ylim(ylim)
        ax1.set_aspect("equal")
        ax1.grid()
        
        y = np.linspace(ylim[0], ylim[1], 500)
        z = np.linspace(zlim[0], zlim[1], 300)
        Y, Z = np.meshgrid(y,z)

        B = magnets.B(0, Y, Z) * 10000
        By, Bz = B[..., 1], B[..., 2]
        normB  = np.sqrt(By**2 + Bz**2)

        ax2.plot(ey, ez, lw=1.5)
        ax2.streamplot(Y, Z, By, Bz, cmap='jet', color=np.clip(normB, 200, 1500), linewidth=1.5, density=1)

        ax2.set_title("(Fig. 4.5) Side View Electron Trajectory", fontsize=15)
        ax2.set_xlabel("$y$ [m]", fontsize=11); ax2.set_xlim(ylim)
        ax2.set_ylabel("$z$ [m]", fontsize=11); ax2.set_ylim(zlim)
        ax1.set_aspect("equal")
        ax2.grid()

        fig.tight_layout()
        plt.show()

    if 1:   # Figure 4-10 / Figure 4-11
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 8))

        ax1.plot(ex, evx, lw=1.5)
        ax1.set_title("Phase Space: $v_x$ vs. $x$", fontsize=15)
        ax1.set_xlabel("$x$ [m]", fontsize=11); ax1.set_xlim([0,0.014])
        ax1.set_ylabel("$v_x$ [m/s]", fontsize=11)
        ax1.grid()

        ax2.plot(ey, evy, lw=1.5)
        ax2.set_title("Phase Space: $v_y$ vs. $y$", fontsize=15)
        ax2.set_xlabel("$y$ [m]", fontsize=11); ax2.set_xlim([-0.008,0.008])
        ax2.set_ylabel("$v_y$ [m/s]", fontsize=11)
        ax2.grid()

        ax3.plot(ez, evz, lw=1.5)
        ax3.set_title("Phase Space: $v_z$ vs. $z$", fontsize=15)
        ax3.set_xlabel("$z$ [m]", fontsize=11); ax3.set_xlim([0.0,0.0022])
        ax3.set_ylabel("$v_z$ [m/s]", fontsize=11)
        ax3.grid()

        fig.tight_layout()
        plt.show()
