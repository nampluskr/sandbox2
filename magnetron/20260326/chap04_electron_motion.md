```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from common.magnet import Magnet, MagnetPack
from common.magnet import Target, lorentz, rk4
from scipy.integrate import odeint
```

### 4.1 Motion with Two Magnets

```python
target = Target(voltage=-300, sheath=0.001)
magnets = MagnetPack(
    Magnet(corner=(-0.2, -0.02, -0.02), size=(0.4, 0.01, 0.01), Br=1.4),
    Magnet(corner=(-0.2,  0.01, -0.02), size=(0.4, 0.01, 0.01), Br=-1.4),
)

t = np.arange(0, 100, 0.01) * 1.0E-9        # [ns]  Tracing time
p0 = 0.0, -0.007, 0.0                       # [m]   Initial position [px0, py0, pz0]
v0 = 0.0, 0.0, 0.0                          # [m/s] Initial velocity [vx0, vy0, vz0]
u0 = [*p0, *v0]                             # Inital values for ODE Solver

sol = rk4(lorentz, u0, t, args=(target, magnets))
# sol = odeint(lorentz, u0, t, args=(target, magnets), tfirst=True)
px, py, pz, vx, vy, vz = sol.T
```

```python
# Figure 4-4. Electron Trajectory in X-Y Plane.
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(px, py)
ax.set_xlim(0, 0.1)
ax.set_ylim(-0.01, 0.01)

ax.set_title("(Fig. 4.4) Electron Trajectory in $x-y$ Plane")
ax.set_xlabel("$x$ [m]")
ax.set_ylabel("$y$ [m]")
ax.set_aspect("equal") 

ax.grid()
fig.tight_layout()
plt.show()
```

```python
# Figure 4-5. Side View Electron Trajectory.
y = np.linspace(-0.01, 0.01, 501)
z = np.linspace(0, 0.0025, 301)
Y, Z = np.meshgrid(y, z)

B = magnets.B(0, Y, Z) * 10000
By, Bz = B[..., 1], B[..., 2]
normB  = np.sqrt(By**2 + Bz**2)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(py, pz, lw=1.5)
ax.streamplot(Y, Z, By, Bz, cmap='jet', color=np.clip(normB, 200, 1500), linewidth=1.5, density=1)
ax.set_xlim(-0.01, 0.01)
ax.set_ylim(0, 0.0025)

ax.set_title("(Fig. 4.5) Side View Electron Trajectory")
ax.set_xlabel("$y$ [m]")
ax.set_ylabel("$z$ [m]")
ax.set_aspect("equal")
ax.grid()

fig.tight_layout()
plt.show()
```

```python
# Figure 4-10 y position vs y velocity
# Figure 4-11 z and x positions vs z and x components of velocity
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 8))

ax1.plot(px, vx)
ax1.set_xlim([0, 0.014])
ax1.set_title("Phase Space: $v_x$ vs. $x$")
ax1.set_xlabel("$x$ [m]")
ax1.set_ylabel("$v_x$ [m/s]")

ax2.plot(py, vy)
ax2.set_xlim([-0.008, 0.008])
ax2.set_title("Phase Space: $v_y$ vs. $y$")
ax2.set_xlabel("$y$ [m]")
ax2.set_ylabel("$v_y$ [m/s]")

ax3.plot(pz, vz)
ax3.set_xlim([0.0, 0.0022])
ax3.set_title("Phase Space: $v_z$ vs. $z$")
ax3.set_xlabel("$z$ [m]")
ax3.set_ylabel("$v_z$ [m/s]")

for ax in (ax1, ax2, ax3):
    ax.grid()

fig.tight_layout()
plt.show()
```

### 4.3 Varying Magnet Strength

```python
target = Target(voltage=-300, sheath=0.001)
magnets = MagnetPack(
    Magnet(corner=(-0.2, -0.02, -0.02), size=(0.2, 0.01, 0.01), Br= 1.4/2.0),   # Left
    Magnet(corner=( 0.0, -0.02, -0.02), size=(0.2, 0.01, 0.01), Br= 1.4),       # Left
    Magnet(corner=(-0.2,  0.01, -0.02), size=(0.2, 0.01, 0.01), Br=-1.4/2.0),   # Right
    Magnet(corner=( 0.0,  0.01, -0.02), size=(0.2, 0.01, 0.01), Br=-1.4),       # Right
)
t = np.arange(0, 150, 0.01) * 1.0E-9    # [ns]  Tracing time
p0 = -0.1, -0.007, 0.0                  # [m]   Initial position [px0, py0, pz0]
v0 = 0.0, 0.0, 0.0                      # [m/s] Initial velocity [vx0, vy0, vz0]
u0 = [*p0, *v0]                         # Inital values for ODE Solver

sol = rk4(lorentz, u0, t, args=(target, magnets))
px, py, pz, vx, vy, vz = sol.T
```

```python
# (Fig. 4.19) Electron Trajectory in $x-y$ Plane
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(px, py)
ax.set_xlim(-0.1, 0.05)
ax.set_ylim(-0.01, 0.01)

ax.set_title("(Fig. 4.19) Electron Trajectory in $x-y$ Plane")
ax.set_xlabel("$x$ [m]")
ax.set_ylabel("$y$ [m]")
ax.set_aspect("equal")
ax.grid(True)
fig.tight_layout()
plt.show()
```

```python
# (Fig. 4.20) Side View Electron Trajectory
y = np.linspace(-0.1, 0.05, 501)
z = np.linspace(0.0, 0.0035, 300)
Y, Z = np.meshgrid(y, z)

magnets = MagnetPack(
    Magnet(corner=(-0.2, -0.02, -0.02), size=(0.4, 0.01, 0.01), Br= 1.4),   # Left
    Magnet(corner=(-0.2,  0.01, -0.02), size=(0.4, 0.01, 0.01), Br=-1.4),   # Right    
)
B = magnets.B(0, Y, Z) * 10000
By, Bz = B[..., 1], B[..., 2]
normB  = np.sqrt(By**2 + Bz**2)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(py, pz)
# ax.streamplot(Y, Z, By, Bz, cmap='jet', color=np.clip(normB, 200, 1500), linewidth=1.5, density=1)
ax.set_xlim(-0.01, 0.01)
ax.set_ylim(0.0, 0.0035)

ax.set_title("(Fig. 4.20) Side View Electron Trajectory")
ax.set_xlabel("$y$ [m]")
ax.set_ylabel("$z$ [m]")

ax.set_aspect("equal")
ax.grid(True)
fig.tight_layout()
plt.show()
```

### 4.4 Full Magnetron

```python
target = Target(voltage=-300, sheath=0.001)
magnets = MagnetPack(
    Magnet(corner=(-0.175, -0.010, -0.02), size=(0.35, 0.02, 0.01), Br= 1.4),   # Center
    Magnet(corner=(-0.200, -0.025, -0.02), size=(0.01, 0.05, 0.01), Br=-1.4),   # Bottom
    Magnet(corner=(-0.200,  0.025, -0.02), size=(0.40, 0.01, 0.01), Br=-1.4),   # Right
    Magnet(corner=( 0.190, -0.025, -0.02), size=(0.01, 0.05, 0.01), Br=-1.4),   # Top
    Magnet(corner=(-0.200, -0.035, -0.02), size=(0.40, 0.01, 0.01), Br=-1.4),   # Left
)

y = np.linspace(-0.04, 0.04, 501)
z = np.linspace(-0.02, 0.03, 301)
Y, Z = np.meshgrid(y,z)
B = magnets.B(0, Y, Z) * 10000
Bx, By, Bz = B[..., 0], B[..., 1], B[..., 2]
normB  = np.sqrt(By**2 + Bz**2)
```

```python
# Figure 4-24. Magnetic field vercotrs at x = 0. The horizontal line represents the target surface.
fig, ax = plt.subplots(figsize=(8,6))
ax.add_patch(Rectangle((-0.035, -0.02), 0.01, 0.01, facecolor="gray", alpha=0.3 ))
ax.add_patch(Rectangle((-0.010, -0.02), 0.02, 0.01, facecolor="gray", alpha=0.3 ))
ax.add_patch(Rectangle(( 0.025, -0.02), 0.01, 0.01, facecolor="gray", alpha=0.3 ))

# ax.contourf(Y, Z, normB, cmap='coolwarm', levels=np.linspace(50, 8000, 21))
ax.contourf(Y, Z, Bz, levels=np.linspace(-1,1,3), colors='k')
ax.streamplot(Y, Z, By, Bz, cmap='jet', color=np.clip(normB,200,1500), linewidth=1.5, density=1 )
ax.axhline(y=0, c='k', lw=1.5, ls="--")

ax.set_xlim(-0.04, 0.04)
ax.set_ylim(-0.02, 0.03)

ax.set_title("(Fig. 4.24) Magnetic Field at $x$=0cm")
ax.set_xlabel("$y$ [m]")
ax.set_ylabel("$z$ [m]")

ax.set_aspect("equal")
ax.grid()
fig.tight_layout()
plt.show()
```

```python
# Figure 4-25. Trajectory of an electron starting at (-0.10, -0.007)
t = np.arange(0.0, 260.0, 0.01) * 1.0E-9    # [ns]  Tracing time
p0  = -0.1, -0.007, 0.0                     # [m]   Initial position [px0, py0, pz0]
v0 = 0.0, 0.0, 0.0                          # [m/s] Initial velocity [vx0, vy0, vz0]
u0 = [*p0, *v0]

sol = rk4(lorentz, u0, t, args=(target, magnets))
px, py, pz, vx, vy, vz = sol.T

fig, ax = plt.subplots(figsize=(8, 6))
ax.add_patch(Rectangle((-0.175, -0.010), 0.35, 0.02, facecolor="gray", alpha=0.3 ))
ax.add_patch(Rectangle((-0.200, -0.035), 0.40, 0.01, facecolor="gray", alpha=0.3 ))
ax.add_patch(Rectangle((-0.200,  0.025), 0.40, 0.01, facecolor="gray", alpha=0.3 ))
ax.add_patch(Rectangle(( 0.190, -0.025), 0.01, 0.05, facecolor="gray", alpha=0.3 ))
ax.add_patch(Rectangle((-0.200, -0.025), 0.01, 0.05, facecolor="gray", alpha=0.3 ))
ax.plot(px, py)

ax.set_xlim(-0.205, -0.1)
ax.set_ylim(-0.04, 0.04)

ax.set_title("(Fig. 4.25) Electron Trajectory Starting at (-0.1,-0.007)")
ax.set_xlabel("$x$ [m]")
ax.set_ylabel("$y$ [m]")

ax.set_aspect("equal")
ax.grid(True)
fig.tight_layout()
plt.show()
```

```python
x = np.linspace(-0.205, -0.1, 201)
y = np.linspace(-0.04, 0.04, 101)
X, Y = np.meshgrid(x, y)
B = magnets.B(X, Y, 0) * 10000
Bx, By, Bz = B[..., 0], B[..., 1], B[..., 2]
normB  = np.sqrt(Bx**2 + By**2)
```

```python
# Figure. 4-26. Electron Trajectory Starting at (-0.1, -0.007).
fig, ax = plt.subplots(figsize=(8, 6))
ax.contourf(X, Y, normB, cmap='coolwarm')
ax.contourf(X, Y, Bz, levels=np.linspace(-10, 10, 3), colors='k')
# ax.streamplot(X, Y, Bx, By, cmap='jet', color=np.clip(normB, 200, 1500), linewidth=1, density=2)

ax.add_patch(Rectangle((-0.175, -0.010), 0.35, 0.02, facecolor="gray", alpha=0.3 ))
ax.add_patch(Rectangle((-0.200, -0.035), 0.40, 0.01, facecolor="gray", alpha=0.3 ))
ax.add_patch(Rectangle((-0.200,  0.025), 0.40, 0.01, facecolor="gray", alpha=0.3 ))
ax.add_patch(Rectangle(( 0.190, -0.025), 0.01, 0.05, facecolor="gray", alpha=0.3 ))
ax.add_patch(Rectangle((-0.200, -0.025), 0.01, 0.05, facecolor="gray", alpha=0.3 ))

ax.plot(px, py)
ax.set_xlim(-0.205, -0.1)
ax.set_ylim(-0.04, 0.04)

ax.set_title("(Fig. 4.26) Electron Trajectory Starting at (-0.1, -0.007)")
ax.set_xlabel("$x$ [m]")
ax.set_ylabel("$y$ [m]")
ax.set_aspect("equal")
# ax.grid()
fig.tight_layout()
plt.show()
```
