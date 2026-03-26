```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from common.magnet import Magnet, MagnetPack
```

```python
magnet = Magnet(corner=(-0.005, -0.005, -0.005), size=(0.01, 0.01, 0.01), Br=1.4)
```

```python
# Figure 2-2. Bx along 2cm above the Magnet
x = np.linspace(-0.05, 0.05, 101)
Bx = magnet.Bx(x, 0, 0.02) * 10000

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, Bx)
ax.axhline(y=0, ls="--", c='k')

ax.set_title("$B_x$ along 2cm above the Magnet")
ax.set_xlabel("$x$ [m]")
ax.set_ylabel("$B_x$ [G]")

ax.grid()
fig.tight_layout()
plt.show()
```

```python
# Figure 2-4. Magnetic field vectors.
y = np.linspace(-0.05, 0.05, 101)
z = np.linspace(-0.05, 0.05, 101)
Y, Z = np.meshgrid(y, z)

B = magnet.B(0, Y, Z) * 10000
By, Bz = B[..., 1], B[..., 2]
normB  = np.sqrt(By**2 + Bz**2)

fig, ax = plt.subplots(figsize=(6, 6))
splot = ax.streamplot(Y, Z, By, Bz, cmap="coolwarm", 
    color=np.clip(normB, 50, 500), linewidth=1, density=[1, 2])
cbar = fig.colorbar(splot.lines, ax=ax, shrink=0.7)
cbar.ax.set_ylabel("$|B|$ [G]", fontsize=12)
ax.add_patch(Rectangle((-0.005, -0.005), 0.01, 0.01, facecolor='gray', alpha=0.5))

ax.set_xlim(y.min(), y.max())
ax.set_ylim(z.min(), z.max())

ax.set_title("Magnetic Vector Field")
ax.set_xlabel("$y$ [m]") 
ax.set_ylabel("$z$ [m]") 
ax.set_aspect("equal")

ax.grid()
fig.tight_layout()
plt.show()
```

```python
magnets = MagnetPack(
    Magnet(corner=(-0.175, -0.010, 0.0), size=(0.35, 0.02, 0.01), Br= 1.4),     # center
    Magnet(corner=(-0.200, -0.025, 0.0), size=(0.01, 0.05, 0.01), Br=-1.4),     # bottom
    Magnet(corner=(-0.200,  0.025, 0.0), size=(0.40, 0.01, 0.01), Br=-1.4),     # right
    Magnet(corner=( 0.190, -0.025, 0.0), size=(0.01, 0.05, 0.01), Br=-1.4),     # top
    Magnet(corner=(-0.200, -0.035, 0.0), size=(0.40, 0.01, 0.01), Br=-1.4),     # left
)

magnets_info = [
    [(-0.175, -0.010), 0.35, 0.02],
    [(-0.200, -0.025), 0.01, 0.05],
    [(-0.200,  0.025), 0.40, 0.01],
    [( 0.190, -0.025), 0.01, 0.05],
    [(-0.200, -0.035), 0.40, 0.01],
]
```

```python
# Figure 2-14. Magnetic field 2cm above magnet array.
x = np.linspace(0.05, 0.21, 501)
y = np.linspace(-0.04, 0.04, 301)
X, Y = np.meshgrid(x, y)

B = magnets.B(X, Y, 0.02) * 10000
Bx, By, Bz = B[..., 0], B[..., 1], B[..., 2]
normB  = np.sqrt(Bx**2 + By**2)

fig, ax = plt.subplots(figsize=(8,6))
ax.contourf(X, Y, normB, cmap='coolwarm')
ax.contourf(X, Y, Bz, levels=np.linspace(-10, 10, 3), colors='k')
ax.streamplot(X, Y, Bx, By, cmap='jet', color=np.clip(normB, 200, 1500), linewidth=1, density=1.5)

for (x0, y0), dx, dy in magnets_info:
    ax.add_patch(Rectangle((x0, y0), dx, dy, facecolor='gray', alpha=0.5))

ax.set_title("Parallel Magnetic Field at $z$=2cm")
ax.set_xlabel("$x$ [m]"); ax.set_xlim(x.min(), x.max())
ax.set_ylabel("$y$ [m]"); ax.set_ylim(y.min(), y.max())

ax.set_aspect("equal")
fig.tight_layout()
plt.show()
```
