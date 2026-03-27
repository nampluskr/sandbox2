```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from common.sputter import Target, Magnet, MagnetPack
from common.plasma import CrossSection, new_velocity
from common.solver import trace_next, trace, trace_all
from common.solver import lorentz, rk4
```

```python
target = Target(voltage=-300, sheath=0.001)
magnet = MagnetPack(
    Magnet(corner=(-0.2, -0.02, -0.02), size=(0.4, 0.01, 0.01), Br=1.4),
    Magnet(corner=(-0.2,  0.01, -0.02), size=(0.4, 0.01, 0.01), Br=-1.4),
)
time, timestep = np.linspace(0, 100e-9, 10001, retstep=True)
pos0 = 0.0, -0.007, 0.0     # [m]   Initial position [px0, py0, pz0]
vel0 = 0.0, 0.0, 0.0        # [m/s] Initial velocity [vx0, vy0, vz0]
```

```python
pos, vel = trace_all(pos0, vel0, time, args=(target, magnet))
px, py, pz = pos.T

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
pos, vel = trace(pos0, vel0, time, args=(target, magnet))
px, py, pz = pos.T

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
E_CHARGE = 1.602e-19   # [C]
E_MASS   = 9.109e-31    # [kg]

gas = CrossSection("common\\ArCrossSections.csv", torr=0.005, kelvin=293)

def kinetic_energy(vel):
    vx, vy, vz = vel
    return 0.5 * (vx**2 + vy**2 + vz**2) * (E_MASS /E_CHARGE)

def trace_single(pos0, vel0, t, target, magnet, gas):
    pos_list = []
    event_list = []

    for i in range(t.size - 1):
        t0, dt = t[i], t[i+1] - t[i]
        pos, vel = trace_next(pos0, vel0, t0, dt, args=(target, magnet))

        energy = kinetic_energy(vel)
        dist = np.linalg.norm(pos - pos0)
        r = np.random.rand()

        if r < gas.collision_probability(energy, dist):
            event = gas.decide_event(energy)
            vel = new_velocity(vel, event)
            pos_list.append(pos)
            event_list.append(event)
            
        potential = target.potential(*pos)
        if energy + potential < 26:
            break

        pos0, vel0 = pos, vel

    return pos_list, event_list

np.random.seed(1234)
pos_list, event_list = trace_single(pos0, vel0, time, target, magnet, gas)

for event, pos in zip(event_list, pos_list):
    px, py, pz = pos
    print(f"{event}: {px:.4f}, {py:4f}, {pz:4f}")
```

```python
from time import clock

def trace_many(pos0_list, vel0_list, time, target, magnet, gas):
    pos_list, event_list = [], []
    
    for i, pos0, vel0 in enumerate(zip(pos0_list, vel0_list)):
        tstart = clock()
        pos_list, event_list = trace_single(pos0, vel0, time, target, magnet, gas)
        tend = clock()
        print(f"[%4d/%4d] collisions: %3d / %.2f [s]" % (i+1, max_num, len(c_list), tend-tstart))
        pos_list  += p_list
        coll_list += c_list
            
    return np.concatenate(pos_list).reshape(-1,3), np.array(coll_list)
```
