```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.interpolate import interp1d

from common.magnet import Magnet, MagnetPack
```

```python
from scipy.interpolate import interp1d

class CrossSection:
    def __init__(self, filename, torr, kelvin):
        data = np.genfromtxt(filename, delimiter=',', skip_header=3)
        self.sig_el = interp1d(data[:, 0], data[:, 1])
        self.sig_ex = interp1d(data[:, 0], data[:, 2])
        self.sig_iz = interp1d(data[:, 0], data[:, 3])
        self.sig_tt = interp1d(data[:, 0], data[:, 4])
        
        self.ng = torr * 133.32 * 6.022E23 / 8.314 / kelvin
        
    def prob_el(self, energy, distance):
        return 1 - np.exp(-self.ng * self.sig_el(energy) * 1.0E-20 * distance)

    def prob_ex(self, energy, distance):
        return 1 - np.exp(-self.ng * self.sig_ex(energy) * 1.0E-20 * distance)

    def prob_iz(self, energy, distance):
        return 1 - np.exp(-self.ng * self.sig_iz(energy) * 1.0E-20 * distance)

    def prob_tt(self, energy, distance):
        return 1 - np.exp(-self.ng * self.sig_tt(energy) * 1.0E-20 * distance)

    def collision_probability(self, energy, dx):
        sig_tt = self.func_tt(energy)
        return 1.0 - np.exp(-self.ng * sig_tt * 1.0e-20 * dx)

    def collision_type(self, energy):
        if energy <= 1.0:
            return -1

        sig_el = self.func_el(energy)
        sig_ex = self.func_ex(energy)
        sig_iz = self.func_iz(energy)
        sig_tt = self.func_tt(energy)

        if sig_tt <= 0:
            return -1

        r = np.random.rand()
        if energy > 16.0 and r > (sig_el + sig_ex) / sig_tt:
            return 2  # izonization
        elif energy > 12.0 and r > sig_el / sig_tt:
            return 1  # excitation
        elif energy > 1.0:
            return 0  # elastic
        else:
            return -1
```

```python
voltage = -300          # voltage [V]
sheath = 0.001          # sheath thickness [m]
torr = 0.05             # pressure [torr]
kelvin = 300            # temperature [K]
tstep = 2e-11           # time step [s]
max_tsteps = 100000     # max. time steps
num_electrons = 1       # number of electrons

ar = ArCrossSection("common\\ArCrossSections.csv", torr, kelvin)
```

```python
E_MASS = 9.10938356e-31     # electron mass [Kg]
E_CHARGE = 1.60217662e-19   # electron charge [C]

def kinetic_energy(vx, vy, vz):
    return 0.5 * E_MASS * (vx**2 + vy**2 + vz**2) / E_CHARGE

def potential_energy(x, y, z, E):
    if z < E.sheath:
        return -0.5 * E.voltage * (z - E.sheath)**2 / E.sheath**2
    else:
        return 0.0

def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2)**2))
```

```python
class CrossSection:
    def __init__(self, filename, torr, kelvin):
        data = np.genfromtxt(filename, delimiter=',', skip_header=3)
        self.sig_el = interp1d(data[:, 0], data[:, 1])
        self.sig_ex = interp1d(data[:, 0], data[:, 2])
        self.sig_iz = interp1d(data[:, 0], data[:, 3])
        self.sig_tt = interp1d(data[:, 0], data[:, 4])
        
        self.ng = torr * 133.32 * 6.022E23 / 8.314 / kelvin
        
    def prob_el(self, energy, distance):
        return 1 - np.exp(-self.ng * self.sig_el(energy) * 1.0E-20 * distance)

    def prob_ex(self, energy, distance):
        return 1 - np.exp(-self.ng * self.sig_ex(energy) * 1.0E-20 * distance)

    def prob_iz(self, energy, distance):
        return 1 - np.exp(-self.ng * self.sig_iz(energy) * 1.0E-20 * distance)

    def prob_tt(self, energy, distance):
        return 1 - np.exp(-self.ng * self.sig_tt(energy) * 1.0E-20 * distance)
    
    def collision(self, energy):
        sig_el = self.sig_el(energy)
        sig_ex = self.sig_ex(energy)
        sig_tt = self.sig_tt(energy)
        r = np.random.rand()
        
        if r < sig_el/sig_tt and energy > 1:
            return 0 # Elastic
        if r < (sig_el + sig_ex)/sig_tt and energy > 12:
            return 1 # Excitation
        if energy > 16:
            return 2 # Ionization 
```

```python
# (Fig. 5.2) Electron-Argon Cross Sections

ar = CrossSection("common\\ArCrossSections.csv", torr=0.005, kelvin=293)
# data = np.genfromtxt("common\\ArCrossSections.csv", delimiter=',', skip_header=3)
# sig_el = interp1d(data[:, 0], data[:, 1])
# sig_ex = interp1d(data[:, 0], data[:, 2])
# sig_iz = interp1d(data[:, 0], data[:, 3])
# sig_tt = interp1d(data[:, 0], data[:, 4])

energy = np.linspace(0, 1000, 201)

fig, ax = plt.subplots()
ax.plot(energy, ar.sig_el(energy) * 1.0E-20, label="Elastic")
ax.plot(energy, ar.sig_ex(energy) * 1.0E-20, label="Excitation")
ax.plot(energy, ar.sig_iz(energy) * 1.0E-20, label="Ionization")

ax.set_title("Cross Sections")
ax.set_xlabel("Energy [eV]")
ax.set_ylabel("Cross Section [$m^2$]")
ax.grid()
ax.legend()
fig.tight_layout()
plt.show()
```

```python
# (Fig. 5.3) Electron-Argon Collision Probability
x = np.linspace(0, 1, 201)  # distance [m]
        
fig, ax = plt.subplots()
ax.plot(x, ar.prob_el(100, x), label="Elastic")
ax.plot(x, ar.prob_ex(100, x), label="Excitation")
ax.plot(x, ar.prob_iz(100, x), label="Ionization")

ax.set_title("(Fig. 5.3) Electron-Argon Collision Probability")
ax.set_xlabel("Energy [eV]")
ax.set_ylabel("Probability")
ax.grid(True)
ax.legend(loc="best")

fig.tight_layout()
plt.show()
```

```python
# (Fig. 5.4) Electron-Argon Collision Probability
x = np.linspace(0,1,200)

fig, ax = plt.subplots()
ax.plot(x, ar.prob_iz(100, x), label="100eV")
ax.plot(x, ar.prob_iz(200, x), label="200eV")
ax.plot(x, ar.prob_iz(400, x), label="400eV")

ax.set_title("(Fig. 5.4) Electron-Argon Collision Probability")
ax.set_xlabel("Energy [eV]")
ax.set_ylabel("Probability")
ax.grid(True)
ax.legend(loc="best")

fig.tight_layout()
plt.show()
```

```python
# (Fig. 5.5) Electron-Argon Collision Probability
ar1 = CrossSection("common\\ArCrossSections.csv", torr=0.002, kelvin=293)
ar2 = CrossSection("common\\ArCrossSections.csv", torr=0.005, kelvin=293)
ar3 = CrossSection("common\\ArCrossSections.csv", torr=0.010, kelvin=293)
x = np.linspace(0,1,200)

fig, ax = plt.subplots()
ax.plot(x, ar1.prob_iz(100, x), label=" 2 mtorr")
ax.plot(x, ar2.prob_iz(100, x), label=" 5 mtorr")
ax.plot(x, ar3.prob_iz(100, x), label="10 mtorr")

ax.set_title("(Fig. 5.5) Electron-Argon Collision Probability")
ax.set_xlabel("Energy [eV]")
ax.set_ylabel("Probability")
ax.grid(True)
ax.legend(loc="best")

fig.tight_layout()
plt.show()
```

```python
# (Fig. 5.7) Distribution of Axial Deflection
r = np.random.rand(100000)

fig, ax = plt.subplots()

ke = 500
angle = np.arccos(1.0 - 2*r/(1 + 8 * ke * (1-r) / 27.21)) * 180 / np.pi
hist, bins = np.histogram(angle, bins=200)
width = 1.0 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2.0
ax.bar(center, hist, align="center", width=width, color='b', label="500eV")

ke = 50
angle = np.arccos(1.0 - 2*r/(1 + 8 * ke * (1-r) / 27.21)) * 180 / np.pi
hist, bins = np.histogram(angle, bins=200)
width = 1.0 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2.0
ax.bar(center, hist, align="center", width=width, color='r', label="50eV")

ax.set_title("(Fig. 5.7) Distribution of Axial Deflection")
ax.set_xlabel("Angle [degrees]")
ax.set_ylabel("Number")
ax.grid(True)
ax.legend(loc="best")

fig.tight_layout()
plt.show()
```

```python
ar = CrossSection("common\\ArCrossSections.csv", torr=0.1, kelvin=293)
num = 1000
ke = np.linspace(0, 200, num)
coll_type = np.empty(num)
for i in range(num):
    coll_type[i] = ar.collision(ke[i])

fig, ax = plt.subplots()
ax.plot(ke, coll_type, 'o', lw=2, label="Elastic")
ax.set_ylim(-1, 3)

fig.tight_layout()
plt.show()
```
