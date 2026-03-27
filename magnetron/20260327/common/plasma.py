import numpy as np
from numpy import sin, cos
from scipy.interpolate import interp1d


E_CHARGE = 1.602E-19    # [C]
E_MASS   = 9.109E-31    # [Kg]

class CrossSection:
    def __init__(self, filename, torr, kelvin):
        db = np.genfromtxt(filename, delimiter=',', skip_header=3)
        self.cross_sections = {
            'el': interp1d(db[:, 0], db[:, 1]),
            'ex': interp1d(db[:, 0], db[:, 2]),
            'iz': interp1d(db[:, 0], db[:, 3]),
            'tt': interp1d(db[:, 0], db[:, 4]),
        }
        self.ng = torr * 133.32 * 6.022E23 / 8.314 / kelvin
        self.event_probability = {
            'el': self.elastic_probability,
            'ex': self.excitation_probability,
            'iz': self.ionization_probability,
            'tt': self.collision_probability,
        }

    def elastic_probability(self, energy, distance):
        sigma = self.cross_sections['el'](energy) * 1.0E-20
        return 1 - np.exp(-self.ng * sigma  * distance)

    def excitation_probability(self, energy, distance):
        sigma = self.cross_sections['ex'](energy) * 1.0E-20
        return 1 - np.exp(-self.ng * sigma * distance)
    
    def ionization_probability(self, energy, distance):
        sigma = self.cross_sections['iz'](energy) * 1.0E-20
        return 1 - np.exp(-self.ng * sigma * distance)

    def collision_probability(self, energy, distance):
        sigma = self.cross_sections['tt'](energy) * 1.0E-20
        return 1 - np.exp(-self.ng * sigma * distance)
    
    def decide_event(self, energy):
        sig_el = self.cross_sections['el'](energy)
        sig_ex = self.cross_sections['ex'](energy)
        sig_tt = self.cross_sections['tt'](energy)

        r = np.random.rand()
        if r < sig_el / sig_tt and energy > 1:
            return "el"    # elastic
        if r < (sig_el + sig_ex) / sig_tt and energy > 12:
            return "ex"    # excitation
        if energy > 16:
            return "iz"    # ionization


def new_velocity(vel, event):
    energy_loss = {'el': 1, 'ex': 12, 'iz': 16}
    vx, vy, vz = vel
    ke  = 0.5 * (vx**2 + vy**2 + vz**2) * (E_MASS / E_CHARGE)
    new_ke = ke - energy_loss[event]
    vmag  = np.linalg.norm(vel)
    new_vmag = vmag * np.sqrt(new_ke / ke)
    
    theta = np.arccos(vx / vmag)
    phi   = np.arctan2(vy, vx)
    r     = np.random.rand()
    chi   = np.arccos(1 - 2 * r / (1 + 8 * ke * (1 - r) / 27.21))
    psi   = 2* np.pi * np.random.rand()

    new_vx = cos(theta) * cos(phi) * sin(chi) * cos(psi) \
           - sin(phi) * sin(chi) * sin(psi) + sin(theta) * cos(phi) * cos(chi)
    new_vy = cos(theta) * sin(phi) * sin(chi) * cos(psi) \
           + cos(phi) * sin(chi) * sin(psi) + sin(theta) * sin(phi) * cos(chi)
    new_vz = -sin(theta) * sin(chi) * cos(psi) + cos(theta) * cos(chi)

    return np.array([new_vx, new_vy, new_vz]) * new_vmag
