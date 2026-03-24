from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import interp1d
from time import clock
import cypct as cy
import pypct as py

if __name__ == "__main__":

##############################################################################
### Simulation Input Parameters:
##############################################################################
    py.TIME_STEP    = 0.01E-9
    py.TIME_MAX_NUM = 100000

    voltage  = -300
    sheath   = 0.001
    scale    = 1.0
    filename = "ArCrossSections.csv"
    torr     = 0.005
    kelvin   = 293
    num_electron = 1
##############################################################################

    E = cy.ElectricField(voltage, sheath)
    mag1 = cy.Magnet([-0.08,-0.01,-0.02],[0.16,0.02,0.01], scale) # Center
    mag2 = cy.Magnet([-0.10,-0.02,-0.02],[0.01,0.04,0.01],-scale) # Bottom
    mag3 = cy.Magnet([-0.10, 0.02,-0.02],[0.20,0.01,0.01],-scale) # Right
    mag4 = cy.Magnet([ 0.09,-0.02,-0.02],[0.01,0.04,0.01],-scale) # Top
    mag5 = cy.Magnet([-0.10,-0.03,-0.02],[0.20,0.01,0.01],-scale) # Left
    B = cy.MagneticField([mag1, mag2, mag3, mag4, mag5])
    ar_plasma = py.Plasma(filename, torr, kelvin)
    
    p0_list, v0_list = [], []
    for i in xrange(num_electron):
        p0 = v0 = np.zeros(3)
        p0[0] = np.random.uniform(-0.095,0.095)
        p0[1] = np.random.uniform(-0.025,0.025)
        p0_list.append(p0)
        v0_list.append(v0)

    if 1:
        tstart = clock()
        uu, cc = py.trace_many_electrons(p0_list,v0_list,E,B,ar_plasma)
        uu_el  = uu[np.where(cc==0)]
        uu_ex  = uu[np.where(cc==1)]
        uu_iz  = uu[np.where(cc==2)]
        print "Time >> %.4e [sec]" % (clock()-tstart)

        fig = plt.figure(figsize=(10,5))
        ax1 = fig.add_subplot(111)
        ax1.plot(uu_el[:,0], uu_el[:,1], 'go', lw=1.5)
        ax1.plot(uu_ex[:,0], uu_ex[:,1], 'bo', lw=1.5)
        ax1.plot(uu_iz[:,0], uu_iz[:,1], 'ro', lw=1.5)
        py.draw_magnets(ax1,xlim=[-0.105,0.105],ylim=[-0.035,0.035])
        
        fig.tight_layout()
        plt.show()
