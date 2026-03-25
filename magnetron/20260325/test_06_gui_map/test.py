import numpy as np
import matplotlib.pyplot as plt
from time import clock
import os
from pypct import *
import cypct as cy

if __name__ == "__main__":

##############################################################################
### Simulation Parameters:
##############################################################################
    TIME_STEP = 0.01E-9 # [s]
    TIME_MAX_NUM = 100000

    VOLTAGE = -300
    SHEATH  = 0.001
    SCALE   = 1.0
    TORR    = 0.005
    KELVIN  = 300
    DBFILE  = "ArCrossSections.csv"
    OUTFILE = "pct_result_01.txt"
    NUM_ELECTRON = 1

##############################################################################
    E = cy.ElectricField(VOLTAGE, SHEATH)
    mag1 = cy.Magnet(-0.08,-0.01,-0.02, 0.16,0.02,0.01, SCALE) # Center
    mag2 = cy.Magnet(-0.10,-0.02,-0.02, 0.01,0.04,0.01,-SCALE) # Bottom
    mag3 = cy.Magnet(-0.10, 0.02,-0.02, 0.20,0.01,0.01,-SCALE) # Right
    mag4 = cy.Magnet( 0.09,-0.02,-0.02, 0.01,0.04,0.01,-SCALE) # Top
    mag5 = cy.Magnet(-0.10,-0.03,-0.02, 0.20,0.01,0.01,-SCALE) # Left
    B = cy.MagneticField([mag1, mag2, mag3, mag4, mag5])

    db = np.genfromtxt(DBFILE,delimiter=',',skip_header=3)
    ng = TORR*133.32*6.022E23/8.314/KELVIN
    p0_list, v0_list = position_velocity(NUM_ELECTRON)

##############################################################################

    if 1:
        is_trajectory = True
        p0, v0 = p0_list[0], v0_list[0]
        print ">>> Simulation of particle Tracing has been started. Wait..."        
        
        tstart = clock()
        result = cy.trace_single(p0,v0,E,B,db,ng, \
            TIME_STEP,TIME_MAX_NUM, is_trajectory)
        tend = clock()
        print "Time: %.2f [s] %3d [collisions]" % (tend-tstart, len(result[3]))

        tt = np.array(result[0])
        pp = np.concatenate(result[1]).reshape(-1,3)
        vv = np.concatenate(result[2]).reshape(-1,3)
        cc = np.array(result[3])
        save_result(OUTFILE,tt,pp,vv,cc)

    if 0:
        is_trajectory = False
        max_num = num_electron = len(p0_list)
        time_list, pos_list, vel_list, coll_list = [], [], [], []

        print ">>> Simulation of particle Tracing has been started. Wait..."

        for i in xrange(max_num):
            p0, v0 = p0_list.pop(), v0_list.pop()
            num_electron -= 1
            tstart = clock()
            result = cy.trace_single(p0,v0,E,B,db,ng, \
                TIME_STEP,TIME_MAX_NUM, is_trajectory)
            tend = clock()
            print ">>> [%4d/%4d] Time: %.2f [s] %3d [collisions]" \
                % (i+1,max_num,(tend-tstart), len(result[3]))
            time_list += result[0]
            pos_list  += result[1]
            vel_list  += result[2]
            coll_list += result[3]

        tt = np.array(time_list)
        pp = np.concatenate(pos_list).reshape(-1,3)
        vv = np.concatenate(vel_list).reshape(-1,3)
        cc = np.array(coll_list)
        save_result(OUTFILE,tt,pp,vv,cc)

    if 1: # Draw Collision Positions and Electron Trajectories
        #is_trajectory = True
        result = np.genfromtxt(os.path.join("result",OUTFILE),skip_header=1)

        fig = plt.figure(figsize=(16,8))
        ax1 = plt.subplot2grid((5,6),(0,0),colspan=5,rowspan=3)
        ax2 = plt.subplot2grid((5,6),(0,5),rowspan=3)
        ax3 = plt.subplot2grid((5,6),(3,0),colspan=5,rowspan=2)

        ax1, ax2, ax3 = draw_target(ax1,ax2,ax3)
        ax1, ax2, ax3 = draw_collision(ax1,ax2,ax3,result,("el","ex","iz"),is_trajectory)

        fig.tight_layout()
        plt.show()
        
    if 1: # Electron Energy Loss due to Collision
        result = np.genfromtxt(os.path.join("result",OUTFILE),skip_header=1)
        tt, pp, vv = result[:,0], result[:,1:4], result[:,4:7]
        pe = pe_vec(pp[:,0],pp[:,1],pp[:,2],VOLTAGE,SHEATH)
        ke = ke_vec(vv[:,0],vv[:,1],vv[:,2])

        fig = plt.figure(figsize=(8,4))
        ax = fig.add_subplot(111)

        ax.plot(tt, pe+ke,'kx')
        ax.set_title("Energy Loss due to Collisions", weight="bold", size=15)
        ax.set_xlabel("Time [s]", weight="bold", size=11)
        ax.set_ylabel("Energy [eV]", weight="bold", size=11)
        ax.set_ylim(0,350)
        ax.grid(True)
        fig.tight_layout()
        plt.show()
