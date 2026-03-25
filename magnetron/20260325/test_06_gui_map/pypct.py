from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cypct as cy
from time import clock
import os

########## Global constants for electrons:
E_CHARGE = 1.602E-19 # [C]
E_MASS   = 9.109E-31 # [Kg]

def ke_vec(vx, vy, vz):
    return 0.5*E_MASS*(vx**2 + vy**2 + vz**2)/E_CHARGE

def pe_vec(x, y, z, vol, sh):
    return -0.5*(-np.sign(z-sh)+1)*vol*(z-sh)**2/sh**2

def get_initial_position(num):
    p0_list, v0_list = [], []
    for i in xrange(num):
        p0, v0 = np.zeros(3), np.zeros(3)
        p0[0] = np.random.uniform(-0.095,0.095)
        p0[1] = np.random.uniform(-0.025,0.025)
        p0_list.append(p0)
        v0_list.append(v0)

    return p0_list, v0_list

def save_result(filename, tt, pp, vv, cc, nn, para):
    def exit_dir(dirname):
        for root,dirs,files in os.walk(os.getcwd()):
            if dirname in dirs: return True
    comments  = "[Simulation Input Parameters]\n"
    comments += "%s%.1f\n" % ("Cathode Voltage(V) : ",para["voltage"])
    comments += "%s%.3f\n" % ("Sheath Thickness(m): ", para["sheath"])
    comments += "%s%.1f\n" % ("Magnet Scale       : ",  para["scale"])
    comments += "%s%.3f\n" % ("Pressure(torr)     : ",   para["torr"])
    comments += "%s%.1f\n" % ("Temperature(K)     : ", para["kelvin"])
    comments += "%s%.2e\n" % ("Time Step Size(s)  : ",  para["tstep"])
    comments += "%s%d\n"   % ("Max. Time Steps    : ",   para["tnum"])
    comments += "%s%d\n"   % ("Num. Electrons     : ",   para["enum"])
    header = "#_electron t[s] x[m] y[m] z[m] vx[m/s] vy[m/s] vz[m/s] collision_type"
    fmt    = "%d %.2e %.2e %.2e %.2e %.2e %.2e %.2e %d"
    result = np.column_stack((nn,tt,pp,vv,cc))
    if not exit_dir("result"): os.mkdir("result")
    #np.savetxt(os.path.join("result",filename), result, fmt=fmt, \
    #           header=header)
    np.savetxt(filename, result, fmt=fmt, header=header, comments=comments)
    print ">>> %s has been saved." % os.path.join("result",filename)

def draw_target(ax1, ax3, ax4, xlim, ylim, zlim):
    ax1.add_patch(patches.Rectangle((-0.08,-0.01),0.16,0.02,\
        facecolor="gray", alpha=0.3 ))
    ax1.add_patch(patches.Rectangle((-0.10,-0.02),0.01,0.04,\
        facecolor="gray", alpha=0.3 ))
    ax1.add_patch(patches.Rectangle((-0.10, 0.02),0.20,0.01,\
        facecolor="gray", alpha=0.3 ))
    ax1.add_patch(patches.Rectangle(( 0.09,-0.02),0.01,0.04,\
        facecolor="gray", alpha=0.3 ))
    ax1.add_patch(patches.Rectangle((-0.10,-0.03),0.20,0.01,\
        facecolor="gray", alpha=0.3 ))
    ax1.set_xlabel("X [m]", weight="bold", size=11)
    ax1.set_ylabel("Y [m]", weight="bold", size=11)
    ax1.set_xlim(xlim)
    ax1.set_xticks([-0.10+0.02*i for i in range(11)])
    ax1.set_ylim(ylim)
    ax1.grid(True)

    ax4.add_patch(patches.Rectangle((-0.02,-0.03),0.01,0.01,\
        facecolor="gray", alpha=0.3 ))
    ax4.add_patch(patches.Rectangle((-0.02,-0.01),0.01,0.02,\
        facecolor="gray", alpha=0.3 ))
    ax4.add_patch(patches.Rectangle((-0.02, 0.02),0.01,0.01,\
        facecolor="gray", alpha=0.3 ))
    ax4.set_ylabel("Z [m]", weight="bold", size=11)
    ax4.set_xlabel("Y [m]", weight="bold", size=11)
    ax4.set_ylim(zlim)
    ax4.set_yticks([-0.02,-0.01,0,0.01,0.02])
    ax4.set_xlim(ylim)
    ax4.grid(True)

    ax3.add_patch(patches.Rectangle((-0.10,-0.02),0.01,0.01,\
        facecolor="gray", alpha=0.3 ))
    ax3.add_patch(patches.Rectangle((-0.08,-0.02),0.16,0.01,\
        facecolor="gray", alpha=0.3 ))
    ax3.add_patch(patches.Rectangle(( 0.09,-0.02),0.01,0.01,\
        facecolor="gray", alpha=0.3 ))
    ax3.set_xlabel("X [m]", weight="bold", size=11)
    ax3.set_ylabel("Z [m]", weight="bold", size=11)
    ax3.set_xlim(xlim)
    ax3.set_xticks([-0.10+0.02*i for i in range(11)])
    ax3.set_ylim(zlim)
    ax3.set_yticks([-0.02,-0.01,0,0.01,0.02])
    ax3.grid(True)

    return ax1, ax3, ax4

def draw_collision(ax1, ax2, ax3, result, selection, is_trajectory):
    nn, pp, cc = result[:,0], result[:,2:5], result[:,-1]
    max_num = long(result[-1,0])

    if is_trajectory:
        for i in xrange(max_num):
            pp_i = pp[np.where(nn==i+1)]
            ax1.plot(pp_i[:,0], pp_i[:,1], 'k', lw=0.5)
            ax2.plot(pp_i[:,2], pp_i[:,1], 'k', lw=0.5)
            ax3.plot(pp_i[:,0], pp_i[:,2], 'k', lw=0.5)
    if "el" in selection:
        pp_el  = pp[np.where(cc==0)]
        ax1.plot(pp_el[:,0], pp_el[:,1], 'gx')
        ax2.plot(pp_el[:,2], pp_el[:,1], 'gx')
        ax3.plot(pp_el[:,0], pp_el[:,2], 'gx')
    if "ex" in selection:
        pp_ex  = pp[np.where(cc==1)]
        ax1.plot(pp_ex[:,0], pp_ex[:,1], 'bx')
        ax2.plot(pp_ex[:,2], pp_ex[:,1], 'bx')
        ax3.plot(pp_ex[:,0], pp_ex[:,2], 'bx')
    if "iz" in selection:
        pp_iz  = pp[np.where(cc==2)]
        ax1.plot(pp_iz[:,0], pp_iz[:,1], 'rx')
        ax2.plot(pp_iz[:,2], pp_iz[:,1], 'rx')
        ax3.plot(pp_iz[:,0], pp_iz[:,2], 'rx')

    return ax1, ax2, ax3


def draw_energy_loss(ax, result, voltage, sheath):
    nn, tt, pp, vv = result[:,0], result[:,1], result[:,2:5], result[:,5:8]
    max_num = long(result[-1,0])

    for i in xrange(max_num):
        pp_i = pp[np.where(nn==i+1)]
        vv_i = vv[np.where(nn==i+1)]
        tt_i = tt[np.where(nn==i+1)]
        pe = pe_vec(pp_i[:,0],pp_i[:,1],pp_i[:,2],voltage,sheath)
        ke = ke_vec(vv_i[:,0],vv_i[:,1],vv_i[:,2])
        #ax.plot(tt_i, pe+ke,'k',lw=0.5)
        ax.semilogx(tt_i, pe+ke,'k',lw=0.5)
    ax.set_xlabel("Time [s]", weight="bold", size=11)
    ax.set_ylabel("Energy [eV]", weight="bold", size=11)
    ax.grid(True)

    return ax

def get_map_target(result, selection, resolution, xmin, xmax, ymin, ymax):
    xx, yy, cc = result[:,2], result[:,3], result[:,8]
    dx = dy = resolution
    nx, ny = int((xmax-xmin)/dx), int((ymax-ymin)/dy)
    tmap = np.zeros((ny,nx))

    for k in xrange(cc.shape[0]):
        if ["el","ex","iz"][int(cc[k])] in selection:
            ti = int((ymax-yy[k])/dy)
            tj = int((xx[k]-xmin)/dx)
            if 0<=ti<ny and 0<=tj<nx:
                tmap[ti,tj] += 1
    return tmap

def get_map_substrate(tmap, gap, resolution, txmin, txmax, tymin, tymax,
                      sxmin, sxmax, symin, symax):
    tdx = tdy = resolution
    tnx, tny = int((txmax-txmin)/tdx), int((tymax-tymin)/tdy)
    sdx = sdy = resolution
    snx, sny = int((sxmax-sxmin)/sdx), int((symax-symin)/sdy)
    smap = np.zeros((sny,snx))

    for t in xrange(tmap.size):
        ti = t//tnx
        tj = t%tnx 
        tx = txmin + tdx*(tj+0.5)
        ty = tymax - tdy*(ti+0.5)
        if tmap[ti,tj] > 0:
            for s in xrange(smap.size):
                si = s//snx
                sj = s%snx
                sx = sxmin + sdx*(sj+0.5)
                sy = symax - sdy*(si+0.5)
                smap[si,sj] += tmap[ti,tj]*vf(tx,ty,sx,sy,gap)*tdx*tdy/sdx/sdy
    return smap

def vf(x1, y1, x2, y2, d):
    r2 = (x2-x1)**2 + (y2-y1)**2 + d**2
    return d**2/r2**2/np.pi


if __name__ == "__main__":

    if 0: # Collision Positions and Electron Trajectories
        is_trajectory = False
        result = np.genfromtxt(os.path.join("result","pct_result.txt"),skip_header=10)

        fig = plt.figure(figsize=(16,8))
        ax1 = plt.subplot2grid((5,6),(0,0),colspan=5,rowspan=3)
        ax2 = plt.subplot2grid((5,6),(0,5),rowspan=3)
        ax3 = plt.subplot2grid((5,6),(3,0),colspan=5,rowspan=2)

        ax1, ax2, ax3 = draw_target(ax1,ax2,ax3)
        ax1, ax2, ax3 = draw_collision(ax1,ax2,ax3,result,("el","ex","iz"),is_trajectory)

        fig.tight_layout()
        plt.show()
        
    if 0: # Electron Energy Loss due to Collision
        voltage = -300
        sheath  = 0.001
        result = np.genfromtxt(os.path.join("result","pct_result.txt"),skip_header=10)

        fig = plt.figure(figsize=(8,4))
        ax = fig.add_subplot(111)
        ax = draw_energy_loss(ax,result,voltage,sheath)
        fig.tight_layout()
        plt.show()

    if 0: # Erosion Map
        result = np.genfromtxt(os.path.join("result","pct_0.05torr_1000.txt"),
                               skip_header=10)
        #data = result[:,2:]
        tstart = clock()
        map_target = cy.get_map_target(result,["el","ex","iz"],0.001,-0.11,0.11,-0.04,0.04)
        #map_target = get_map_target(result,["el","ex","iz"],0.001)
        print clock()-tstart

        fig = plt.figure(figsize=(8,4))
        ax = fig.add_subplot(111)

        ax.imshow(map_target)
        fig.tight_layout()
        plt.show()

    if 1: # Deposition Rate Map
        txlim, tylim = [-0.11,0.11], [-0.04,0.04]
        sxlim, sylim = [-0.11,0.11], [-0.04,0.04]
        selection = ["el","ex","iz"]
        #selection = ["iz"]
        resolution = 0.002
        gap = 0.02
        #filename = os.path.join("result","pct_0.05torr_1000.txt")
        filename = os.path.join("result","pct_result_100.txt")

        t0 = clock()
        result = np.genfromtxt(filename, skip_header=10)
        map_target = cy.get_map_target(result, selection, resolution,
                                       txlim[0], txlim[1], tylim[0], tylim[1])

        t1 = clock()
        print "Target", t1-t0
        map_substrate = cy.get_map_substrate(map_target, gap, resolution,
                                          txlim[0], txlim[1], tylim[0], tylim[1],
                                          sxlim[0], sxlim[1], sylim[0], sylim[1])

        t2 = clock()
        print "Substrate", t2-t1
        
        fig = plt.figure(figsize=(8,8))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        ax1.imshow(map_target)
        ax2.imshow(map_substrate)

        fig.tight_layout()
        plt.show()
