import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
import matplotlib.gridspec as gridspec
from Tkinter import *
from tkFileDialog import askopenfilename
from pypct import draw_target, get_initial_position, \
    save_result, draw_collision, draw_energy_loss
import cypct as cy
from time import clock
import os

class AppMain(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        self.master = master
        self.pack(fill=BOTH, expand=1)
        self.master.title("Particle Tracing for Magnetron Sputter ver.0.1")

        ########################################################################
        ### Simulation Default Values:
        self.para = {}
        self.para_default = {"voltage":-300,   # Cathode Voltage (V)
                             "sheath":0.001,   # Sheath Thickness (m)
                             "scale":1.0,      # Magnet Scale
                             "torr":0.005,     # Pressure (torr)
                             "kelvin":300,     # Temperature (K)
                             "tstep":2.0E-11,  # Time Step Size (s)
                             "tnum":100000,    # Max. Number of Time Steps
                             "enum":1}         # Number of Electrons
        self.outfile = ""
        self.outfile_default = "pct_result.txt"
                
        self.selection = []
        self.is_trajectory = BooleanVar()

        self.isnewfig = False

        self.xlim = [-0.11,0.11]
        self.ylim = [-0.04,0.04]
        self.zlim = [-0.02,0.02]        
        ########################################################################
        ### Make Simulator Layout
        frame_btn1 = Frame(self)
        frame_btn1.grid(row=0, column=0, sticky=W+N+E)
        frame_top= Frame(self)
        frame_top.grid(row=0, column=1, sticky=W+N+E)
        frame_opt = Frame(self)
        frame_opt.grid(row=1, column=0, sticky=W+N)
        frame_fig = Frame(self)
        frame_fig.grid(row=1, column=1)
        frame_btn2 = Frame(self)
        frame_btn2.grid(row=2, column=0, sticky=W+N+E)

        labelfont = ("Consolas",10,"bold")

        ### frame_btn1
        Button(frame_btn1, text="Run", command=self.cmd_run).pack(fill=X)

        ### frame_btn2
        Button(frame_btn2, text="Show Maps", command=self.cmd_showmap).pack(fill=X)

        ### frame_top
        Button(frame_top, text="  Clear  ", command=self.cmd_clear).pack(side=LEFT)
        Button(frame_top, text="Redraw", command=self.cmd_redraw).pack(side=LEFT)
        Label(frame_top, text="Output Filename>>..result\\", \
            font=labelfont).pack(side=LEFT)
        self.en_outfile = Entry(frame_top,width=80)
        self.en_outfile.pack(side=LEFT)
        Button(frame_top, text="Quit", command=self.cmd_quit).pack(side=RIGHT)

        ### frame_fig
        self.fig = plt.figure(figsize=(13,6.5), facecolor="white")
        self.fig = plt.figure(1)

        self.ax1 = self.fig.add_subplot(221)
        #self.ax2 = self.fig.add_subplot(222)
        self.ax3 = self.fig.add_subplot(223)
        self.ax4 = self.fig.add_subplot(224)
        
        self.ax1, self.ax3, self.ax4 = draw_target(self.ax1,self.ax3,self.ax4,
                                            self.xlim, self.ylim, self.zlim)

        self.ax1.set_aspect("equal")
        self.ax3.set_aspect("equal")
        self.ax4.set_aspect("equal")
        
        #self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, frame_fig)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(fill=BOTH, expand=1)

        ### frame_opt
        frame_sub0 = Frame(frame_opt, bd=2, relief=SUNKEN)
        frame_sub0.grid(row=0, column=0, sticky=W+N+E)
        frame_sub1 = Frame(frame_opt, bd=2, relief=SUNKEN)
        frame_sub1.grid(row=1, column=0, sticky=W+N+E)
        frame_sub2 = Frame(frame_opt, bd=2, relief=SUNKEN)
        frame_sub2.grid(row=2, column=0, sticky=W+N+E)
        frame_sub3 = Frame(frame_opt, bd=2, relief=SUNKEN)
        frame_sub3.grid(row=3, column=0, sticky=W+N+E)
        frame_sub4 = Frame(frame_opt, bd=2, relief=SUNKEN)
        frame_sub4.grid(row=4, column=0, sticky=W+S+E)

        self.opt_el = IntVar()
        self.opt_ex = IntVar() 
        self.opt_iz = IntVar()
        self.opt_trajectory = IntVar()

        ### frame_opt /frame_sub0
        Label(frame_sub0, text="Cathode Voltage", \
            font=labelfont).grid(row=0,column=0,columnspan=2,sticky=W)
        Label(frame_sub0, text="[V]", \
            font=labelfont).grid(row=1,column=1,sticky=W)
        Label(frame_sub0, text="Sheath Thickness", \
            font=labelfont).grid(row=2,column=0,columnspan=2,sticky=W)
        Label(frame_sub0, text="[m]", \
            font=labelfont).grid(row=3,column=1,sticky=W)
        Label(frame_sub0, text="Magnet Scale", \
            font=labelfont).grid(row=4,column=0,columnspan=2,sticky=W)
        Label(frame_sub0, text="X", \
            font=labelfont).grid(row=5,column=1,sticky=W)
        Label(frame_sub0, text="Pressure", \
            font=labelfont).grid(row=6,column=0,columnspan=2,sticky=W)
        Label(frame_sub0, text="[torr]", \
            font=labelfont).grid(row=7,column=1,sticky=W)
        Label(frame_sub0, text="Temperature", \
            font=labelfont).grid(row=8,column=0,columnspan=2,sticky=W)
        Label(frame_sub0, text="[K]", \
            font=labelfont).grid(row=9,column=1,sticky=W)
        self.en_voltage = Entry(frame_sub0,width=10)
        self.en_sheath  = Entry(frame_sub0,width=10)
        self.en_scale   = Entry(frame_sub0,width=10)
        self.en_torr    = Entry(frame_sub0,width=10)
        self.en_kelvin  = Entry(frame_sub0,width=10)
        self.en_voltage.grid(row=1,column=0,sticky=W)
        self.en_sheath.grid(row=3,column=0,sticky=W)
        self.en_scale.grid(row=5,column=0,sticky=W)
        self.en_torr.grid(row=7,column=0,sticky=W)
        self.en_kelvin.grid(row=9,column=0,sticky=W)

        ### frame_opt /frame_sub1
        Label(frame_sub1, text="Time Step Size",
              font=labelfont).grid(row=0,column=0,columnspan=2,sticky=W)
        Label(frame_sub1, text="[s]",
              font=labelfont).grid(row=1,column=1,sticky=W)
        Label(frame_sub1, text="Max. Time Steps",
              font=labelfont).grid(row=2,column=0,columnspan=2,sticky=W)
        self.en_tstep = Entry(frame_sub1,width=10)
        self.en_tnum  = Entry(frame_sub1,width=10)
        self.en_tstep.grid(row=1,column=0,sticky=W)
        self.en_tnum.grid(row=3,column=0,sticky=W)

        ### frame_opt /frame_sub2
        Label(frame_sub2, text="# of Electrons",
              font=labelfont).grid(row=0,column=0,columnspan=2,sticky=W)
        Label(frame_sub2, text="[ea]",
              font=labelfont).grid(row=1,column=1,sticky=W)
        self.en_enum = Entry(frame_sub2,width=10)
        self.en_enum.grid(row=1, column=0, sticky=W)

        ### frame_opt /frame_sub3
        Label(frame_sub3, text="[Options]",
              font=labelfont).grid(row=0,column=0,columnspan=2,sticky=W)
        Checkbutton(frame_sub3, text="Elastic",
                    variable=self.opt_el).grid(row=1,column=0,sticky=W)
        Checkbutton(frame_sub3, text="Excitation",
                    variable=self.opt_ex).grid(row=2,column=0,sticky=W)
        Checkbutton(frame_sub3, text="Ionization",
                    variable=self.opt_iz).grid(row=3,column=0,sticky=W)
        Checkbutton(frame_sub3, text="Show Trajectory",
                    variable=self.opt_trajectory).grid(row=5,column=0,sticky=W)

        ### frame_opt /frame_sub4
        Label(frame_sub4, text="Resolution",
              font=labelfont).grid(row=0,column=0,columnspan=2,sticky=W)
        Label(frame_sub4, text="[m]",
              font=labelfont).grid(row=1,column=1,sticky=W)
        Label(frame_sub4, text="Substrate Gap",
              font=labelfont).grid(row=2,column=0,columnspan=2,sticky=W)
        Label(frame_sub4, text="[m]",
              font=labelfont).grid(row=3,column=1,sticky=W)
        self.en_resolution = Entry(frame_sub4,width=10)
        self.en_gap        = Entry(frame_sub4,width=10)
        self.en_resolution.grid(row=1,column=0,sticky=W)
        self.en_gap.grid(row=3,column=0,sticky=W)

        ########################################################################
        ### Make Menu
        menu = Menu(self.master)
        self.master.config(menu=menu)

        menu_file = Menu(menu)
        menu.add_cascade(label="File", underline=0, menu=menu_file)
        menu_file.add_command(label="New", underline=0,
                              accelerator="Ctrl+N", command=self.cmd_new)
        menu_file.add_command(label="Run", underline=0,
                              accelerator="Ctrl+R", command=self.cmd_run)
        menu_file.add_command(label="Open", underline=0,
                              accelerator="Ctrl+O", command=self.cmd_open)
        menu_file.add_separator()
        menu_file.add_command(label="Quit", underline=1, accelerator="Ctrl+Q",
                              command=self.cmd_quit)

        self.master.bind("<Control-n>", self.cmd_new)
        self.master.bind("<Control-r>", self.cmd_run)
        self.master.bind("<Control-o>", self.cmd_open)
        self.master.bind("<Control-q>", self.cmd_quit)
        
        menu_help = Menu(menu)
        menu.add_cascade(label="Help", underline=0, menu=menu_help)
        menu_help.add_command(label="About", command=self.cmd_about)

        ########################################################################
        self.set_sim_para(self.para_default,self.outfile_default,0.002,0.05)
        self.set_sim_opt(["el","ex","iz"],False)
        #self.cmd_new()

    ############################################################################
    ### Menu/Button Action Command

    def cmd_new(self):   # Ctrl+N
        self.set_sim_para(self.para_default,self.outfile_default,0.002,0.05)
        self.set_sim_opt(["el","ex","iz"],False)
        self.cmd_clear()

    def cmd_run(self):   # Ctrl+R
        self.get_sim_para()
        self.get_sim_opt()
        filename = os.path.join("result",self.outfile)
        self.sim_run_save(self.para,self.is_trajectory,filename)
        self.sim_draw_result(filename)

    def cmd_open(self):  # Ctrl+O
        filename = askopenfilename(
            filetypes=[('.txt files','.txt'),('All files','*.*')],
            title="Select files", multiple=1)
        para = self.get_result_para(filename[0])

        self.get_sim_opt()
        self.cmd_clear()
        self.sim_draw_result(filename[0])

        outfile = os.path.basename(filename[0])
        self.set_sim_para(para,outfile,0.002,0.05)

    def cmd_quit(self): # Ctrl+Q
        root.quit()
        root.destroy()

    def cmd_about(self):
        top = Toplevel(self.master)
        content = ["Programmed using Python 2.7", "Powered by Cython"]
        AppMsg(top, '\n'.join(content))

    def cmd_clear(self):
        self.fig.clear()
        self.canvas.show()

        self.ax1 = self.fig.add_subplot(221)
        #self.ax2 = self.fig.add_subplot(222)
        self.ax3 = self.fig.add_subplot(223)
        self.ax4 = self.fig.add_subplot(224)

        
        self.ax1, self.ax3, self.ax4 = draw_target(self.ax1,self.ax3,self.ax4,
                                            self.xlim, self.ylim, self.zlim)
        self.ax1.set_aspect("equal")
        self.ax3.set_aspect("equal")
        self.ax4.set_aspect("equal")
        
        #self.fig.tight_layout()
        self.canvas.show()
        
    def cmd_clear_(self):
        self.fig.clear()

        if self.isnewfig == True:
            self.fig = plt.figure(figsize=(13,6.5), facecolor="white")
            self.isnewfig = False
            self.ax1 = plt.subplot2grid((5,4),(0,0),colspan=3,rowspan=3)
            self.ax2 = plt.subplot2grid((5,4),(0,3),colspan=1,rowspan=3)
            self.ax3 = plt.subplot2grid((5,4),(3,0),colspan=3,rowspan=2)
            self.ax4 = plt.subplot2grid((5,4),(3,3),colspan=1,rowspan=2)
            self.ax1, self.ax2, self.ax3 = draw_target(self.ax1,self.ax2,self.ax3,
                                            self.xlim, self.ylim, self.zlim)
            self.fig.tight_layout()
            self.canvas = BodyCanvas(self.frame_fig, self.fig)
        else:
            self.ax1 = plt.subplot2grid((5,4),(0,0),colspan=3,rowspan=3)
            self.ax2 = plt.subplot2grid((5,4),(0,3),colspan=1,rowspan=3)
            self.ax3 = plt.subplot2grid((5,4),(3,0),colspan=3,rowspan=2)
            self.ax4 = plt.subplot2grid((5,4),(3,3),colspan=1,rowspan=2)
            self.ax1, self.ax2, self.ax3 = draw_target(self.ax1,self.ax2,self.ax3,
                                            self.xlim, self.ylim, self.zlim)
            self.fig.tight_layout()
            self.canvas.show(self.fig)

    def cmd_redraw(self):
        self.get_sim_para()
        self.get_sim_opt()
        self.cmd_clear()
        self.sim_draw_result(os.path.join("result",self.outfile))

    def cmd_showmap(self):
        self.get_sim_para()
        self.get_sim_opt()

        txlim = sxlim = self.xlim
        tylim = sylim = self.ylim

        t0 = clock()
        filename = os.path.join("result",self.outfile)
        result = np.genfromtxt(filename, skip_header=10)
        map_target = cy.get_map_target(result, self.selection, self.resolution,
                                       txlim[0], txlim[1], tylim[0], tylim[1])
        t1 = clock()
        print(">>> Target map has been created.    >>> Time(s): %.2f" % (t1-t0))
        map_substrate = cy.get_map_substrate(map_target, self.gap, self.resolution,
                                          txlim[0], txlim[1], tylim[0], tylim[1],
                                          sxlim[0], sxlim[1], sylim[0], sylim[1])
        t2 = clock()
        print(">>> Substrate map has been created. >>> Time(s): %.2f" % (t2-t1))
        
        fig_map = plt.figure(figsize=(9,8), facecolor="white")
        ax1 = fig_map.add_subplot(211)
        ax2 = fig_map.add_subplot(212)
        ax1.imshow(map_target)
        ax2.imshow(map_substrate)
        ax1.set_title("Target - Erosion Map", weight="bold", size=14)
        ax2.set_title("Substrate - Deposition Map (Gap: %.3f m)" % self.gap, weight="bold", size=14)
        ax1.set_xlabel("X [pixel]", weight="bold", size=11)
        ax1.set_ylabel("Y [pixel]", weight="bold", size=11)
        ax2.set_xlabel("X [pixel]", weight="bold", size=11)
        ax2.set_ylabel("Y [pixel]", weight="bold", size=11)
        fig_map.tight_layout()
        
        top = Toplevel(self.master)
        #maps = BodyCanvas(top, fig_map)
        self.isnewfig = True
        maps = FigureCanvasTkAgg(fig_map, top)
        maps.show()
        maps.get_tk_widget().pack(fill=BOTH, expand=1)
        
    ############################################################################
    ### Simulation Process
    def sim_run_save(self, para, is_trajectory, filename):
        voltage = para["voltage"]
        sheath  = para["sheath"]
        scale   = para["scale"]
        torr    = para["torr"]
        kelvin  = para["kelvin"]
        tstep   = para["tstep"]
        tnum    = para["tnum"]
        enum    = para["enum"]
        
        E = cy.ElectricField(voltage, sheath)
        mag1 = cy.Magnet(-0.08,-0.01,-0.02,0.16,0.02,0.01, scale) # Center
        mag2 = cy.Magnet(-0.10,-0.02,-0.02,0.01,0.04,0.01,-scale) # Bottom
        mag3 = cy.Magnet(-0.10, 0.02,-0.02,0.20,0.01,0.01,-scale) # Right
        mag4 = cy.Magnet( 0.09,-0.02,-0.02,0.01,0.04,0.01,-scale) # Top
        mag5 = cy.Magnet(-0.10,-0.03,-0.02,0.20,0.01,0.01,-scale) # Left
        B = cy.MagneticField([mag1, mag2, mag3, mag4, mag5])
        db = np.genfromtxt("ArCrossSections.csv",delimiter=',',skip_header=3)
        ng = torr*133.32*6.022E23/8.314/kelvin

        p0_list, v0_list = get_initial_position(enum)
        max_num = num_electron = len(p0_list)
        time_list, pos_list, vel_list, coll_list, num_list = [],[],[],[],[]

        print ">>> ..."
        print ">>> ..."
        print ">>> ..."
        print "="*70
        print ">>> Simulation of particle Tracing has been started. Wait..."
        print "="*70
        tstart = clock()

        for i in xrange(max_num):
            p0, v0 = p0_list.pop(), v0_list.pop()
            num_electron -= 1
            tstart_i = clock()
            result = cy.trace_single(p0,v0,E,B,db,ng,tstep,tnum,is_trajectory)
            tend_i = clock()

            num_step = result[-1]
            num_el   = result[3].count(0)
            num_ex   = result[3].count(1)
            num_iz   = result[3].count(2)

            if max_num <= 10:
                print ">>> [%s/%s] el: %2d, ex: %2d, iz: %2d >>> Steps: %5d, Time(s): %.2f" \
                    % (str(i+1).rjust(len(str(max_num))), str(max_num), \
                    num_el, num_ex, num_iz, num_step,tend_i-tstart_i)
            elif i % 10 == 0:
                print ">>> [%s/%s] el: %2d, ex: %2d, iz: %2d >>> Steps: %5d, Time(s): %.2f" \
                    % (str(i+1).rjust(len(str(max_num))), str(max_num), \
                    num_el, num_ex, num_iz, num_step,tend_i-tstart_i)

            time_list += result[0]
            pos_list  += result[1]
            vel_list  += result[2]
            coll_list += result[3]
            num_list  += [[i+1] for j in range(len(result[3]))]

        print ">>> Total Simulation Time(s): %.2f" % (clock()-tstart)
        tt = np.array(time_list)
        pp = np.concatenate(pos_list).reshape(-1,3)
        vv = np.concatenate(vel_list).reshape(-1,3)
        cc = np.array(coll_list)
        nn = np.array(num_list)
        save_result(filename,tt,pp,vv,cc,nn,para)

    def sim_draw_result(self,filename):
        para = self.get_result_para(filename)
        voltage = para["voltage"]
        sheath  = para["sheath"]
        result = np.genfromtxt(filename,skip_header=10)
        if self.isnewfig == True:
            self.fig.clear()
            self.fig = plt.figure(figsize=(13,6.5), facecolor="white")
            self.isnewfig = False
            self.ax1 = plt.subplot2grid((5,4),(0,0),colspan=3,rowspan=3)
            self.ax2 = plt.subplot2grid((5,4),(0,3),colspan=1,rowspan=3)
            self.ax3 = plt.subplot2grid((5,4),(3,0),colspan=3,rowspan=2)
            self.ax4 = plt.subplot2grid((5,4),(3,3),colspan=1,rowspan=2)
            self.ax1,self.ax2,self.ax3 = draw_collision(self.ax1,self.ax2,self.ax3,\
                    result,self.selection,self.is_trajectory)
            self.ax4 = draw_energy_loss(self.ax4,result,voltage,sheath)
            self.fig.tight_layout()
            self.canvas = BodyCanvas(self.frame_fig, self.fig)
        else:
            self.ax1,self.ax2,self.ax3 = draw_collision(self.ax1,self.ax2,self.ax3,\
                    result,self.selection,self.is_trajectory)
            self.ax4 = draw_energy_loss(self.ax4,result,voltage,sheath)
            self.fig.tight_layout()
            self.canvas.show(self.fig)

    def get_result_para(self,filename):
        comments = open(filename,'r').readlines()[1:9]
        para = {}
        para["voltage"] = float(comments[0].split(':')[-1].strip())
        para["sheath"]  = float(comments[1].split(':')[-1].strip())
        para["scale"]   = float(comments[2].split(':')[-1].strip())
        para["torr"]    = float(comments[3].split(':')[-1].strip())
        para["kelvin"]  = float(comments[4].split(':')[-1].strip())
        para["tstep"]   = float(comments[5].split(':')[-1].strip())
        para["tnum"]    =  long(comments[6].split(':')[-1].strip())
        para["enum"]    =  long(comments[7].split(':')[-1].strip())
        return para

    def get_sim_para(self):
        self.para["voltage"] = float(self.en_voltage.get())
        self.para["sheath"]  = float(self.en_sheath.get())
        self.para["scale"]   = float(self.en_scale.get())
        self.para["torr"]    = float(self.en_torr.get())
        self.para["kelvin"]  = float(self.en_kelvin.get())
        self.para["tstep"]   = float(self.en_tstep.get())
        self.para["tnum"]    =  long(self.en_tnum.get())
        self.para["enum"]    =  long(self.en_enum.get())
        self.outfile         = self.en_outfile.get()
        self.resolution      = float(self.en_resolution.get())
        self.gap             = float(self.en_gap.get())

    def get_sim_opt(self):
        self.selection = []
        if self.opt_el.get() == 1:
            self.selection.append("el")
        if self.opt_ex.get() == 1:
            self.selection.append("ex")
        if self.opt_iz.get() == 1:
            self.selection.append("iz")

        if self.opt_trajectory.get() == 1:
            self.is_trajectory = True
        else:
            self.is_trajectory = False

    def set_sim_para(self, para, outfile, resolution, gap):
        self.para["voltage"] = para["voltage"]
        self.para["sheath"]  = para["sheath"]
        self.para["scale"]   = para["scale"]
        self.para["torr"]    = para["torr"]
        self.para["kelvin"]  = para["kelvin"]
        self.para["tstep"]   = para["tstep"]
        self.para["tnum"]    = para["tnum"]
        self.para["enum"]    = para["enum"]
        self.outfile         = self.en_outfile.get()
        self.resolution      = self.en_resolution.get()
        self.gap             = self.en_gap.get()
        
        self.en_voltage.delete(0, END)
        self.en_sheath.delete(0, END)
        self.en_scale.delete(0, END)
        self.en_torr.delete(0, END)
        self.en_kelvin.delete(0, END)
        self.en_tstep.delete(0, END)
        self.en_tnum.delete(0, END)
        self.en_enum.delete(0, END)
        self.en_outfile.delete(0, END)
        self.en_resolution.delete(0, END)
        self.en_gap.delete(0, END)
        
        self.en_voltage.insert(END,para["voltage"])
        self.en_sheath.insert(END, para["sheath"])
        self.en_scale.insert(END,  para["scale"])
        self.en_torr.insert(END,   para["torr"])
        self.en_kelvin.insert(END, para["kelvin"])
        self.en_tstep.insert(END,  para["tstep"])
        self.en_tnum.insert(END,   para["tnum"])
        self.en_enum.insert(END,   para["enum"])
        self.en_outfile.insert(END,outfile)
        self.en_resolution.insert(END,resolution)
        self.en_gap.insert(END,gap)

    def set_sim_opt(self, selection, is_trajectory):
        self.selection = selection
        self.is_trajectory = is_trajectory

        if "el" in self.selection:
            self.opt_el.set(1)
        else:
            self.opt_el.set(0)
        if "ex" in self.selection:
            self.opt_ex.set(1)
        else:
            self.opt_ex.set(0)
        if "iz" in self.selection:
            self.opt_iz.set(1)
        else:
            self.opt_iz.set(0)
        if is_trajectory:
            self.opt_trajectory.set(1)
        else:
            self.opt_trajectory.set(0)

class BodyCanvas(Frame):
    def __init__(self, master, content):
        Frame.__init__(self, master)
        self.master = master
        self.pack(fill=BOTH, expand=1)
        self.content = content

        frame_canvas = Frame(master)
        frame_canvas.pack(fill=BOTH, expand=1)

        self.body = FigureCanvasTkAgg(self.content, frame_canvas)
        self.body.show()
        #self.body.draw()
        self.body.get_tk_widget().pack(fill=BOTH, expand=1)
        toolbar = NavigationToolbar2TkAgg(self.body, frame_canvas)
        toolbar.update()
        self.body._tkcanvas.pack(fill=BOTH, expand=1)

    def show(self, content):
        self.content = content
        self.body.draw()

class AppMsg(Frame):
    def __init__(self, master, msg):
        Frame.__init__(self, master)
        self.master = master
        self.pack(fill=BOTH, expand=1)

        message = Label(master, text=msg, justify=LEFT, font=("Consolas", 10))
        message.pack()
        button_quit = Button(master, text="Close", command=self.master.destroy)
        button_quit.pack()

if __name__ == "__main__":
    root = Tk()
    app = AppMain(root)
    root.mainloop()
