from numpy import (
    convolve, ones, cos, sin, power, genfromtxt, arange, array, sqrt, pi,
    linspace, meshgrid, arctan2, rot90, transpose, loadtxt, reshape, mean
    )
#import numpy.matlib
from scipy.interpolate import interp1d, RegularGridInterpolator
import time
from scipy.integrate import quad
from joblib import Parallel, delayed, Memory
from math import ceil
import custom_minimizer 
from scipy.optimize import basinhopping

''' 
Functions and methods
'''

def moving_average(x, w):
    return convolve(x, ones(w), 'valid') / w #np.convolve

def pol2cart(rho, phi):
    x = rho * cos(phi) #np.cos
    y = rho * sin(phi) #np.sin
    return x, y

sqr = lambda x: power(x, 2) #np.power

def dep_profile(X, Y, center_x, center_y, C):
    #load('depline_Kaufman.mat')
    #load('ExpData/depline_exp_130mm.mat')
    depliney = genfromtxt('depliney.csv',delimiter=',') #np.genfromtxt
    profile_x_len = 300
    r = arange(0, profile_x_len, profile_x_len/len(depliney)) #np.arange
    f = interp1d(r, depliney, fill_value=depliney.min(),  #scipy.interpolate.interp1d
                             bounds_error=False)
    
    Z = f(sqrt(sqr(X+center_x)+sqr(Y+center_y)))#np.sqrt
    norm = Z.max()
    Z = C*Z/Z.max()
    t = (lambda xy: C*f(sqrt(sqr(xy[0]+center_x)+sqr(xy[1]+center_y)))/norm) #np.sqrt
    return Z, t

'''
####INPUTS####
'''

class Model:
    def __init__(self, 
                 filename='depz.txt', #path to dep profile
                 C=4.46,  #thickness [nm] per minute
                 source=0, #Choose source of get thickness data 1 - seimtra, 0 - experiment
                 val=3, #1, 2, 3 - magnetron position
                 substrate_x_len=100, # Substrate width, mm
                 substrate_y_len=100, # Substrate length, mm
                 substrate_x_res=0.05, # Substrate x resolution, 1/mm
                 substrate_y_res=0.05, # Substrate y resolution, 1/mm
                 cores=1, # number of jobs for paralleling
                 verbose=True,  # True: print message each time when function of deposition called
                 delete_cache=True, # True: delete history of function evaluations in the beggining 
                            #of work. Warning: if = False, some changes in the code may be ignored
                 point_tolerance=5/100, # needed relative tolerance for thickness in each point
                 max_angle_divisions = 10, # limit of da while integration = 1 degree / max_angle_divisions
                 holder_inner_radius = 20, # mm
                 holder_outer_radius = 145, # mm
                 deposition_len_x = 290, # mm
                 deposition_len_y = 290, # mm
                 deposition_res_x = 1, # 1/mm
                 deposition_res_y = 1, # 1/mm
                 R_step = 1, #mm
                 k_step = 0.01,
                 NR_step = 0.01,
                 R_min = 10, # mm
                 R_max = 70,
                 k_min = 1, 
                 k_max = 50, 
                 NR_min = 1,
                 NR_max = 100,
                 x0_1 = 35, #initial guess for optimisation [R0, k0]
                 x0_2 = 4.1,
                 x0_3 = 1,
                 minimizer = 'NM_custom',
                 R_mc_interval = 5/100, #step for MC <= R_mc_interval*(R_max_bound-R_min_bound)
                 k_mc_interval = 5/100, #step for MC <= k_mc_interval*(k_max_bound-k_min_bound)\
                 NR_mc_interval = 15/100,
                 R_min_step = 1, #step for MC >= R_min_step
                 k_min_step = 0.01, #step for MC >= k_min_step
                 NR_min_step = 1,
                 mc_iter = 2, # number of Monte-Carlo algoritm's iterations (number of visited local minima) 
                 T = 2 #"temperature" for MC algoritm
                 ):
        
        self.memory = Memory('cache', verbose=0)
        self.count = 0
        #sputter_profile = 'depline_Kaufman.mat'
        #sputter_profile = 'ExpData/depline_exp_130mm.mat'
        self.filename = filename
        self.C = C #thickness [nm] per minute
        self.source = source #Choose source of get thickness data 1 - seimtra, 0 - experiment
        self.val = val #1, 2, 3 - magnetron position
        self.alpha0_sub = 0*pi
        self.substrate_x_len = substrate_x_len # Substrate width, mm
        self.substrate_y_len = substrate_y_len # Substrate length, mm
        self.substrate_x_res = substrate_x_res # Substrate x resolution, 1/mm
        self.substrate_y_res = substrate_y_res # Substrate y resolution, 1/mm
        self.cores = cores # number of jobs for paralleling
        self.verbose = verbose # True: print message each time when function of deposition called
        self.delete_cache = delete_cache # True: delete history of function evaluations in the beggining 
                            #of work. Warning: if = False, some changes in the code may be ignored
        self.point_tolerance = point_tolerance # needed relative tolerance for thickness in each point
        self.max_angle_divisions = max_angle_divisions # limit of da while integration = 1 degree / max_angle_divisions
        self.holder_inner_radius = holder_inner_radius  # mm
        self.holder_outer_radius = holder_outer_radius  # radius sampleholder, mm
        self.deposition_len_x = deposition_len_x # mm
        self.deposition_len_y = deposition_len_y # mm
        self.deposition_res_x = deposition_res_x # 1/mm
        self.deposition_res_y = deposition_res_y # 1/mm
        self.R_step = R_step
        self.k_step = k_step
        self.NR_step = NR_step
        self.R_bounds = (R_min, R_max) # (min, max) mm
        self.k_bounds = (k_min, k_max) # (min, max)
        self.NR_bounds = (NR_min, NR_max)
        self.x0 = [x0_1, x0_2, x0_3] #initial guess for optimisation [R0, k0]
        NM = {"method":"Nelder-Mead", "options":{"disp": True, "xatol":0.01, 
                                                 "fatol":0.01, 'maxfev':200}, 
                                                 "bounds":(self.R_bounds, self.k_bounds, self.NR_bounds)}
        
        NM_custom = {"method":custom_minimizer.minimize_custom_neldermead,
                     "options":{"disp": True, "xatol":(0.1, 0.001, 0.01), 
                                                 "fatol":0.01, 'maxfev':200}, 
                                                 "bounds":(self.R_bounds, self.k_bounds, self.NR_bounds)}
        
        Powell = {"method":"Powell", "options":{"disp": True, "xtol":0.0001, 
                                                "ftol":0.01, 'maxfev':500, 
                                                "direc": array([[1,0.01, 0.1],[-1,0.01,-0.1]])}, #np.array
                                                "bounds":(self.R_bounds, self.k_bounds, self.NR_bounds)}
        minimizers = {'NM':NM, 'NM_custom':NM_custom, 'Powell':Powell}
        self.minimizer = minimizers[minimizer]
        self.R_mc_interval = R_mc_interval #step for MC <= R_mc_interval*(R_max_bound-R_min_bound)
        self.k_mc_interval = k_mc_interval #step for MC <= k_mc_interval*(k_max_bound-k_min_bound)\
        self.NR_mc_interval = NR_mc_interval
        self.R_min_step = R_min_step #step for MC >= R_min_step
        self.k_min_step = k_min_step #step for MC >= k_min_step
        self.NR_min_step = NR_min_step
        self.mc_iter = mc_iter # number of Monte-Carlo algoritm's iterations (number of visited local minima) 
        self.T = T #"temperature" for MC algoritm
        
        ####GEOMETRY + INITIALIZATION####

        if self.delete_cache: self.memory.clear(warn=False)
        else: print('WARNING: memory has not cleared, changes in the code or settings may be ignored')
        self.F_axial = False
        deposition_offset_x = -deposition_len_x/2 # mm
        deposition_offset_y = -deposition_len_y/2 # mm
        
        self.deposition_rect_x = [deposition_offset_x, deposition_offset_x+deposition_len_x, 
                             deposition_offset_x+deposition_len_x, deposition_offset_x, 
                             deposition_offset_x]
        
        self.deposition_rect_y = [deposition_offset_y, deposition_offset_y, 
                             deposition_offset_y+deposition_len_y, 
                             deposition_offset_y+deposition_len_y, deposition_offset_y]
            
        ang=arange(0, 2*pi,0.01) #np.aarange
        self.holder_circle_inner_x=holder_inner_radius*cos(ang) #np.cos
        self.holder_circle_inner_y=holder_inner_radius*sin(ang) #np.sin
        self.holder_circle_outer_x=holder_outer_radius*cos(ang) #np.cos
        self.holder_circle_outer_y=holder_outer_radius*sin(ang) #np.sin 
        
        #### depoition profile meshing
        deposition_coords_x = linspace(deposition_offset_x,                     #np.linspace
                                          deposition_offset_x+deposition_len_x, 
                                          num=ceil(deposition_len_x*deposition_res_x)) #math.ceil
        
        deposition_coords_y = linspace(deposition_offset_y, 
                                          deposition_offset_y+deposition_len_y, 
                                          num=ceil(deposition_len_y*deposition_res_y))
        
        self.deposition_coords_map_x, self.deposition_coords_map_y = meshgrid(deposition_coords_x, #np.meshgrid
                                                                       deposition_coords_y)
        
        substrate_coords_x = linspace(-substrate_x_len/2, substrate_x_len/2, 
                                         num=ceil(substrate_x_len*substrate_x_res))
        
        substrate_coords_y = linspace(-substrate_y_len/2, substrate_y_len/2, 
                                         num=ceil(substrate_y_len*substrate_y_res))
        
        self.substrate_coords_map_x, self.substrate_coords_map_y = meshgrid(substrate_coords_x, 
                                                                     substrate_coords_y)
        
        self.substrate_rect_x = [substrate_coords_x.min(), substrate_coords_x.max(), 
                            substrate_coords_x.max(), substrate_coords_x.min(), 
                            substrate_coords_x.min()]
        
        self.substrate_rect_y = [substrate_coords_y.max(), substrate_coords_y.max(), 
                            substrate_coords_y.min(), substrate_coords_y.min(), 
                            substrate_coords_y.max()]
        
        

        self.rho = sqrt(sqr(self.substrate_coords_map_x) + sqr(self.substrate_coords_map_y))
        self.alpha0 = arctan2(self.substrate_coords_map_y, self.substrate_coords_map_x) #np.arctan2
        self.ind = [(i, j) for i in range(len(self.substrate_coords_map_x)) for j in range(len(self.substrate_coords_map_x[i]))]
        #self.ind = [(i, len(self.substrate_coords_map_x[0])//2) for i in range(len(self.substrate_coords_map_x))]
        #self.ind = self.ind + [(len(self.substrate_coords_map_x)//2, i) for i in range(len(self.substrate_coords_map_x[0]))]
        
        if source == 1:
            RELdeposition_coords_map_z = rot90(loadtxt(filename, skiprows=1)) #np.rot90
            row_dep = RELdeposition_coords_map_z.max()
            self.deposition_coords_map_z = self.C*(RELdeposition_coords_map_z/row_dep)
        elif source == 0: 
            if val == 1:
                center_x, center_y = 80, 59
            elif val == 2:
                center_x, center_y = 80, -59
            elif val == 3:
                center_x, center_y = -105.8, 0
            else:
                raise ValueError(f'Incorrect magnetron position {val}.')
            self.deposition_coords_map_z, self.F = dep_profile(self.deposition_coords_map_x, 
                                                     self.deposition_coords_map_y, 
                                                     center_x, center_y, self.C)
            self.F_axial = True
        
        if not self.F_axial:
            self.F = RegularGridInterpolator((deposition_coords_x, deposition_coords_y),  #scipy.interpolate.RegularGridInterpolator
                                             transpose(self.deposition_coords_map_z), #np.transpose
                                             bounds_error=False)
        self.time_f = []
        if cores>1:
            self.deposition = self.memory.cache(self.deposition_parallel, ignore=['self'])
        elif cores==1:
            self.deposition = self.memory.cache(self.deposition_serial, ignore=['self'])           
        else: raise ValueError('incorrect parameter "cores"')
        
    def deposition_parallel(self, R, k, NR, omega):#parallel
                t0 = time.time()
                ########### INTEGRATION #################
                I, I_err = zip(*Parallel(n_jobs=self.cores)(delayed(self.calc)(ij, R, k, NR, omega) for ij in self.ind)) #parallel
                I = reshape(I, (len(self.substrate_coords_map_x), len(self.substrate_coords_map_x[0]))) #np.reshape
                t1 = time.time()
                self.time_f.append(t1-t0)
                if self.verbose: print('%d calculation func called. computation time: %.1f s' % (len(self.time_f), self.time_f[-1]))
                return I
            
    def deposition_serial(self, R, k, NR, omega): #serial
                t0 = time.time()
                ########### INTEGRATION #################
                I, I_err = zip(*[self.calc(ij, R, k, NR, omega) for ij in self.ind]) #serial
                I = reshape(I, (len(self.substrate_coords_map_x), len(self.substrate_coords_map_x[0])))
                t1 = time.time()
                self.time_f.append(t1-t0)
                if self.verbose: print('%d calculation func called. computation time: %.1f s' % (len(self.time_f), self.time_f[-1]))
                return I
        
    def print_fun(self, x, f, accepted):
        self.count+=1
        if accepted == 1: s = 'accepted'
        else: s = 'rejected' 
        print("\n##############\n%d/%d Monte-Carlo step: minimum %.2f at R = %.3f, k = %.3f, NR = %.1f was %s\n##############\n" % (self.count, self.mc_iter, f, *x, s))

    def xyp(self, i, j, a, R, k):
        x = R*cos(a+self.alpha0_sub)+self.rho[i,j]*cos(-a*k + self.alpha0[i,j])
        y = R*sin(a+self.alpha0_sub)+self.rho[i,j]*sin(-a*k + self.alpha0[i,j])
        return x, y
    
    def calc(self, ind, R, k, NR, omega):
        i, j = ind
        I, I_err = quad(lambda a: self.F(self.xyp(i, j, a, R, k)), 0, NR*2*pi, #scipy.integrate.quad
                                  limit=int(round(self.max_angle_divisions*360*NR)), 
                                  epsrel=self.point_tolerance)
        
        return I/(2*pi*omega), I_err/(2*pi*omega) #Jacobian time to alpha
    
    def func(self, x):
        c = 0
        gate=100
        delta=(0.1, 0.01)
        if x[0]<self.R_bounds[0]+delta[0]:
            c+=gate*(self.R_bounds[0]+delta[0]-x[0])
        if x[0]>self.R_bounds[1]-delta[0]:  
            c+=gate*(x[0]+delta[0]-self.R_bounds[0])
        if x[1]<self.k_bounds[0]+delta[1]:
            c+=gate*(self.k_bounds[0]+delta[1]-x[1])
        if x[1]>self.k_bounds[1]-delta[1]:  
            c+=gate*(x[1]+delta[1]-self.k_bounds[0])
        h = self.heterogeneity(self.deposition(*x, 1))
        if self.verbose: 
            print('At R = %.2f, k = %.3f, NR = %.2f ---------- heterogeneity = %.2f ' % (*x, h))
        return c+h
    
    def heterogeneity(self, I):
        h_1 = (1-I[len(I)//2,:].min()/I[len(I)//2,:].max())
        h_2 = (1-I[:,len(I[0])//2].min()/I[:,len(I[0])//2].max())
        return max(h_1, h_2)*100
    
    def optimisation(self):
        t0 = time.time()
        mytakestep = custom_minimizer.CustomTakeStep(
                                (self.R_bounds[1]-self.R_bounds[0])*self.R_mc_interval, 
                                (self.k_bounds[1]-self.k_bounds[0])*self.k_mc_interval, 
                                (self.NR_bounds[1]-self.NR_bounds[0])*self.NR_mc_interval, 
                                self.R_min_step, self.k_min_step, self.NR_min_step, 
                                self.R_bounds, self.k_bounds, self.NR_bounds)
        
        mybounds = custom_minimizer.CustomBounds(self.R_bounds, 
                                                 self.k_bounds, 
                                                 self.NR_bounds)
    
        ret = basinhopping(self.func, self.x0, minimizer_kwargs=self.minimizer, 
                           niter=self.mc_iter, callback=self.print_fun, 
                           take_step=mytakestep, T=self.T, accept_test=mybounds)
    
        R, k, NR = ret.x
        h = ret.fun #heterogeneity
        print("global minimum: R = %.1f, k = %.3f, NR = %.2f, heterogeneity = %.2f" % (R, k, NR, h))
        
        I = self.deposition(R, k, NR, 1)[0]
        t1 = time.time()
        print('Full time: %d s\nfunc calls: %d\navg func computation time: %.2f s' % (t1-t0, len(self.time_f), mean(self.time_f)))
        return I