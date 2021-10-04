from numpy import (
    convolve, ones, cos, sin, power, genfromtxt, arange, array, sqrt, pi,
    linspace, meshgrid, arctan2, rot90, transpose, loadtxt, log10, arcsin
    )
import numpy as np
from multiprocessing import Pool
from scipy.interpolate import interp1d, RegularGridInterpolator
import time
from scipy.integrate import quad
from joblib import Memory
from math import ceil
import custom_minimizer 
from scipy.optimize import basinhopping
from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5 import QtWidgets
#from ray.util.multiprocessing import Pool as Pool_ray
#import ray
import re
import os
import shutil
''' 
Functions and methods
'''
def error(msg):
    errorbox = QtWidgets.QMessageBox()
    errorbox.setText(msg)
    errorbox.exec_()
    
def message(msg):
    errorbox = QtWidgets.QMessageBox()
    errorbox.setText(msg)
    errorbox.exec_()
    
def moving_average(x, w):
    return convolve(x, ones(w), 'valid') / w #np.convolve

def pol2cart(rho, phi):
    x = rho * cos(phi) #np.cos
    y = rho * sin(phi) #np.sin
    return x, y

def sqr(x):
    return power(x, 2) #np.power

class interp_axial:
    def __init__(self, center_x, center_y, norm, f, C):
        self.center_x = center_x
        self.center_y = center_y
        self.norm = norm
        self.f = f
        self.C = C
    
    def F(self, xy):
        center_x = self.center_x
        center_y = self.center_y
        norm = self.norm
        f = self.f
        C = self.C
        return C*f(sqrt(sqr(xy[0]+center_x)+sqr(xy[1]+center_y)))/norm #np.sqrt


'''
####INPUTS####
'''

class Model(QObject):
    
    warn_signal = pyqtSignal(str,str)
    
    def update(self,
                 fname_sim, fname_exp, rotation_type, C, source, magnetron_x, 
                 magnetron_y, substrate_shape, substrate_radius, 
                 substrate_x_len, substrate_y_len, substrate_res, cores, 
                 verbose, delete_cache, point_tolerance, max_angle_divisions, 
                 holder_inner_radius, holder_outer_radius, deposition_len_x, 
                 deposition_len_y, R_step,
                 k_step, NR_step, R_extra_bounds, R_min, R_max, k_min, k_max, 
                 NR_min, NR_max, omega_s_max, omega_p_max, x0_1, x0_2, x0_3,
                 minimizer, R_mc_interval, k_mc_interval, NR_mc_interval,
                 R_min_step, k_min_step, NR_min_step, mc_iter, T, parent = None):
        
        super().__init__(parent)
        self.errorbox = QtWidgets.QErrorMessage()
        self.memory = Memory('cache', verbose=0)
        self.count = 0
        self.rotation_type = rotation_type
        #sputter_profile = 'depline_Kaufman.mat'
        #sputter_profile = 'ExpData/depline_exp_130mm.mat'
        self.fname_sim = fname_sim
        self.fname_exp = fname_exp
        self.C = C #thickness [nm] per minute
        self.magnetron_x = magnetron_x
        self.magnetron_y = magnetron_y
        self.source = source #Choose source of get thickness data 1 - seimtra, 0 - experiment
        self.alpha0_sub = 0*pi
        self.substrate_shape = substrate_shape
        if substrate_shape == 'Circle':
            self.substrate_radius = substrate_radius
            substrate_x_len = substrate_radius*2
            substrate_y_len = substrate_radius*2
        else:
            self.substrate_radius = sqrt(substrate_x_len**2+substrate_y_len**2)/2
        self.substrate_x_len = substrate_x_len # Substrate width, mm
        self.substrate_y_len = substrate_y_len # Substrate length, mm
        self.substrate_res = substrate_res # Substrate x resolution, 1/mm
        self.substrate_rows = ceil(substrate_y_len*substrate_res)
        self.substrate_columns = ceil(substrate_x_len*substrate_res)
        self.cores = cores # number of jobs for paralleling
        self.verbose = verbose # True: print message each time when function of deposition called
        self.delete_cache = delete_cache # True: delete history of function evaluations in the beggining 
                            #of work. Warning: if = False, some changes in the code may be ignored
        self.point_tolerance = point_tolerance/100 # needed relative tolerance for thickness in each point
        self.max_angle_divisions = max_angle_divisions # limit of da while integration = 1 degree / max_angle_divisions
        self.holder_inner_radius = holder_inner_radius  # mm
        self.holder_outer_radius = holder_outer_radius  # radius sampleholder, mm
        self.deposition_len_x = deposition_len_x # mm
        self.deposition_len_y = deposition_len_y # mm
        self.R_step = R_step
        R_decimals = int(log10(1/self.R_step))
        self.k_step = k_step
        self.NR_step = NR_step
        if rotation_type == 'Planet':
            R_min_self = holder_inner_radius+self.substrate_radius
            R_max_self = holder_outer_radius-self.substrate_radius
        else:
            R_min_self = holder_inner_radius+substrate_x_len/2
            R_max_self = holder_outer_radius*cos(arcsin(substrate_y_len/2/holder_outer_radius))-substrate_x_len/2
            k_min = 1
            k_max = 1
        if R_min_self>R_max_self:
            print(R_min_self, R_max_self, cos(arcsin(substrate_y_len/2/holder_outer_radius)))
            r = (holder_outer_radius-holder_inner_radius)/2
            if rotation_type == 'Solar':
                r_ = holder_inner_radius
                R_ = holder_outer_radius
                l = (4/5)*(sqrt(5*R_**2-r_**2)/2-r_)
            else:
                l = (holder_outer_radius-holder_inner_radius)/sqrt(2)
            error(f'При такой контрукции подложкодержателя и камеры максимально возможный радиус (для прямоугольной - полудиагональ) подложки {round(r, R_decimals+1)} мм.\nВ случае квадратной подложки максимальная длина стороны {round(l, R_decimals+1)} мм')
            self.success = False
            return False
        else:
            if R_extra_bounds:
                if R_min < R_min_self:
                    error(f'При такой конструкции подложкодержателя и размере подложки минимальный радиус {round(R_min_self, R_decimals+1)} мм. Это значение установленно автоматически.')
                if R_max > R_max_self:
                    error(f'При такой конструкции подложкодержателя и размере подложки максимальный радиус {round(R_max_self, R_decimals+1)} мм. Это значение установленно автоматически.')
        if R_extra_bounds and R_min>R_max:
            error('Верхняя граница R не может быть меньше нижней!')
            self.success = False
            return False
        if k_min>k_max:
            error('Верхняя граница k не может быть меньше нижней!')
            self.success = False
            return False
        if NR_min>NR_max:
            error('Верхняя граница NR не может быть меньше нижней!')
            self.success = False
            return False
        if R_extra_bounds:
            self.R_bounds = (max(R_min, R_min_self), min(R_max, R_max_self)) # (min, max) mm
        else: 
            self.R_bounds = (R_min_self, R_max_self)
        self.k_bounds = (k_min, k_max) # (min, max)
        self.NR_bounds = (NR_min, NR_max)
        self.omega_s_max = omega_s_max
        self.omega_p_max = omega_p_max
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
        self.R_mc_interval = R_mc_interval/100 #step for MC <= R_mc_interval*(R_max_bound-R_min_bound)
        self.k_mc_interval = k_mc_interval/100 #step for MC <= k_mc_interval*(k_max_bound-k_min_bound)\
        self.NR_mc_interval = NR_mc_interval/100
        self.R_min_step = R_min_step #step for MC >= R_min_step
        self.k_min_step = k_min_step #step for MC >= k_min_step
        self.NR_min_step = NR_min_step
        self.mc_iter = mc_iter # number of Monte-Carlo algoritm's iterations (number of visited local minima) 
        self.T = T #"temperature" for MC algoritm
        self.F_axial = False
        ####GEOMETRY + INITIALIZATION####

        if self.delete_cache: 
            try:
                self.memory.clear(warn=False)
                try:
                    dirpath = 'temp_cache'
                    if os.path.exists(dirpath) and os.path.isdir(dirpath):
                        shutil.rmtree(dirpath)
                except PermissionError:
                    error('Не получилось удалить "temp_cache"')
            except PermissionError as err:
                msg1 = 'Нет доступа к кэшу, не получилось стереть кэш:\n\n'
                msg2 = '\n\nДля текущей модели будет создана дополнительная папка "temp_cache".\nПеред следующим сеансом рекомендуется перезапустить Python//программу'
                error(msg1+str(err)+msg2)
                self.memory = Memory('temp_cache', verbose=0)
                try:
                    self.memory.clear(warn=False)
                except PermissionError:
                    error('Для новой папки произошла таже ошибка')
                    self.success = False
                    return False
        else: print('WARNING: memory has not cleared, changes in the code or settings may be ignored')
        
        
            
        ang=arange(0, 2*pi,0.01) #np.aarange
        self.holder_circle_inner_x=holder_inner_radius*cos(ang) #np.cos
        self.holder_circle_inner_y=holder_inner_radius*sin(ang) #np.sin
        self.holder_circle_outer_x=holder_outer_radius*cos(ang) #np.cos
        self.holder_circle_outer_y=holder_outer_radius*sin(ang) #np.sin 
        
        #### depoition profile meshing
  
        substrate_coords_x = linspace(-substrate_x_len/2, substrate_x_len/2, 
                                         num=self.substrate_columns)
        
        substrate_coords_y = linspace(-substrate_y_len/2, substrate_y_len/2, 
                                         num=self.substrate_rows)
        
        self.substrate_coords_map_x, self.substrate_coords_map_y = meshgrid(substrate_coords_x, 
                                                                     substrate_coords_y)
        
        if substrate_shape == 'Rectangle':
            self.substrate_rect_x = [substrate_coords_x.min(), substrate_coords_x.max(), 
                                substrate_coords_x.max(), substrate_coords_x.min(), 
                                substrate_coords_x.min()]
            
            self.substrate_rect_y = [substrate_coords_y.max(), substrate_coords_y.max(), 
                                substrate_coords_y.min(), substrate_coords_y.min(), 
                                substrate_coords_y.max()]
            
            self.rho = sqrt(sqr(self.substrate_coords_map_x) + sqr(self.substrate_coords_map_y))
            self.alpha0 = arctan2(self.substrate_coords_map_y, self.substrate_coords_map_x) #np.arctan2
            self.rho = self.rho.ravel()
            self.alpha0 = self.alpha0.ravel()
            self.xs = self.substrate_coords_map_x.ravel()
            self.ys = self.substrate_coords_map_y.ravel()            
        else:
            a = linspace(0, 2*pi, num=45)
            self.substrate_rect_x = self.substrate_radius*cos(a)
            self.substrate_rect_y = self.substrate_radius*sin(a)
            r_ = self.substrate_radius
            m = ceil(r_*self.substrate_res)
            if m<2:
                m+=1
            rho = linspace(0, r_, num=m)
            rho = rho[rho>0]
            angles = []
            rs = []
            for r in rho:
                n = ceil(2*pi*r*self.substrate_res)
                if n<4:
                    n = 4
                a = linspace(0,2*pi, num=n+1)
                a = a[a<2*pi]
                angles.append(a)
                rs.append([r]*n)
            alpha0 = np.concatenate(angles)
            rho = np.concatenate(rs)   
            #self.rho, self.alpha0 = np.meshgrid(rho, alpha0)
            self.rho = rho.ravel()
            self.alpha0 = alpha0.ravel()
            self.rho = np.concatenate(([0], self.rho))
            self.alpha0 = np.concatenate(([0], self.alpha0))
            self.xs, self.ys = pol2cart(self.rho, self.alpha0) 

        self.ind = list(range(len(self.xs)))
        '''
        import matplotlib.pyplot as plt
        plt.plot(self.xs, self.ys, 'x')
        for i in self.ind:
            plt.text(self.xs[i], self.ys[i], str(i))
        '''
        #self.ind = [(i, len(self.substrate_coords_map_x[0])//2) for i in range(len(self.substrate_coords_map_x))]
        #self.ind = self.ind + [(len(self.substrate_coords_map_x)//2, i) for i in range(len(self.substrate_coords_map_x[0]))]
        
        if source == 'SIMTRA':
            self.success = self.open_simtra_file(fname_sim)
            
        elif source == 'Experiment':
            self.success = self.open_exp_file(fname_exp)

        else: raise TypeError(f'incorrect source {source}')
        if not self.success: 
            return None
        if not self.F_axial:
            self.F = RegularGridInterpolator((self.deposition_coords_x, 
                                              self.deposition_coords_y),  #scipy.interpolate.RegularGridInterpolator
                                              transpose(self.deposition_coords_map_z), #np.transpose
                                              bounds_error=False)

        joblib_ignore=['self']
        '''
        if rotation_type == 'Planet':
            self.xyp = self.xyp_planet
        elif rotation_type == 'Solar':
            self.xyp = self.xyp_solar
            joblib_ignore.append('k')
        '''    
        self.time_f = []
        self.deposition = Deposition(self.rho, self.alpha0, self.F, self.cores)
        self.success = True
        
    def init_deposition_mesh(self, M=None, N=None, res_x=None, res_y=None):
        deposition_offset_x = -self.deposition_len_x/2 # mm
        deposition_offset_y = -self.deposition_len_y/2 # mm
        if not M: 
            assert res_x
            M = ceil(self.deposition_len_x*res_x)
        if not N: 
            assert res_y
            N = ceil(self.deposition_len_y*res_y)
        
        self.deposition_rect_x = [deposition_offset_x, 
                                  deposition_offset_x+self.deposition_len_x, 
                                  deposition_offset_x+self.deposition_len_x, 
                                  deposition_offset_x, 
                                  deposition_offset_x]
        
        self.deposition_rect_y = [deposition_offset_y, 
                                  deposition_offset_y, 
                                  deposition_offset_y+self.deposition_len_y, 
                                  deposition_offset_y+self.deposition_len_y, 
                                  deposition_offset_y]
        
        self.deposition_coords_x = linspace(deposition_offset_x,                     #np.linspace
                                       deposition_offset_x+self.deposition_len_x, 
                                       num=M) #math.ceil
        
        self.deposition_coords_y = linspace(deposition_offset_y, 
                                       deposition_offset_y+self.deposition_len_y, 
                                       num=N)
        
        self.deposition_coords_map_x, self.deposition_coords_map_y = meshgrid(self.deposition_coords_x, #np.meshgrid
                                                                       self.deposition_coords_y)
             
    def open_simtra_file(self, fname):
        try:
            with open(fname, 'r') as f:
                line = list(f)[0]
                M, N, I_tot, s = re.split(r'\t', line)
                assert s == 'Number of particles\n'
                M, N, I_tot = int(M), int(N), int(I_tot)
        except FileNotFoundError: 
            success = False
            error(f"Файл {fname} не найден")
        except: 
            success = False
            error("Неверный формат файла с результатами расчёта SIMTRA")
        else:
            self.init_deposition_mesh(M=M, N=N)
            RELdeposition_coords_map_z = rot90(loadtxt(fname, skiprows=1)) #np.rot90
            if RELdeposition_coords_map_z.shape != (M, N):
                success = False
                m, n = RELdeposition_coords_map_z.shape
                error(f"Неверно указана размерность сетки: {M}x{N} вместо {m}x{n}")
            
            msg = 'Согласно резудьтатам расчёта SIMTRA:\n {} % потока осаждено на заданную поверхность'.format(round(100*RELdeposition_coords_map_z.sum()/I_tot))
            type = 'simtra'
            self.warn_signal.emit(msg, type)
            message(msg)
            row_dep = RELdeposition_coords_map_z.max()
            self.deposition_coords_map_z = self.C*(RELdeposition_coords_map_z/row_dep)
            success = True

        return success
            
    def open_exp_file(self, fname):
        try:
            with open(fname, 'r') as f:
                r, h = genfromtxt(f, delimiter=',', unpack=True)
        except FileNotFoundError: 
            success = False
            error(f"Файл {fname} не найден")
        except:
            success = False
            error('Неверный формат файла с экспериментальным профилем напыления')
        else:
            dr = r[1:]-r[:-1]
            res = 1/dr.min()
            self.init_deposition_mesh(res_x=res, res_y=res)
            f = interp1d(r, h, fill_value='extrapolate', bounds_error=False)
            Z = f(sqrt(sqr(self.deposition_coords_map_x+self.magnetron_x)+
                       sqr(self.deposition_coords_map_y+self.magnetron_y)))#np.sqrt
            norm = Z.max()
            Z = self.C*Z/Z.max()
            assert Z.max()<=self.C
            interp = interp_axial(self.magnetron_x, self.magnetron_x, norm, f, self.C)
            self.deposition_coords_map_z = Z
            self.F = interp.F
            self.F_axial = True
            success = True
        return success

    def heterogeneity(self, I):
        return (1-I.min()/I.max())*100
    
class Deposition(QObject):
    
    def __init__(self, rho, alpha, F, njobs=1, parent=None):
        super().__init__(parent)
        self.time = []
        n = len(rho)
        rho_p = []
        alpha_p = []
        if njobs>n:
            njobs = n
        m = n//njobs
        if n%njobs == 0:
            for i in range(njobs):
                rho_p.append(rho[i*m:(i+1)*m])
                alpha_p.append(alpha[i*m:(i+1)*m])
        else:
            k = njobs-n%njobs
            m+=1
            for i in range(njobs-k):
                rho_p.append(rho[i*m:(i+1)*m])
                alpha_p.append(alpha[i*m:(i+1)*m])
            i0 = m*(njobs-k)
            m-=1
            for i in range(k):
                rho_p.append(rho[i0+i*m:i0+(i+1)*m])
                alpha_p.append(alpha[i0+i*m:i0+(i+1)*m])            
            
        self.workers = [Worker(F, rho_p[i], alpha_p[i]) for i in range(njobs)]
            
        
    def __call__(self, R, k, NR, omega, alpha0_sub, point_tolerance, 
                 max_angle_divisions, cores):
        t0 = time.time()
        if len(self.workers) == 1:
            self.workers[0].set_properties(R, k, NR, omega, alpha0_sub,
                                           point_tolerance, max_angle_divisions)
            self.hs = self.workers[0]()
            t = time.time()-t0
            self.time.append(t)
            return self.hs
            
        hs = []
        self.procs = []
        for worker in self.workers:
            worker.set_properties(R, k, NR, omega, alpha0_sub,
                                  point_tolerance, max_angle_divisions)

        result = []
        with Pool(processes=len(self.workers)) as pool:      
            for worker in self.workers:
                a = pool.apply_async(worker)
                result.append(a)
            hs = [result[i].get() for i in range(len(self.workers)) ]
        self.hs = np.concatenate(hs)
        t = time.time()-t0
        self.time.append(t)
        return self.hs
    

class Worker:
    def __init__(self, F, rho, alpha):
        self.F = F
        self.rho = rho
        self.alpha = alpha
        self.ind = range(len(rho))
        self.hs = None
        
    def set_properties(self, R, k, NR, omega, alpha0_sub,
                       point_tolerance, max_angle_divisions):
        self.R = R
        self.k = k
        self.NR = NR
        self.omega = omega
        self.alpha0_sub = alpha0_sub
        self.max_angle_divisions = max_angle_divisions
        self.point_tolerance = point_tolerance
        
    def xyp(self, a, i):
        x = self.R*cos(a)+self.rho[i]*cos(a*self.k + self.alpha[i])
        y = self.R*sin(a)+self.rho[i]*sin(a*self.k + self.alpha[i])
        return x, y
        
    def __call__(self):
        hs = [quad(lambda a: self.F(self.xyp(a, i)), 
                   self.alpha0_sub, self.NR*2*pi,
                   limit=int(round(self.max_angle_divisions*360*self.NR)), 
                   epsrel=self.point_tolerance)[0] for i in self.ind]
        hs = np.array(hs)/(2*pi*self.omega)
        return hs
    

class Optimizer(QObject):
    upd_signal = pyqtSignal(str)
    def __init__(self, deposition, parent=None):
        super().__init__(parent)
        self.deposition = deposition
        
    def optimisation(self, heterogeneity,
                     alpha0_sub, point_tolerance, max_angle_divisions, cores, 
                     R_bounds, k_bounds, NR_bounds, 
                     R_min_step, k_min_step, NR_min_step, 
                     R_step, k_step, NR_step,
                     R_mc_interval, k_mc_interval, NR_mc_interval, x0, 
                     minimizer, mc_iter, T, verbose):
        t0 = time.time()
        self.heterogeneity = heterogeneity
        self.count = 0
        self.mc_count = 0
        self.log = ''
        self.alpha0_sub = alpha0_sub
        self.point_tolerance = point_tolerance
        self.max_angle_divisions = max_angle_divisions
        self.cores = cores
        self.R_bounds = R_bounds
        self.k_bounds = k_bounds
        self.NR_bounds = NR_bounds
        self.R_step = R_step
        self.k_step = k_step
        self.NR_step = NR_step
        self.R_min_step = R_min_step
        self.k_min_step = k_min_step
        self.NR_min_step = NR_min_step
        self.R_mc_interval = R_mc_interval
        self.k_mc_interval = k_mc_interval
        self.NR_mc_interval = NR_mc_interval
        self.x0 = x0
        self.minimizer = minimizer
        self.mc_iter = mc_iter
        self.T = T
        self.verbose = verbose
        
        takestep = custom_minimizer.CustomTakeStep(self.R_mc_interval, 
                                                     self.k_mc_interval, 
                                                     self.NR_mc_interval, 
                                                     self.R_min_step, 
                                                     self.k_min_step, 
                                                     self.NR_min_step, 
                                                     self.R_bounds, 
                                                     self.k_bounds, 
                                                     self.NR_bounds)
        
        bounds = custom_minimizer.CustomBounds(self.R_bounds, 
                                                 self.k_bounds, 
                                                 self.NR_bounds)
    
        ret = basinhopping(self.func, self.x0, minimizer_kwargs=self.minimizer, 
                           niter=self.mc_iter, callback=self.print_fun, 
                           take_step=takestep, T=self.T, accept_test=bounds)
    
        R, k, NR = ret.x
        h = ret.fun #heterogeneity
        message = "global minimum: R = %.1f, k = %.3f, NR = %.2f, heterogeneity = %.2f" % (R, k, NR, h)    
        t1 = time.time()        
        message +='\nFull time: %d s\nfunc calls: %d\navg func computation time: %.2f s' % (t1-t0, self.count, (t1-t0)/self.count)
        self.log += (message+'\n')
        self.upd_signal.emit(message)
        return True    
    
    def func(self, x):
        self.count += 1
        c = 0
        gate=100
        delta=(self.R_step*2, self.k_step*2)
        if x[0]<self.R_bounds[0]+delta[0]:
            c+=gate*(self.R_bounds[0]+delta[0]-x[0])
        if x[0]>self.R_bounds[1]-delta[0]:  
            c+=gate*(x[0]+delta[0]-self.R_bounds[0])
        if x[1]<self.k_bounds[0]+delta[1]:
            c+=gate*(self.k_bounds[0]+delta[1]-x[1])
        if x[1]>self.k_bounds[1]-delta[1]:  
            c+=gate*(x[1]+delta[1]-self.k_bounds[0])
        args = [*x, 1, self.alpha0_sub, self.point_tolerance, 
                self.max_angle_divisions, self.cores]
        h = self.heterogeneity(self.deposition(*args))
        if self.verbose: 
            message = 'At R = %.2f, k = %.3f, NR = %.2f ---------- heterogeneity = %.2f ' % (*x, h)
            self.log += (message+'\n')
            self.upd_signal.emit(message)
            
        return c+h
    
    def print_fun(self, x, f, accepted):
        self.mc_count+=1
        if accepted == 1: s = 'accepted'
        else: s = 'rejected' 
        message = "\n##############\n%d/%d Monte-Carlo step: minimum %.2f at R = %.3f, k = %.3f, NR = %.1f was %s\n##############\n" % (self.mc_count, 1+self.mc_iter, f, *x, s)
        self.log += (message+'\n')
        self.upd_signal.emit(message)
