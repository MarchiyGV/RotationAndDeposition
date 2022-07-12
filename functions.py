from numpy import (
    convolve, ones, cos, sin, power, genfromtxt, arange, array, sqrt, pi,
    linspace, meshgrid, arctan2, rot90, loadtxt, log10, arcsin
    )
from numpy import matlib
#import pickle
import numpy as np
from scipy.interpolate import interp1d, RectBivariateSpline
import time
from scipy.integrate import quad
from scipy.optimize import brentq
from joblib import Memory
from math import ceil
import custom_minimizer 
from scipy.optimize import basinhopping
from PyQt5.QtCore import pyqtSignal, QObject, pyqtSlot, QThread
from PyQt5 import QtWidgets
#from ray.util.multiprocessing import Pool as Pool_ray
#import ray
import re
import os
import shutil
from bsplines import bspline_basis_set
from numba import njit, prange
#import dill
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
    
    log_signal = pyqtSignal(str,str)

    def __init__(self, parent=None):
        super().__init__(parent)
    
    def update(self,
                 fname_sim, fname_exp, rotation_type, C, source, magnetron_x, 
                 magnetron_y, substrate_shape, substrate_radius, 
                 substrate_x_len, substrate_y_len, substrate_res, cores, 
                 verbose, delete_cache, point_tolerance, 
                 holder_inner_radius, holder_outer_radius, deposition_len_x, 
                 deposition_len_y, R_step,
                 k_step, NR_step, R_extra_bounds, R_min, R_max, k_min, k_max, 
                 NR_min, NR_max, omega_s_max, omega_p_max, x0_1, x0_2, x0_3,
                 minimizer, R_mc_interval, k_mc_interval, NR_mc_interval,
                 R_min_step, k_min_step, NR_min_step, mc_iter, T, smooth, 
                 spline_order):
        
        
        self.smooth = smooth
        self.spline_order = spline_order
        self.memory = Memory('cache', verbose=0)
        self.count = 0
        self.rotation_type = rotation_type
        self.fname_sim = fname_sim
        self.fname_exp = fname_exp
        self.C = C
        self.magnetron_x = magnetron_x
        self.magnetron_y = magnetron_y
        self.source = source
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
        #self.n_sup_points = n_sup_points
        #if substrate_shape == 'Circle':
            
        self.substrate_res = substrate_res
        self.substrate_rows = ceil(substrate_y_len*substrate_res)
        self.substrate_columns = ceil(substrate_x_len*substrate_res)
        self.cores = cores # number of jobs for paralleling
        self.verbose = verbose # True: print message each time when function of deposition called
        self.delete_cache = delete_cache
        self.point_tolerance = point_tolerance/100 
        self.holder_inner_radius = holder_inner_radius  
        self.holder_outer_radius = holder_outer_radius
        self.deposition_len_x = deposition_len_x 
        self.deposition_len_y = deposition_len_y 
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
        else: error('WARNING: memory has not cleared, changes in the code or settings may be ignored')
        
        ang=arange(0, 2*pi,0.01) #np.aarange
        self.holder_circle_inner_x=holder_inner_radius*cos(ang) #np.cos
        self.holder_circle_inner_y=holder_inner_radius*sin(ang) #np.sin
        self.holder_circle_outer_x=holder_outer_radius*cos(ang) #np.cos
        self.holder_circle_outer_y=holder_outer_radius*sin(ang) #np.sin 
        
        #### depoition profile meshing
        if substrate_shape == 'Rectangle':
            substrate_coords_x = linspace(-substrate_x_len/2, substrate_x_len/2, 
                                             num=self.substrate_columns)
            
            substrate_coords_y = linspace(-substrate_y_len/2, substrate_y_len/2, 
                                             num=self.substrate_rows)
            
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

        if source == 'SIMTRA':
            self.success = self.open_simtra_file(fname_sim)
            
        elif source == 'Experiment':
            self.success = self.open_exp_file(fname_exp)

        else: raise TypeError(f'incorrect source {source}')
        if not self.success: 
            return None
        if not self.F_axial:
            self.F = RectBivariateSpline(self.deposition_coords_x, 
                                         self.deposition_coords_y,  #scipy.interpolate.RegularGridInterpolator
                                         self.deposition_coords_map_z,
                                         s=self.smooth, kx=self.spline_order,
                                         ky=self.spline_order)
            self.F_coef = self.F.get_coeffs()
            self.F_knots = self.F.get_knots()
            
        print('calculation x bspline representation...')
        self.x_bspline_set = bspline_basis_set(self.spline_order, 
                                               self.F_knots[0])
        
        print('calculation y bspline representation...')
        self.y_bspline_set = bspline_basis_set(self.spline_order, 
                                               self.F_knots[1])
            
        print('calculation matrix representation of spline... ')
        n = len(self.F_knots[0])-(self.spline_order+1)
        m = len(self.F_knots[1])-(self.spline_order+1)
        k = self.spline_order+1
        self.F_matrix = np.zeros((n, m, k, k))
        for i in range(len(self.x_bspline_set)):
            N = self.x_bspline_set[i]
            N_matrix = np.zeros((k, k))
            l = len(N)
            coeffs = np.zeros((l, k))
            ai = np.zeros((l), dtype=np.int32)
            bi = np.zeros((l), dtype=np.int32)
            t = 0
            for nn in N:
                coeffs[t, :(len(nn)-2)] = nn[2:]
                ai[t] = self.dep_xi(nn[0])
                bi[t] = self.dep_xi(nn[1])
                t+=1

            ai0 = ai.min()
            ai = ai-ai0
            bi0 = bi.max()
            bi = bi-ai0
            for q in range(t):
                for ki in range(ai[q], bi[q]):
                    N_matrix[ki, :coeffs.shape[1]] += coeffs[q, :]

            for j in range(len(self.y_bspline_set)):
                M = self.y_bspline_set[j]
                M_matrix = np.zeros((k, k))
                s = len(M)
                coeffs = np.zeros((s, k))
                aj = np.zeros((s), dtype=np.int32)
                bj = np.zeros((s), dtype=np.int32)
                t = 0
                for mm in M:
                    coeffs[t, :(len(mm)-2)] = mm[2:]
                    aj[t] = self.dep_yi(mm[0])
                    bj[t] = self.dep_yi(mm[1])
                    t+=1
                    
                aj0 = aj.min()
                aj = aj-aj0
                bj0 = bj.max()
                bj = bj-aj0
                for q in range(t):
                    for ki in range(aj[q], bj[q]):
                        M_matrix[ki, :coeffs.shape[1]] += coeffs[q, :]

                NM = np.zeros((bi0-ai0, bj0-aj0, k, k))
                for ki in range(NM.shape[0]):
                    for kj in range(NM.shape[1]):
                        NM[ki, kj] = self.F_coef[i*m+j]*np.tensordot(N_matrix[ki], M_matrix[kj], axes=0)
                self.F_matrix[ai0:bi0, aj0:bj0] += NM
        print('done')

        n = len(self.deposition_coords_x)
        m = len(self.deposition_coords_y)
        z = np.zeros((n, m))
        z0 = self.F(self.deposition_coords_x, self.deposition_coords_y)
        
        for i in range(len(self.deposition_coords_x)):
            x = self.deposition_coords_x[i]
            for j in range(len(self.deposition_coords_y)):
                y = self.deposition_coords_y[j]
                z[i, j] = self.F_spline(x, y)
        self.matrix_err = np.max(np.abs(z-z0))
        '''
        from matplotlib import pyplot as plt
        plt.contourf(self.deposition_coords_map_x, 
                     self.deposition_coords_map_y, z, 100)
        plt.hlines(self.deposition_coords_y, self.deposition_coords_x.min(),
                   self.deposition_coords_x.max())
        plt.vlines(self.deposition_coords_x, self.deposition_coords_y.min(),
                   self.deposition_coords_y.max())
        plt.colorbar()
        plt.show()
        #print(self.F_knots[0])
        #print(self.F_knots[1])
        print(len(self.F_knots[0]), len(self.F_knots[1]))
        print(len(self.F_coef), self.F_matrix.shape[0]*self.F_matrix.shape[1])
        print(self.F_matrix.shape)
        print(self.F_coef[-1])
        '''
        print('spline dimension:', len(self.F_coef))
        print('pline matrix shape:', self.F_matrix.shape)
        print('matrix error:', self.matrix_err)
        '''
        #joblib_ignore=['self']
        
        if rotation_type == 'Planet':
            self.xyp = self.xyp_planet
        elif rotation_type == 'Solar':
            self.xyp = self.xyp_solar
            joblib_ignore.append('k')
        '''    
        self.time_f = []
        self.deposition = Deposition(self.rho, self.alpha0, self.F, self.dep_dr, self, njobs=self.cores)
        
        self.success = True
        
    def F_spline(self, x, y):
        z = 0
        if x == self.deposition_coords_x[-1]:
            x = x-0.000001
        p = self.dep_xi(x)
        
        if y == self.deposition_coords_y[-1]:
            y = y-0.000001
        q = self.dep_yi(y)
    
        coeffs = self.F_matrix[p, q]
        for ki in range(coeffs.shape[0]):
            for kj in range(coeffs.shape[1]):
                z += coeffs[ki, kj]*(x**ki)*(y**kj)
        return z
    
    def dep_xi(self, x):
        a = np.sum(self.F_knots[0][self.spline_order:-self.spline_order]<=x)-1
        return a
        
    def dep_yi(self, y):
        a = np.sum(self.F_knots[1][self.spline_order:-self.spline_order]<=y)-1
        return a
        
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
             
        self.dep_dr = np.min((self.deposition_len_x/M, 
                              self.deposition_len_y/N))
        
    def open_simtra_file(self, fname):
        try:
            with open(fname, 'r') as f:
                line = list(f)[0]
                
                #M, N, I_tot, s = re.split(r'\t', line)
                M, N, I_tot = re.findall(r'\d+', line)[:3]
                M, N, I_tot = int(M), int(N), int(I_tot)
        except FileNotFoundError: 
            success = False
            error(f"Файл {fname} не найден")
        except: 
            success = False
            error(f"Неверный формат файла с результатами расчёта SIMTRA:\n{M, N, I_tot}")
        else:
            self.init_deposition_mesh(M=M, N=N)
            with open(fname, 'r') as f:
                RELdeposition_coords_map_z = rot90(loadtxt(f, skiprows=1)) #np.rot90
            if RELdeposition_coords_map_z.shape != (M, N):
                success = False
                m, n = RELdeposition_coords_map_z.shape
                error(f"Неверно указана размерность сетки: {M}x{N} вместо {m}x{n}")
            
            msg = 'Согласно резудьтатам расчёта SIMTRA: {} % потока осаждено на заданную поверхность'.format(round(100*RELdeposition_coords_map_z.sum()/I_tot))
            type = 'Чтение файла с результатами расчёта SIMTRA'
            self.log_signal.emit(msg, type)
            #message(msg)
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
            dr = np.diff(r)
            res = 1/dr.min()
            self.init_deposition_mesh(res_x=res, res_y=res)
            f = interp1d(r, h, fill_value='extrapolate', bounds_error=False)
            Z = f(sqrt(sqr(self.deposition_coords_map_x+self.magnetron_x)+
                       sqr(self.deposition_coords_map_y+self.magnetron_y)))#np.sqrt
            norm = Z.max()
            Z = self.C*Z/Z.max()
            assert Z.max()<=self.C
            interp = interp_axial(self.magnetron_x, self.magnetron_y, norm, f, self.C)
            self.deposition_coords_map_z = Z
            self.F = interp.F
            self.F_axial = True
            success = True
        return success

    def heterogeneity(self, I):
        return (1-I.min()/I.max())*100
    
    def profile_info(self):
        x0 = np.linspace(0, self.holder_outer_radius)
        y0 = np.zeros_like(x0)
        h0 = np.zeros_like(x0)
        N = 360
        ang = np.linspace(0, 2*pi, num=N)
        da = ang[1]-ang[0]
        for a in ang: 
            x = x0*np.cos(a)+y0*np.sin(a)
            y = -x0*np.sin(a)+y0*np.cos(a)
            h0 += self.F(x,y, grid=False)/da
        h = matlib.repmat(h0, N, 1)
        r = np.linspace(0, self.holder_outer_radius)
        a = np.linspace(0, 2*pi, num=N)
        x, y = pol2cart(*np.meshgrid(r,a))
        return x0, y0, h0, x, y, h
    
class Deposition(QThread):
    progress_signal = pyqtSignal(float)
    msg_signal = pyqtSignal(str)
    debug_signal = pyqtSignal(str)
    
    def __init__(self, rho, alpha, F, dep_dr, model, njobs=1, parent=None):
        super().__init__(parent)
        self.time = []
        n = len(rho)
        self.n = n
        self.model = model
        self.rho = rho
        self.ind = range(len(rho))
        self.alpha = alpha
        self.count = 0
        self.njobs = njobs
        if njobs == 1:
            self.workers = [Worker_single(F, rho, alpha, dep_dr)]
            self.workers[0].progress_signal.connect(self.progress)
            self.workers[0].msg_signal.connect(self.msg)
            self.workers[0].debug_signal.connect(self.debug)
        else:
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
                
            self.workers = [Worker(F, rho_p[i], alpha_p[i], dep_dr) for i in range(njobs)]
        
    @pyqtSlot()
    def progress(self):
        self.count += 1
        self.progress_signal.emit(self.count/self.n) 
        
    @pyqtSlot(str)    
    def msg(self, s):
        self.msg_signal.emit(s) 
        
    @pyqtSlot(str)    
    def debug(self, s):
        self.debug_signal.emit(s) 
            
    def task(self, R, k, NR, omega, alpha0_sub, point_tolerance, cores):
         self.R = R
         self.k = k
         self.NR = NR
         self.omega = omega
         self.alpha0_sub = alpha0_sub
         self.point_tolerance = point_tolerance
         self.cores = cores
         self.count = 0
         self.I_lenx = 2
         self.I_leny = 2
         
    def xyp(self, a, i):
        x = self.R*cos(a)+self.rho[i]*cos(a*self.k + self.alpha[i])
        y = self.R*sin(a)+self.rho[i]*sin(a*self.k + self.alpha[i])
        return x, y
    
    def dxyp(self, a, i):
        dx = -self.R*sin(a)-self.rho[i]*self.k*sin(a*self.k + self.alpha[i])
        dy = self.R*cos(a)+self.rho[i]*self.k*cos(a*self.k + self.alpha[i])
        return dx, dy
    
    def xy_sym(self, i):
        '''
        x(a) and y(a) in format:
        [[a1, b1, k1, phi1], [a2, b2, k2, phi2], ...]
        where 
        x(a) = sum_i ai*cos(ki*a + phi_i)
        y(a) = sum_i bi*sin(ki*a + phi_i)
        '''
        x = np.array([[self.R, 1, 0], [self.rho[i], self.k, self.alpha[i]]]) 
        y = np.array([[self.R, 1, 0], [self.rho[i], self.k, self.alpha[i]]]) 
        return x, y

    def run(self):
        t0 = time.time()
        """
        R = self.R
        k = self.k
        NR = self.NR
        omega = self.omega
        alpha0_sub = self.alpha0_sub
        point_tolerance = self.point_tolerance
        if self.njobs == 1:
            self.workers[0].set_properties(R, k, NR, omega, alpha0_sub,
                                           point_tolerance)
            
            self.hs = self.workers[0]()
        else:
            hs = []
            #result = []
            '''
            for worker in self.workers:
                worker.set_properties(R, k, NR, omega, alpha0_sub,
                                      point_tolerance)
    
            with Pool(processes=len(self.workers)) as pool:      
                for worker in self.workers:
                    a = pool.apply_async(worker)
                    result.append(a)
                hs = [result[i].get() for i in range(len(self.workers)) ]
            self.hs = np.concatenate(hs)
            '''
        """
        
        """
        hs = []
        _s = slice(self.model.spline_order, 
                   -self.model.spline_order)
        xs = self.model.F_knots[0][_s] 
        ys = self.model.F_knots[1][_s]
        dx = np.min(np.diff(xs))
        dy = np.min(np.diff(ys))
        n_points_per_interval = 1 #how many points should be in each alpha interval
        tolerance = 1e-3 # x(breaks[i])-xs[i] < tolerance * dx (or y)
        xtol = tolerance * dx
        ytol = tolerance * dy
        a = self.alpha0_sub
        a1 = self.alpha0_sub + self.NR*2*pi
        for i in self.ind:
            print('finding edges...')
            x, y = self.xyp(a, i)
            breaks_x = []
            breaks_y = []
            ni0 = []
            nj0 = []
            
            '''x'''
            max_dxy_da = self.R+self.rho[i]*self.k # dx/da, dy/da <= R+rho*k
            da = (dx/max_dxy_da) / n_points_per_interval
            num = ceil((a1-a)/da)+1
            ang = np.linspace(a, a1, num=num, endpoint=True)
            p = self.model.dep_xi(x)
            ni0.append(p)
            for j, aa in enumerate(ang):
                x, _ = self.xyp(aa, i)
                p1 = self.model.dep_xi(x)
                if abs(p1-p)==1:
                    def f(a):
                        x, _ = self.xyp(a, i)
                        if p1>p:
                            c = 1
                        else:
                            c = 0
                        x0 = xs[p+c]
                        return x-x0
                    breaks_x.append(brentq(f, ang[j-1], aa, xtol=xtol))
                    ni0.append(p1)
                    p = p1
                elif abs(p1-p)>1:
                    raise ValueError('finding edges: angle step for x is too big!')
                    
            '''y'''
            da = (dy/max_dxy_da) / n_points_per_interval
            num = ceil((a1-a)/da)+1
            ang = np.linspace(a, a1, num=num, endpoint=True)
            q = self.model.dep_yi(y)
            nj0.append(q)
            for j, aa in enumerate(ang):
                _, y = self.xyp(aa, i)
                q1 = self.model.dep_yi(y)
                if abs(q1-q)==1:
                    def f(a):
                        _, y = self.xyp(a, i)
                        if q1>q:
                            c = 1
                        else:
                            c = 0
                        y0 = ys[q+c]
                        return y-y0
                    breaks_y.append(brentq(f, ang[j-1], aa, xtol=ytol))
                    nj0.append(q1)
                    q = q1
                elif abs(q1-q)>1:
                    raise ValueError('finding edges: angle step for y is too big!')
                
            '''combining'''
            breaks = np.concatenate(([a], breaks_x, breaks_y))
            ni = np.empty((breaks.shape[0]), dtype=np.int32)
            x, y = self.xyp(a, i)
            ni[0] = self.model.dep_xi(x)
            nj = np.empty((breaks.shape[0]), dtype=np.int32)
            nj[0] = self.model.dep_yi(y)
            ind = np.argsort(breaks)
            for ki in range(1, breaks.shape[0]):
                if ind[ki]<1+len(breaks_x):
                    ni[ki] = ni0[ind[ki]]#заменить на ind[ki]
                    nj[ki] = nj[ki-1]
                else:
                    ni[ki] = ni[ki-1]
                    nj[ki] = nj0[ind[ki]-len(breaks_x)]
            #print(ni, nj, ni0, nj0)
            breaks = np.concatenate((breaks[ind], [a1]))
            '''
            if i ==0:
                import matplotlib.pyplot as plt
                x, y = self.xyp(breaks, i)
                _ys = ys[ys>y.min()]
                _ys = _ys[_ys<y.max()]
                _xs = xs[xs>x.min()]
                _xs = _xs[_xs<x.max()]
                plt.hlines(_ys, xmin=x.min(), xmax=x.max())
                plt.vlines(_xs, ymin=y.min(), ymax=y.max())
                plt.plot(x, y, 'x', color='red')
                for j in range(len(ni)):
                    plt.text(xs[ni[j]], ys[nj[j]], f'{ni[j]}, {nj[j]}')
                plt.savefig('path.png')
                plt.show()
            '''
                    
            print('done')
            
            print('integration...')
            x_sym, y_sym = self.xy_sym(i)
            I = np.zeros((len(ni)))
            for j in range(I.shape[0]):
                I[j] = self.integrate(self.model.F_matrix[ni[j], nj[j]], x_sym, y_sym, breaks[j], breaks[j+1])
            print('done')
            hs.append(np.sum(np.sort(I)))
        temp = self.hs
        """
        _s = slice(self.model.spline_order, 
                   -self.model.spline_order)
        xs = self.model.F_knots[0][_s] 
        ys = self.model.F_knots[1][_s]
        #temp = self.hs
        hs = self.do(self.R, self.k, self.NR, self.alpha0_sub, xs, ys,
                     len(self.ind), self.rho, self.alpha, self.model.F_matrix)
        self.hs = np.array(hs)/(2*pi*self.omega)
        #print('relative error:', np.max(np.abs((self.hs-temp)/temp)))
        t = time.time()-t0
        self.time.append(t)
        return
    
    @staticmethod
    @njit(parallel=False)
    def do(R, k, NR, alpha0_sub, xs, ys, max_ind, rho, alpha, F_matrix):
        hs = np.zeros((max_ind))
        dx = np.min(np.diff(xs))
        dy = np.min(np.diff(ys))
        n_points_per_interval = 3 #how many points should be in each alpha interval
        tolerance = 1e-3 # x(breaks[i])-xs[i] < tolerance * dx (or y)
        xtol = tolerance * dx
        ytol = tolerance * dy
        a = alpha0_sub
        a1 = alpha0_sub + NR*2*pi
        for i in prange(max_ind):
            #print('finding edges...')
            x = R*cos(a) + rho[i]*cos(a*k + alpha[i])
            y = R*sin(a) + rho[i]*sin(a*k + alpha[i])

            
            '''x'''
            max_dxy_da = R+rho[i]*k # dx/da, dy/da <= R+rho*k
            da = (dx/max_dxy_da) / n_points_per_interval
            num = ceil((a1-a)/da)+1
            ang = np.linspace(a, a1, num)
            p =  np.sum(xs<=x)-1
            breaks_x = np.zeros((len(ang)))
            ni0 = np.zeros((len(ang)+1))
            ni0[0] = p
            len_breaks_x = 0
            
            for j, aa in enumerate(ang):
                x = R*cos(aa) + rho[i]*cos(aa*k + alpha[i])
                p1 = np.sum(xs<=x)-1
                if abs(p1-p)==1:
                    if p1>p:
                        c = 1
                        a_pre = aa
                        a_cur = ang[j-1]
                        x_pre = x
                        x_cur = R*cos(a_cur) + rho[i]*cos(a_cur*k + alpha[i])
                    else:
                        c = 0
                        a_pre = ang[j-1]
                        a_cur = aa
                        x_pre = R*cos(a_pre) + rho[i]*cos(a_pre*k + alpha[i])
                        x_cur = x
                        
                    x0 = xs[p+c]
            
                    a_mid = (a_pre*(x0-x_cur) + a_cur*(x_pre-x0))/(x_pre - x_cur)

                    x = R*cos(a_mid) + rho[i]*cos(a_mid*k + alpha[i])
                    while abs(x-x0)>xtol:
                        #print('dx', x-x0)
                        if x-x0 > 0:
                            a_pre = a_mid
                            x_pre = x
                        else:
                            a_cur = a_mid
                            x_cur = x
                        
                        a_mid = (a_pre*(x0-x_cur) + a_cur*(x_pre-x0))/(x_pre - x_cur)
                        x = R*cos(a_mid) + rho[i]*cos(a_mid*k + alpha[i])
                        
                    breaks_x[len_breaks_x] = a_mid
                    ni0[len_breaks_x+1] = p1
                    len_breaks_x += 1
                    p = p1
                elif abs(p1-p)>1:
                    print('finding edges: angle step for x is too big!')
                    
            breaks_x = breaks_x[:len_breaks_x]
            ni0 = ni0[:len_breaks_x+1]
                    
            '''y'''
            da = (dy/max_dxy_da) / n_points_per_interval
            num = ceil((a1-a)/da)+1
            ang = np.linspace(a, a1, num)
            q = np.sum(ys<=y)-1
            breaks_y = np.zeros((len(ang)))
            nj0 = np.zeros((len(ang)+1))
            len_breaks_y = 0
            nj0[0] = q
            
            for j, aa in enumerate(ang):
                y = R*sin(aa) + rho[i]*sin(aa*k + alpha[i])
                q1 = np.sum(ys<=y)-1
                if abs(q1-q)==1:
                    if q1>q:
                        c = 1
                        a_pre = aa
                        a_cur = ang[j-1]
                        y_pre = y
                        y_cur = R*sin(a_cur) + rho[i]*sin(a_cur*k + alpha[i])
                    else:
                        c = 0
                        a_pre = ang[j-1]
                        a_cur = aa
                        y_pre = R*sin(a_pre) + rho[i]*sin(a_pre*k + alpha[i])
                        y_cur = y
                        
                        
                    y0 = ys[q+c]
                    
                    a_mid = (a_pre*(y0-y_cur) + a_cur*(y_pre-y0))/(y_pre - y_cur)

                    y = R*sin(a_mid) + rho[i]*sin(a_mid*k + alpha[i])
                    while abs(y-y0)>ytol:
                        #print('dy', y-y0)
                        if y-y0 > 0: 
                            a_pre = a_mid
                            y_pre = y
                        else:
                            a_cur = a_mid
                            y_cur = y
                        
                        a_mid = (a_pre*(y0-y_cur) + a_cur*(y_pre-y0))/(y_pre - y_cur)
                        y = R*sin(a_mid) + rho[i]*sin(a_mid*k + alpha[i])
                        
                    breaks_y[len_breaks_y] = a_mid
                    nj0[len_breaks_y+1] = q1
                    len_breaks_y += 1
                    q = q1
                elif abs(q1-q)>1:
                    print('finding edges: angle step for y is too big!')
                
                                    
            breaks_y = breaks_y[:len_breaks_y]
            nj0 = nj0[:len_breaks_y+1]
                
            '''combining'''
            breaks = np.concatenate((np.array([a]), breaks_x, breaks_y))
            ni = np.empty((breaks.shape[0]), dtype=np.int64)
            x = R*cos(a) + rho[i]*cos(a*k + alpha[i])
            y = R*sin(a) + rho[i]*sin(a*k + alpha[i])
            ni[0] = np.sum(xs<=x)-1
            nj = np.empty((breaks.shape[0]), dtype=np.int64)
            nj[0] = np.sum(ys<=y)-1
            ind = np.argsort(breaks)
            for ki in range(1, breaks.shape[0]):
                if ind[ki]<1+len(breaks_x):
                    ni[ki] = ni0[ind[ki]]#заменить на ind[ki]
                    nj[ki] = nj[ki-1]
                else:
                    ni[ki] = ni[ki-1]
                    nj[ki] = nj0[ind[ki]-len(breaks_x)]
            #print(ni, nj, ni0, nj0)
            breaks = np.concatenate((breaks[ind], np.array([a1])))
            #print('done')
            
            #print('integration...')
            '''integration'''
            x_sym = np.array([[R, 1, 0], [rho[i], k, alpha[i]]]) 
            y_sym = np.array([[R, 1, 0], [rho[i], k, alpha[i]]]) 
            I = np.zeros((len(ni)))
            for j in range(I.shape[0]):
                I[j] = integrate(F_matrix[ni[j], nj[j]], x_sym, y_sym, breaks[j], breaks[j+1])
            #print('done')
            _mask = np.zeros((max_ind))
            _mask[i] = 1
            hs += _mask*(np.sum(np.sort(I)))
        return hs
@njit
def Iij(i, j, x_matrix, y_matrix, ang0, ang1):
    if j == 0:
        if i == 0:
            return ang1-ang0
        elif i == 1:
            [b0, k0, phi0], [b1, k1, phi1] = y_matrix
            res1 = b0/k0 * np.sin(k0*ang1 + phi0) + b1/k1 * np.sin(k1*ang1 + phi1)
            res0 = b0/k0 * np.sin(k0*ang0 + phi0) + b1/k1 * np.sin(k1*ang0 + phi1)
            return res1 - res0
        else:
            print('invalid index j = ' + str(j) + ' in Iij')
    elif j == 1:
        if i == 0:
            [a0, k0, phi0], [a1, k1, phi1] = x_matrix
            res1 = -a0/k0 * np.cos(k0*ang1 + phi0) - a1/k1 * np.cos(k1*ang1 + phi1)
            res0 = -a0/k0 * np.cos(k0*ang0 + phi0) - a1/k1 * np.cos(k1*ang0 + phi1)
            return res1 - res0
        elif i == 1:
            [a0, k0, phi0], [a1, k1, phi1] = x_matrix
            [b0, k0_b, phi0_b], [b1, k1_b, phi1_b] = y_matrix
            if k0 != k0_b:
                print('case of different k0 has not implemented yet')
            if k1 != k1_b:
                print('case of different k1 has not implemented yet')
            if phi0 != phi0_b:
                print('case of different phi0 has not implemented yet')
            if phi1 != phi1_b:
                print('case of different phi1 has not implemented yet')    
            if k0 == 0 or k1 == 0:
                print('case of zero k0 or k1 has not implemented yet')
            else:
                res1_1 = -a0*b0/(4*k0) * np.cos(2*k0*ang1 + 2*phi0)
                res1_2 = -a1*b1/(4*k1) * np.cos(2*k1*ang1 + 2*phi1)
                res0_1 = -a0*b0/(4*k0) * np.cos(2*k0*ang0 + 2*phi0)
                res0_2 = -a1*b1/(4*k1) * np.cos(2*k1*ang0 + 2*phi1)
                if k1==k0:
                    res1_3 = -(a1*b0-a0*b1)/2 * np.sin(phi0-phi1)*ang1
                    res0_3 = -(a1*b0-a0*b1)/2 * np.sin(phi0-phi1)*ang0
                else:
                    res1_3 = (a1*b0-a0*b1)/(2*(k0-k1)) * np.cos((k0-k1)*ang1 + phi0-phi1)
                    res0_3 = (a1*b0-a0*b1)/(2*(k0-k1)) * np.cos((k0-k1)*ang0 + phi0-phi1)
                if k1==-k0:
                    print('case of k0 == -k1 has not implemented yet')
                else:
                    res1_4 = -(a1*b0+a0*b1)/(2*(k0+k1)) * np.cos((k0+k1)*ang1 + phi0+phi1)
                    res0_4 = -(a1*b0+a0*b1)/(2*(k0+k1)) * np.cos((k0+k1)*ang0 + phi0+phi1)
                res1 = res1_1 + res1_2 + res1_3 + res1_4
                res0 = res0_1 + res0_2 + res0_3 + res0_4
                return res1  - res0
    
@njit
def integrate(F, x, y, a0, a1): #F = F_matrix[p, q]
    res = 0
    for i in range(F.shape[0]):
        for j in range(F.shape[1]):
            if F[i, j] != 0:
                res += F[i,j]*Iij(i, j, x, y, a0, a1)
    return res  
    
class Worker:
    def __init__(self, F, rho, alpha, dep_dr):
        self.F = F
        self.rho = rho
        self.alpha = alpha
        self.ind = range(len(rho))
        self.hs = None
        self.dep_dr = dep_dr
        
    def set_properties(self, R, k, NR, omega, alpha0_sub,
                       point_tolerance):
        self.R = R
        self.k = k
        self.NR = NR
        self.omega = omega
        self.alpha0_sub = alpha0_sub
        self.point_tolerance = point_tolerance
        
    def xyp(self, a, i):
        x = self.R*cos(a)+self.rho[i]*cos(a*self.k + self.alpha[i])
        y = self.R*sin(a)+self.rho[i]*sin(a*self.k + self.alpha[i])
        return x, y
        
    def __call__(self):
        ns = np.round(2*pi*self.NR*(self.R+self.rho*self.k)/self.dep_dr)
        hs = [quad(lambda a: self.F(self.xyp(a, i)), 
                   self.alpha0_sub, self.NR*2*pi,
                   limit=max(1, int(ns[i])), 
                   epsrel=self.point_tolerance)[0] for i in self.ind]
        hs = np.array(hs)/(2*pi*self.omega)
        return hs
    
class Worker_single(QObject):
    progress_signal = pyqtSignal()
    msg_signal = pyqtSignal(str)
    debug_signal = pyqtSignal(str)
    
    def __init__(self, F, rho, alpha, dep_dr, parent=None):
        super().__init__(parent)
        self.F = F
        self.rho = rho
        self.alpha = alpha
        self.ind = range(len(rho))
        self.hs = None
        self.dep_dr = dep_dr
        
    def set_properties(self, R, k, NR, omega, alpha0_sub,
                       point_tolerance):
        self.R = R
        self.k = k
        self.NR = NR
        self.omega = omega
        self.alpha0_sub = alpha0_sub
        self.point_tolerance = point_tolerance
        
        
    def xyp(self, a, i):
        x = self.R*cos(a)+self.rho[i]*cos(a*self.k + self.alpha[i])
        y = self.R*sin(a)+self.rho[i]*sin(a*self.k + self.alpha[i])
        return x, y
    
    def __call__(self):
        hs = []
        for i in self.ind:
            n = int(round(2*pi*self.NR*(self.R+self.rho[i]*self.k)/self.dep_dr/21))
            
            res = quad(lambda a: self.F(*self.xyp(a, i)), 
                                       self.alpha0_sub, self.NR*2*pi,
                                       limit=max(n, 1), 
                                       epsrel=self.point_tolerance,
                                       full_output=1)
            
            if len(res)==4:
                self.msg_signal.emit(str(res[3]))
            k = res[2]['last']    
            if k>=n:
                self.msg_signal.emit('Погрешность может быть недооценена из-за слишком грубой дискретизации профиля напыления')
            self.debug_signal.emit(f'Point {i}: {k} subintervals from {n}')
            hs.append(res[0])
            self.progress_signal.emit()
        hs = np.array(hs)/(2*pi*self.omega)
        return hs

class Optimizer(QObject):
    upd_signal = pyqtSignal(str)
    def __init__(self, deposition, parent=None):
        super().__init__(parent)
        self.deposition = deposition
        
    def optimisation(self, heterogeneity,
                     alpha0_sub, point_tolerance, cores, 
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
        args = [*x, 1, self.alpha0_sub, self.point_tolerance, self.cores]
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
