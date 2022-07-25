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
#from functools import lru_cache
#from numpy_lru_cache_decorator import np_cache
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
                 substrate_x_len, substrate_y_len, substrate_res, tolerance, 
                 holder_inner_radius, holder_outer_radius, deposition_len_x, 
                 deposition_len_y, R_step,
                 k_step, NR_step, R_extra_bounds, R_min, R_max, k_min, k_max, 
                 NR_min, NR_max, omega_s_max, omega_p_max, x0_1, x0_2, x0_3,
                 minimizer, R_mc_interval, k_mc_interval, NR_mc_interval,
                 R_min_step, k_min_step, NR_min_step, mc_iter, T, smooth, 
                 spline_order, debug=True):
        
        self.smooth = smooth
        self.spline_order = spline_order
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
        self.tolerance = tolerance/100 
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
        #if not self.F_axial:
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
        
        t0 = time.time()
        ai0 = np.zeros((len(self.x_bspline_set))).astype(np.int64)
        bi0 = np.zeros((len(self.x_bspline_set))).astype(np.int64)
        N_matrix = np.zeros((len(self.x_bspline_set), k, k))
        for i in range(len(self.x_bspline_set)):
            N = self.x_bspline_set[i]
            bounds = self.F_knots[0][self.spline_order:-self.spline_order]
            ai0[i], bi0[i], N_matrix[i] = self.convert_bspline_to_poly(N, bounds)
            
        aj0 = np.zeros((len(self.y_bspline_set))).astype(np.int64)
        bj0 = np.zeros((len(self.y_bspline_set))).astype(np.int64)
        M_matrix = np.zeros((len(self.y_bspline_set), k, k))    
        for j in range(len(self.y_bspline_set)):
            M = self.y_bspline_set[j]
            bounds = self.F_knots[1][self.spline_order:-self.spline_order]
            aj0[j], bj0[j], M_matrix[j] = self.convert_bspline_to_poly(M, bounds)
        
        self.F_matrix = self.calc_F_matrix(n, m, k, N_matrix, M_matrix, 
                                           ai0, bi0, aj0, bj0, self.F_coef)

        t = time.time()-t0
        print(f'done: {t} s')

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
        print('spline dimension:', len(self.F_coef))
        print('spline matrix shape:', self.F_matrix.shape)
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
        self.deposition = Deposition(self.rho, self.alpha0, self.F,
                                     self.dep_dr, self, debug)
        
        self.success = True
        
    @staticmethod
    @njit(cache=True)
    def calc_F_matrix(n, m, k, N, M, ai0, bi0, aj0, bj0, F_coeff):
        F_matrix = np.zeros((n, m, k, k))
        for i in range(N.shape[0]):
            for j in range(M.shape[0]):
                NM = np.zeros((bi0[i]-ai0[i], bj0[j]-aj0[j], k, k))
                for ki in range(NM.shape[0]):
                    for kj in range(NM.shape[1]):
                        for ni in range(NM.shape[2]):
                            for nj in range(NM.shape[3]):
                                val = F_coeff[i*m+j]*N[i, ki, ni]*M[j, kj, nj]
                                NM[ki, kj, ni, nj] = val
                F_matrix[ai0[i]:bi0[i], aj0[j]:bj0[j]] += NM
        return F_matrix
    
    @staticmethod
    @njit(cache=True)
    def convert_bspline_to_poly(N, bounds):
        k = N.shape[1]-2
        N_matrix = np.zeros((k, k))
        l = N.shape[0]
        coeffs = np.zeros((l, k))
        ai = np.zeros((l)).astype(np.int64)
        bi = np.zeros((l)).astype(np.int64)
        t = 0
        for nn in N:
            coeffs[t, :(len(nn)-2)] = nn[2:]
            ai[t] = np.sum(bounds<=nn[0])-1
            bi[t] = np.sum(bounds<=nn[1])-1
            t+=1

        ai0 = ai.min()
        ai = ai-ai0
        bi0 = bi.max()
        bi = bi-ai0
        for q in range(t):
            for ki in range(ai[q], bi[q]):
                N_matrix[ki, :coeffs.shape[1]] += coeffs[q, :]
                
        return ai0, bi0, N_matrix      
            
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
    
    def __init__(self, rho, alpha, F, dep_dr, model, debug, parent=None):
        super().__init__(parent)
        self.time = []
        n = len(rho)
        self.n = n
        self.model = model
        self.debug_flag = debug
        self.rho = rho
        self.ind = range(len(rho))
        self.alpha = alpha
        self.count = 0
        self.workers = [Worker_single(F, rho, alpha, dep_dr)]
        self.workers[0].progress_signal.connect(self.progress)
        self.workers[0].msg_signal.connect(self.msg)
        self.workers[0].debug_signal.connect(self.debug)
        
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
            
    def task(self, R, k, NR, omega, alpha0_sub, tolerance):
         self.R = R
         self.k = k
         self.NR = NR
         self.omega = omega
         self.alpha0_sub = alpha0_sub
         self.tolerance = tolerance
         self.count = 0
         
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
        [[a1, k1, phi1], [a2, k2, phi2], ...]
        where 
        x(a) = sum_i ai*cos(ki*a + phi_i)
        y(a) = sum_i ai*sin(ki*a + phi_i)
        z = x + jy = sum_i ai*exp(j*(ki*a+phi_i))
        '''
        z = np.array([[self.R, 1, 0], [self.rho[i], self.k, self.alpha[i]]])
        return z

    def run(self):
        t0 = time.time()
        #"""
        if self.debug_flag:
            R = self.R
            k = self.k
            NR = self.NR
            omega = self.omega
            alpha0_sub = self.alpha0_sub
            point_tolerance = 0.01/100
            self.workers[0].set_properties(R, k, NR, omega, alpha0_sub,
                                           point_tolerance)
                
            self.hs = self.workers[0]()
            temp = self.hs
        #"""

        _s = slice(self.model.spline_order, 
                   -self.model.spline_order)
        xs = self.model.F_knots[0][_s] 
        ys = self.model.F_knots[1][_s]
        '''
        hs = self.do(self.R, self.k, self.NR, self.alpha0_sub, xs, ys,
                     len(self.ind), self.rho, self.alpha, self.model.F_matrix,
                     self.tolerance)
        '''
        n = np.array([3, 0, 1])
        H = 50
        x0 = -100
        y0 = 0
        print('calc curvature...')
        hs = self.do_c(self.R, self.k, self.NR, self.alpha0_sub, xs, ys,
                     len(self.ind), self.rho, self.alpha, self.model.F_matrix,
                     self.tolerance, n, x0, y0, H)
        #print('curvature error: ', np.max(np.abs(hs_c-hs))/np.mean(hs))
        self.hs = np.array(hs)/(2*pi*self.omega)
        if self.debug_flag:
            s = f'relative error: {np.max(np.abs((self.hs-temp)/temp))}'
            self.msg(s)
            print(s)
        t = time.time()-t0
        self.time.append(t)
        return
    
    @staticmethod
    @njit(parallel=False, cache=True)
    def do_c(R, k, NR, alpha0_sub, xs, ys, max_ind, rho, alpha, F_matrix, 
           tolerance, n, x0, y0, H):
        hs = np.zeros((max_ind))
        dx = np.min(np.diff(xs))
        dy = np.min(np.diff(ys))
        n_points_per_interval = 3 #how many points should be in each alpha interval
        #tolerance = 1e-3 # x(breaks[i])-xs[i] < tolerance * dx (or y)
        xtol = tolerance * dx
        ytol = tolerance * dy
        ktol = 0.001
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
            '''searching of shade intervals '''
            z_sym = np.array([[R, 1, 0], [rho[i], k, alpha[i]]]) 
            nx, ny, nz = n
            [az0, k0, phi0], [az1, k1, phi1] = z_sym
            max_dK_da = az0*abs(k0 - k1) + abs(k1*ny*x0) + abs(k1*nx*y0) + abs(az0*(k0 - k1)*nx) + abs(k1*(nx*x0 + ny*y0))
            da = (0.01/max_dK_da)
            num = ceil((a1-a)/da)+1
            ang = np.linspace(a, a1, num)
            K_flag = True
            breaks_K_a = np.empty(ang.shape)
            breaks_K_b = np.empty(ang.shape)
            len_breaks_K_a = 0
            len_breaks_K_b = 0
            for j, aa in enumerate(ang):
                K_val = K(z_sym, n, x0, y0, H, aa)
                
                if K_val >= 0:
                    if K_flag:
                        #print('-+ a, k:', aa, k)
                        K_flag = False
                        if j == 0:
                            a_mid = aa
                        else:
                            '''bisection'''
                            a_pre = ang[j-1]
                            a_cur = aa
                            
                            K_pre = K(z_sym, n, x0, y0, H, a_pre)
                            K_cur = K_val
                            a_mid = (a_pre*K_cur - a_cur*K_pre)/(-K_pre + K_cur)
    
                            K_val = K(z_sym, n, x0, y0, H, a_mid)
                            while abs(K_val)>ktol:
                                #print(k)
                                if K_val > 0: 
                                    a_cur = a_mid
                                    K_cur = K_val
                                else:
                                    a_pre = a_mid
                                    K_pre = K_val
                                
                                a_mid = (a_pre*K_cur - a_cur*K_pre)/(-K_pre + K_cur)
                                K_val = K(z_sym, n, x0, y0, H, a_mid)
                            '''end'''
                        breaks_K_a[len_breaks_K_a] = a_mid
                        len_breaks_K_a += 1
                elif not K_flag:
                    #print('+- a, k:', aa, k)
                    K_flag = True
                    '''bisection'''
                    a_pre = ang[j-1]
                    a_cur = aa
                    
                    K_pre = K(z_sym, n, x0, y0, H, a_pre)
                    K_cur = K_val
                    a_mid = (-a_pre*K_cur + a_cur*K_pre)/(K_pre - K_cur)
                    K_val = K(z_sym, n, x0, y0, H, a_mid)
                    while abs(K_val)>ktol:
                        #print()
                        if K_val > 0: 
                            a_pre = a_mid
                            K_pre = K_val
                        else:
                            a_cur = a_mid
                            K_cur = K_val
                        
                        a_mid = (-a_pre*K_cur + a_cur*K_pre)/(K_pre - K_cur)
                        K_val = K(z_sym, n, x0, y0, H, a_mid)
                    '''end'''
                    breaks_K_b[len_breaks_K_b] = a_mid
                    len_breaks_K_b += 1
                
            if len_breaks_K_b == len_breaks_K_a - 1:
                breaks_K_b[len_breaks_K_a-1] = a1
                
            breaks_K_a = breaks_K_a[:len_breaks_K_a]
            breaks_K_b = breaks_K_b[:len_breaks_K_a]
            print('intervals without shading:', breaks_K_a, breaks_K_b)
            '''integration'''
            I = np.zeros((len(ni)))
            for j in range(I.shape[0]):
                interval_flag = False
                for l in range(len_breaks_K_a):
                    if (breaks[j] >= breaks_K_a[l]) and \
                    (breaks[j+1] <= breaks_K_b[l]):
                        aj = breaks[j]
                        bj = breaks[j+1]
                        interval_flag = True
                        break
                    elif (breaks[j] <= breaks_K_a[l]) and \
                    (breaks[j+1] >= breaks_K_a[l]):
                        aj = breaks_K_a[l]
                        if (breaks[j+1] <= breaks_K_b[l]):
                            bj = breaks[j+1]
                            interval_flag = True
                            break
                        else:
                            bj = breaks_K_b[l]
                            interval_flag = True
                            break
                    elif (breaks[j] <= breaks_K_b[l]) and \
                    (breaks[j+1] >= breaks_K_b[l]):
                        bj = breaks_K_b[l]
                        if (breaks[j] <= breaks_K_a[l]):
                            aj = breaks_K_a[l]
                            interval_flag = True
                            break
                        else:
                            aj = breaks[j]
                            interval_flag = True
                            break
                if interval_flag:
                    I[j] = integrate_c(F_matrix[ni[j], nj[j]], z_sym, aj, bj,
                                       n, x0, y0, H)
                else:
                    I[j] = 0
            #print('done')
            _mask = np.zeros((max_ind))
            _mask[i] = 1
            hs += _mask*(np.sum(np.sort(I)))
        return hs/(H*np.sqrt(nx*nx+ny*ny+nz*nz))
    
    @staticmethod
    @njit(parallel=True, cache=True)
    def do(R, k, NR, alpha0_sub, xs, ys, max_ind, rho, alpha, F_matrix, 
           tolerance):
        hs = np.zeros((max_ind))
        dx = np.min(np.diff(xs))
        dy = np.min(np.diff(ys))
        n_points_per_interval = 3 #how many points should be in each alpha interval
        #tolerance = 1e-3 # x(breaks[i])-xs[i] < tolerance * dx (or y)
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
            z_sym = np.array([[R, 1, 0], [rho[i], k, alpha[i]]])  
            I = np.zeros((len(ni)))
            for j in range(I.shape[0]):
                I[j] = integrate(F_matrix[ni[j], nj[j]], z_sym, breaks[j], breaks[j+1])
            #print('done')
            _mask = np.zeros((max_ind))
            _mask[i] = 1
            hs += _mask*(np.sum(np.sort(I)))
        return hs
  
@njit(cache=True)  
def K(z_matrix, n, x0, y0, H, a):
    nx, ny, nz = n
    [a0, k0, phi0], [a1, k1, phi1] = z_matrix
    k_arg = a1*nx + H*nz + a0*nx*np.cos(a*(k0 - k1) + phi0 - phi1) + (nx*x0 + ny*y0)*np.cos(a*k1 + phi1) + a0*ny*np.sin(a*k0 - a*k1 + phi0 - phi1) - ny*x0*np.sin(a*k1 + phi1) + nx*y0*np.sin(a*k1 + phi1)
    k = k_arg/(H*np.sqrt(nx*nx+ny*ny+nz*nz))
    return k

@njit(cache=True)
def integrate_c(F, z, a0, a1, n, x0, y0, H): #F = F_matrix[p, q]
    res = 0
    for i in range(F.shape[0]):
        for j in range(F.shape[1]):
            if F[i, j] != 0:
                res += F[i,j]*Iij_c(i, j, z, a0, a1, n, x0, y0, H)
    return res  

@njit(cache=True)
def Iij_c(i, j, z_matrix, ang0, ang1, n, x0, y0, H):
    [a0, k0, phi0], [a1, k1, phi1] = z_matrix
    nx, ny, nz = n
    if k0 == 0 or k1 == 0:
        print('case of zero k0 or k1 has not implemented yet')
        return 0.0
    elif i == 0:
        if j == 0:
            res0 = ((a1*ang0*k0*k1*nx - a1*ang0*k1**2*nx + ang0*H*k0*k1*nz - ang0*H*k1**2*nz - a0*k1*ny*np.cos(ang0*(k0 - k1) + phi0 - phi1) + (k0 - k1)*(ny*x0 - nx*y0)*np.cos(ang0*k1 + phi1) + a0*k1*nx*np.sin(ang0*k0 - ang0*k1 + phi0 - phi1) + k0*nx*x0*np.sin(ang0*k1 + phi1) - k1*nx*x0*np.sin(ang0*k1 + phi1) + k0*ny*y0*np.sin(ang0*k1 + phi1) - k1*ny*y0*np.sin(ang0*k1 + phi1))/((k0 - k1)*k1))
            res1 = ((a1*ang1*k0*k1*nx - a1*ang1*k1**2*nx + ang1*H*k0*k1*nz - ang1*H*k1**2*nz - a0*k1*ny*np.cos(ang1*(k0 - k1) + phi0 - phi1) + (k0 - k1)*(ny*x0 - nx*y0)*np.cos(ang1*k1 + phi1) + a0*k1*nx*np.sin(ang1*k0 - ang1*k1 + phi0 - phi1) + k0*nx*x0*np.sin(ang1*k1 + phi1) - k1*nx*x0*np.sin(ang1*k1 + phi1) + k0*ny*y0*np.sin(ang1*k1 + phi1) - k1*ny*y0*np.sin(ang1*k1 + phi1))/((k0 - k1)*k1))
            return res1 - res0
        elif j == 1:
            res0 = ((-4*a1*ang0*k0**5*k1*ny*x0 + 10*a1*ang0*k0**4*k1**2*ny*x0 - 10*a1*ang0*k0**2*k1**4*ny*x0 + 4*a1*ang0*k0*k1**5*ny*x0 + 4*a1*ang0*k0**5*k1*nx*y0 - 10*a1*ang0*k0**4*k1**2*nx*y0 + 10*a1*ang0*k0**2*k1**4*nx*y0 - 4*a1*ang0*k0*k1**5*nx*y0 + 2*a0*k1*(-2*k0**4 + 5*k0**3*k1 - 5*k0*k1**3 + 2*k1**4)*(3*a1*nx + 2*H*nz)*np.cos(ang0*k0 + phi0) + 2*a0*a1*k0*k1*(2*k0**3 - k0**2*k1 - 2*k0*k1**2 + k1**3)*nx*np.cos(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1) - 4*a0*k0**4*k1*nx*x0*np.cos(ang0*k0 - ang0*k1 + phi0 - phi1) + 6*a0*k0**3*k1**2*nx*x0*np.cos(ang0*k0 - ang0*k1 + phi0 - phi1) + 6*a0*k0**2*k1**3*nx*x0*np.cos(ang0*k0 - ang0*k1 + phi0 - phi1) - 4*a0*k0*k1**4*nx*x0*np.cos(ang0*k0 - ang0*k1 + phi0 - phi1) - 4*a0*k0**4*k1*ny*y0*np.cos(ang0*k0 - ang0*k1 + phi0 - phi1) + 6*a0*k0**3*k1**2*ny*y0*np.cos(ang0*k0 - ang0*k1 + phi0 - phi1) + 6*a0*k0**2*k1**3*ny*y0*np.cos(ang0*k0 - ang0*k1 + phi0 - phi1) - 4*a0*k0*k1**4*ny*y0*np.cos(ang0*k0 - ang0*k1 + phi0 - phi1) - 2*a0**2*k0**4*k1*nx*np.cos(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1) + 4*a0**2*k0**3*k1**2*nx*np.cos(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1) + 2*a0**2*k0**2*k1**3*nx*np.cos(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1) - 4*a0**2*k0*k1**4*nx*np.cos(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1) - 4*a0**2*k0**5*nx*np.cos(ang0*k1 + phi1) - 8*a1**2*k0**5*nx*np.cos(ang0*k1 + phi1) + 10*a0**2*k0**4*k1*nx*np.cos(ang0*k1 + phi1) + 20*a1**2*k0**4*k1*nx*np.cos(ang0*k1 + phi1) - 10*a0**2*k0**2*k1**3*nx*np.cos(ang0*k1 + phi1) - 20*a1**2*k0**2*k1**3*nx*np.cos(ang0*k1 + phi1) + 4*a0**2*k0*k1**4*nx*np.cos(ang0*k1 + phi1) + 8*a1**2*k0*k1**4*nx*np.cos(ang0*k1 + phi1) - 8*a1*H*k0**5*nz*np.cos(ang0*k1 + phi1) + 20*a1*H*k0**4*k1*nz*np.cos(ang0*k1 + phi1) - 20*a1*H*k0**2*k1**3*nz*np.cos(ang0*k1 + phi1) + 8*a1*H*k0*k1**4*nz*np.cos(ang0*k1 + phi1) - 2*a1*k0**5*nx*x0*np.cos(2*(ang0*k1 + phi1)) + 5*a1*k0**4*k1*nx*x0*np.cos(2*(ang0*k1 + phi1)) - 5*a1*k0**2*k1**3*nx*x0*np.cos(2*(ang0*k1 + phi1)) + 2*a1*k0*k1**4*nx*x0*np.cos(2*(ang0*k1 + phi1)) - 2*a1*k0**5*ny*y0*np.cos(2*(ang0*k1 + phi1)) + 5*a1*k0**4*k1*ny*y0*np.cos(2*(ang0*k1 + phi1)) - 5*a1*k0**2*k1**3*ny*y0*np.cos(2*(ang0*k1 + phi1)) + 2*a1*k0*k1**4*ny*y0*np.cos(2*(ang0*k1 + phi1)) - 4*a0*k0**4*k1*nx*x0*np.cos(ang0*(k0 + k1) + phi0 + phi1) + 14*a0*k0**3*k1**2*nx*x0*np.cos(ang0*(k0 + k1) + phi0 + phi1) - 14*a0*k0**2*k1**3*nx*x0*np.cos(ang0*(k0 + k1) + phi0 + phi1) + 4*a0*k0*k1**4*nx*x0*np.cos(ang0*(k0 + k1) + phi0 + phi1) - 4*a0*k0**4*k1*ny*y0*np.cos(ang0*(k0 + k1) + phi0 + phi1) + 14*a0*k0**3*k1**2*ny*y0*np.cos(ang0*(k0 + k1) + phi0 + phi1) - 14*a0*k0**2*k1**3*ny*y0*np.cos(ang0*(k0 + k1) + phi0 + phi1) + 4*a0*k0*k1**4*ny*y0*np.cos(ang0*(k0 + k1) + phi0 + phi1) - 4*a0*a1*k0**4*k1*ny*np.sin(ang0*k0 + phi0) + 10*a0*a1*k0**3*k1**2*ny*np.sin(ang0*k0 + phi0) - 10*a0*a1*k0*k1**4*ny*np.sin(ang0*k0 + phi0) + 4*a0*a1*k1**5*ny*np.sin(ang0*k0 + phi0) + 4*a0*a1*k0**4*k1*ny*np.sin(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1) - 2*a0*a1*k0**3*k1**2*ny*np.sin(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1) - 4*a0*a1*k0**2*k1**3*ny*np.sin(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1) + 2*a0*a1*k0*k1**4*ny*np.sin(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1) - 4*a0*k0**4*k1*ny*x0*np.sin(ang0*k0 - ang0*k1 + phi0 - phi1) + 6*a0*k0**3*k1**2*ny*x0*np.sin(ang0*k0 - ang0*k1 + phi0 - phi1) + 6*a0*k0**2*k1**3*ny*x0*np.sin(ang0*k0 - ang0*k1 + phi0 - phi1) - 4*a0*k0*k1**4*ny*x0*np.sin(ang0*k0 - ang0*k1 + phi0 - phi1) + 4*a0*k0**4*k1*nx*y0*np.sin(ang0*k0 - ang0*k1 + phi0 - phi1) - 6*a0*k0**3*k1**2*nx*y0*np.sin(ang0*k0 - ang0*k1 + phi0 - phi1) - 6*a0*k0**2*k1**3*nx*y0*np.sin(ang0*k0 - ang0*k1 + phi0 - phi1) + 4*a0*k0*k1**4*nx*y0*np.sin(ang0*k0 - ang0*k1 + phi0 - phi1) - 2*a0**2*k0**4*k1*ny*np.sin(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1) + 4*a0**2*k0**3*k1**2*ny*np.sin(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1) + 2*a0**2*k0**2*k1**3*ny*np.sin(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1) - 4*a0**2*k0*k1**4*ny*np.sin(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1) + 4*a0**2*k0**5*ny*np.sin(ang0*k1 + phi1) - 10*a0**2*k0**4*k1*ny*np.sin(ang0*k1 + phi1) + 10*a0**2*k0**2*k1**3*ny*np.sin(ang0*k1 + phi1) - 4*a0**2*k0*k1**4*ny*np.sin(ang0*k1 + phi1) + 2*a1*k0**5*ny*x0*np.sin(2*(ang0*k1 + phi1)) - 5*a1*k0**4*k1*ny*x0*np.sin(2*(ang0*k1 + phi1)) + 5*a1*k0**2*k1**3*ny*x0*np.sin(2*(ang0*k1 + phi1)) - 2*a1*k0*k1**4*ny*x0*np.sin(2*(ang0*k1 + phi1)) - 2*a1*k0**5*nx*y0*np.sin(2*(ang0*k1 + phi1)) + 5*a1*k0**4*k1*nx*y0*np.sin(2*(ang0*k1 + phi1)) - 5*a1*k0**2*k1**3*nx*y0*np.sin(2*(ang0*k1 + phi1)) + 2*a1*k0*k1**4*nx*y0*np.sin(2*(ang0*k1 + phi1)) + 4*a0*k0**4*k1*ny*x0*np.sin(ang0*(k0 + k1) + phi0 + phi1) - 14*a0*k0**3*k1**2*ny*x0*np.sin(ang0*(k0 + k1) + phi0 + phi1) + 14*a0*k0**2*k1**3*ny*x0*np.sin(ang0*(k0 + k1) + phi0 + phi1) - 4*a0*k0*k1**4*ny*x0*np.sin(ang0*(k0 + k1) + phi0 + phi1) - 4*a0*k0**4*k1*nx*y0*np.sin(ang0*(k0 + k1) + phi0 + phi1) + 14*a0*k0**3*k1**2*nx*y0*np.sin(ang0*(k0 + k1) + phi0 + phi1) - 14*a0*k0**2*k1**3*nx*y0*np.sin(ang0*(k0 + k1) + phi0 + phi1) + 4*a0*k0*k1**4*nx*y0*np.sin(ang0*(k0 + k1) + phi0 + phi1))/(4.*k0*(k0 - 2*k1)*(k0 - k1)*(2*k0 - k1)*k1*(k0 + k1)))
            res1 = ((-4*a1*ang1*k0**5*k1*ny*x0 + 10*a1*ang1*k0**4*k1**2*ny*x0 - 10*a1*ang1*k0**2*k1**4*ny*x0 + 4*a1*ang1*k0*k1**5*ny*x0 + 4*a1*ang1*k0**5*k1*nx*y0 - 10*a1*ang1*k0**4*k1**2*nx*y0 + 10*a1*ang1*k0**2*k1**4*nx*y0 - 4*a1*ang1*k0*k1**5*nx*y0 + 2*a0*k1*(-2*k0**4 + 5*k0**3*k1 - 5*k0*k1**3 + 2*k1**4)*(3*a1*nx + 2*H*nz)*np.cos(ang1*k0 + phi0) + 2*a0*a1*k0*k1*(2*k0**3 - k0**2*k1 - 2*k0*k1**2 + k1**3)*nx*np.cos(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1) - 4*a0*k0**4*k1*nx*x0*np.cos(ang1*k0 - ang1*k1 + phi0 - phi1) + 6*a0*k0**3*k1**2*nx*x0*np.cos(ang1*k0 - ang1*k1 + phi0 - phi1) + 6*a0*k0**2*k1**3*nx*x0*np.cos(ang1*k0 - ang1*k1 + phi0 - phi1) - 4*a0*k0*k1**4*nx*x0*np.cos(ang1*k0 - ang1*k1 + phi0 - phi1) - 4*a0*k0**4*k1*ny*y0*np.cos(ang1*k0 - ang1*k1 + phi0 - phi1) + 6*a0*k0**3*k1**2*ny*y0*np.cos(ang1*k0 - ang1*k1 + phi0 - phi1) + 6*a0*k0**2*k1**3*ny*y0*np.cos(ang1*k0 - ang1*k1 + phi0 - phi1) - 4*a0*k0*k1**4*ny*y0*np.cos(ang1*k0 - ang1*k1 + phi0 - phi1) - 2*a0**2*k0**4*k1*nx*np.cos(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1) + 4*a0**2*k0**3*k1**2*nx*np.cos(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1) + 2*a0**2*k0**2*k1**3*nx*np.cos(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1) - 4*a0**2*k0*k1**4*nx*np.cos(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1) - 4*a0**2*k0**5*nx*np.cos(ang1*k1 + phi1) - 8*a1**2*k0**5*nx*np.cos(ang1*k1 + phi1) + 10*a0**2*k0**4*k1*nx*np.cos(ang1*k1 + phi1) + 20*a1**2*k0**4*k1*nx*np.cos(ang1*k1 + phi1) - 10*a0**2*k0**2*k1**3*nx*np.cos(ang1*k1 + phi1) - 20*a1**2*k0**2*k1**3*nx*np.cos(ang1*k1 + phi1) + 4*a0**2*k0*k1**4*nx*np.cos(ang1*k1 + phi1) + 8*a1**2*k0*k1**4*nx*np.cos(ang1*k1 + phi1) - 8*a1*H*k0**5*nz*np.cos(ang1*k1 + phi1) + 20*a1*H*k0**4*k1*nz*np.cos(ang1*k1 + phi1) - 20*a1*H*k0**2*k1**3*nz*np.cos(ang1*k1 + phi1) + 8*a1*H*k0*k1**4*nz*np.cos(ang1*k1 + phi1) - 2*a1*k0**5*nx*x0*np.cos(2*(ang1*k1 + phi1)) + 5*a1*k0**4*k1*nx*x0*np.cos(2*(ang1*k1 + phi1)) - 5*a1*k0**2*k1**3*nx*x0*np.cos(2*(ang1*k1 + phi1)) + 2*a1*k0*k1**4*nx*x0*np.cos(2*(ang1*k1 + phi1)) - 2*a1*k0**5*ny*y0*np.cos(2*(ang1*k1 + phi1)) + 5*a1*k0**4*k1*ny*y0*np.cos(2*(ang1*k1 + phi1)) - 5*a1*k0**2*k1**3*ny*y0*np.cos(2*(ang1*k1 + phi1)) + 2*a1*k0*k1**4*ny*y0*np.cos(2*(ang1*k1 + phi1)) - 4*a0*k0**4*k1*nx*x0*np.cos(ang1*(k0 + k1) + phi0 + phi1) + 14*a0*k0**3*k1**2*nx*x0*np.cos(ang1*(k0 + k1) + phi0 + phi1) - 14*a0*k0**2*k1**3*nx*x0*np.cos(ang1*(k0 + k1) + phi0 + phi1) + 4*a0*k0*k1**4*nx*x0*np.cos(ang1*(k0 + k1) + phi0 + phi1) - 4*a0*k0**4*k1*ny*y0*np.cos(ang1*(k0 + k1) + phi0 + phi1) + 14*a0*k0**3*k1**2*ny*y0*np.cos(ang1*(k0 + k1) + phi0 + phi1) - 14*a0*k0**2*k1**3*ny*y0*np.cos(ang1*(k0 + k1) + phi0 + phi1) + 4*a0*k0*k1**4*ny*y0*np.cos(ang1*(k0 + k1) + phi0 + phi1) - 4*a0*a1*k0**4*k1*ny*np.sin(ang1*k0 + phi0) + 10*a0*a1*k0**3*k1**2*ny*np.sin(ang1*k0 + phi0) - 10*a0*a1*k0*k1**4*ny*np.sin(ang1*k0 + phi0) + 4*a0*a1*k1**5*ny*np.sin(ang1*k0 + phi0) + 4*a0*a1*k0**4*k1*ny*np.sin(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1) - 2*a0*a1*k0**3*k1**2*ny*np.sin(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1) - 4*a0*a1*k0**2*k1**3*ny*np.sin(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1) + 2*a0*a1*k0*k1**4*ny*np.sin(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1) - 4*a0*k0**4*k1*ny*x0*np.sin(ang1*k0 - ang1*k1 + phi0 - phi1) + 6*a0*k0**3*k1**2*ny*x0*np.sin(ang1*k0 - ang1*k1 + phi0 - phi1) + 6*a0*k0**2*k1**3*ny*x0*np.sin(ang1*k0 - ang1*k1 + phi0 - phi1) - 4*a0*k0*k1**4*ny*x0*np.sin(ang1*k0 - ang1*k1 + phi0 - phi1) + 4*a0*k0**4*k1*nx*y0*np.sin(ang1*k0 - ang1*k1 + phi0 - phi1) - 6*a0*k0**3*k1**2*nx*y0*np.sin(ang1*k0 - ang1*k1 + phi0 - phi1) - 6*a0*k0**2*k1**3*nx*y0*np.sin(ang1*k0 - ang1*k1 + phi0 - phi1) + 4*a0*k0*k1**4*nx*y0*np.sin(ang1*k0 - ang1*k1 + phi0 - phi1) - 2*a0**2*k0**4*k1*ny*np.sin(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1) + 4*a0**2*k0**3*k1**2*ny*np.sin(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1) + 2*a0**2*k0**2*k1**3*ny*np.sin(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1) - 4*a0**2*k0*k1**4*ny*np.sin(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1) + 4*a0**2*k0**5*ny*np.sin(ang1*k1 + phi1) - 10*a0**2*k0**4*k1*ny*np.sin(ang1*k1 + phi1) + 10*a0**2*k0**2*k1**3*ny*np.sin(ang1*k1 + phi1) - 4*a0**2*k0*k1**4*ny*np.sin(ang1*k1 + phi1) + 2*a1*k0**5*ny*x0*np.sin(2*(ang1*k1 + phi1)) - 5*a1*k0**4*k1*ny*x0*np.sin(2*(ang1*k1 + phi1)) + 5*a1*k0**2*k1**3*ny*x0*np.sin(2*(ang1*k1 + phi1)) - 2*a1*k0*k1**4*ny*x0*np.sin(2*(ang1*k1 + phi1)) - 2*a1*k0**5*nx*y0*np.sin(2*(ang1*k1 + phi1)) + 5*a1*k0**4*k1*nx*y0*np.sin(2*(ang1*k1 + phi1)) - 5*a1*k0**2*k1**3*nx*y0*np.sin(2*(ang1*k1 + phi1)) + 2*a1*k0*k1**4*nx*y0*np.sin(2*(ang1*k1 + phi1)) + 4*a0*k0**4*k1*ny*x0*np.sin(ang1*(k0 + k1) + phi0 + phi1) - 14*a0*k0**3*k1**2*ny*x0*np.sin(ang1*(k0 + k1) + phi0 + phi1) + 14*a0*k0**2*k1**3*ny*x0*np.sin(ang1*(k0 + k1) + phi0 + phi1) - 4*a0*k0*k1**4*ny*x0*np.sin(ang1*(k0 + k1) + phi0 + phi1) - 4*a0*k0**4*k1*nx*y0*np.sin(ang1*(k0 + k1) + phi0 + phi1) + 14*a0*k0**3*k1**2*nx*y0*np.sin(ang1*(k0 + k1) + phi0 + phi1) - 14*a0*k0**2*k1**3*nx*y0*np.sin(ang1*(k0 + k1) + phi0 + phi1) + 4*a0*k0*k1**4*nx*y0*np.sin(ang1*(k0 + k1) + phi0 + phi1))/(4.*k0*(k0 - 2*k1)*(k0 - k1)*(2*k0 - k1)*k1*(k0 + k1)))
            return res1 - res0
        else:
            print('invalid index j = ' + str(j) + ' in Iij_c')
            return 0.0
    elif i == 1:
        if j == 0:
            res0 = ((4*a1*ang0*k0**5*k1*nx*x0 - 10*a1*ang0*k0**4*k1**2*nx*x0 + 10*a1*ang0*k0**2*k1**4*nx*x0 - 4*a1*ang0*k0*k1**5*nx*x0 + 4*a1*ang0*k0**5*k1*ny*y0 - 10*a1*ang0*k0**4*k1**2*ny*y0 + 10*a1*ang0*k0**2*k1**4*ny*y0 - 4*a1*ang0*k0*k1**5*ny*y0 + 2*a0*a1*k1*(-2*k0**4 + 5*k0**3*k1 - 5*k0*k1**3 + 2*k1**4)*ny*np.cos(ang0*k0 + phi0) - 2*a0*a1*k0*k1*(2*k0**3 - k0**2*k1 - 2*k0*k1**2 + k1**3)*ny*np.cos(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1) - 4*a0*k0**4*k1*ny*x0*np.cos(ang0*k0 - ang0*k1 + phi0 - phi1) + 6*a0*k0**3*k1**2*ny*x0*np.cos(ang0*k0 - ang0*k1 + phi0 - phi1) + 6*a0*k0**2*k1**3*ny*x0*np.cos(ang0*k0 - ang0*k1 + phi0 - phi1) - 4*a0*k0*k1**4*ny*x0*np.cos(ang0*k0 - ang0*k1 + phi0 - phi1) + 4*a0*k0**4*k1*nx*y0*np.cos(ang0*k0 - ang0*k1 + phi0 - phi1) - 6*a0*k0**3*k1**2*nx*y0*np.cos(ang0*k0 - ang0*k1 + phi0 - phi1) - 6*a0*k0**2*k1**3*nx*y0*np.cos(ang0*k0 - ang0*k1 + phi0 - phi1) + 4*a0*k0*k1**4*nx*y0*np.cos(ang0*k0 - ang0*k1 + phi0 - phi1) - 2*a0**2*k0**4*k1*ny*np.cos(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1) + 4*a0**2*k0**3*k1**2*ny*np.cos(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1) + 2*a0**2*k0**2*k1**3*ny*np.cos(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1) - 4*a0**2*k0*k1**4*ny*np.cos(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1) + 4*a0**2*k0**5*ny*np.cos(ang0*k1 + phi1) - 10*a0**2*k0**4*k1*ny*np.cos(ang0*k1 + phi1) + 10*a0**2*k0**2*k1**3*ny*np.cos(ang0*k1 + phi1) - 4*a0**2*k0*k1**4*ny*np.cos(ang0*k1 + phi1) + 2*a1*k0**5*ny*x0*np.cos(2*(ang0*k1 + phi1)) - 5*a1*k0**4*k1*ny*x0*np.cos(2*(ang0*k1 + phi1)) + 5*a1*k0**2*k1**3*ny*x0*np.cos(2*(ang0*k1 + phi1)) - 2*a1*k0*k1**4*ny*x0*np.cos(2*(ang0*k1 + phi1)) - 2*a1*k0**5*nx*y0*np.cos(2*(ang0*k1 + phi1)) + 5*a1*k0**4*k1*nx*y0*np.cos(2*(ang0*k1 + phi1)) - 5*a1*k0**2*k1**3*nx*y0*np.cos(2*(ang0*k1 + phi1)) + 2*a1*k0*k1**4*nx*y0*np.cos(2*(ang0*k1 + phi1)) + 4*a0*k0**4*k1*ny*x0*np.cos(ang0*(k0 + k1) + phi0 + phi1) - 14*a0*k0**3*k1**2*ny*x0*np.cos(ang0*(k0 + k1) + phi0 + phi1) + 14*a0*k0**2*k1**3*ny*x0*np.cos(ang0*(k0 + k1) + phi0 + phi1) - 4*a0*k0*k1**4*ny*x0*np.cos(ang0*(k0 + k1) + phi0 + phi1) - 4*a0*k0**4*k1*nx*y0*np.cos(ang0*(k0 + k1) + phi0 + phi1) + 14*a0*k0**3*k1**2*nx*y0*np.cos(ang0*(k0 + k1) + phi0 + phi1) - 14*a0*k0**2*k1**3*nx*y0*np.cos(ang0*(k0 + k1) + phi0 + phi1) + 4*a0*k0*k1**4*nx*y0*np.cos(ang0*(k0 + k1) + phi0 + phi1) + 12*a0*a1*k0**4*k1*nx*np.sin(ang0*k0 + phi0) - 30*a0*a1*k0**3*k1**2*nx*np.sin(ang0*k0 + phi0) + 30*a0*a1*k0*k1**4*nx*np.sin(ang0*k0 + phi0) - 12*a0*a1*k1**5*nx*np.sin(ang0*k0 + phi0) + 8*a0*H*k0**4*k1*nz*np.sin(ang0*k0 + phi0) - 20*a0*H*k0**3*k1**2*nz*np.sin(ang0*k0 + phi0) + 20*a0*H*k0*k1**4*nz*np.sin(ang0*k0 + phi0) - 8*a0*H*k1**5*nz*np.sin(ang0*k0 + phi0) + 4*a0*a1*k0**4*k1*nx*np.sin(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1) - 2*a0*a1*k0**3*k1**2*nx*np.sin(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1) - 4*a0*a1*k0**2*k1**3*nx*np.sin(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1) + 2*a0*a1*k0*k1**4*nx*np.sin(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1) + 4*a0*k0**4*k1*nx*x0*np.sin(ang0*k0 - ang0*k1 + phi0 - phi1) - 6*a0*k0**3*k1**2*nx*x0*np.sin(ang0*k0 - ang0*k1 + phi0 - phi1) - 6*a0*k0**2*k1**3*nx*x0*np.sin(ang0*k0 - ang0*k1 + phi0 - phi1) + 4*a0*k0*k1**4*nx*x0*np.sin(ang0*k0 - ang0*k1 + phi0 - phi1) + 4*a0*k0**4*k1*ny*y0*np.sin(ang0*k0 - ang0*k1 + phi0 - phi1) - 6*a0*k0**3*k1**2*ny*y0*np.sin(ang0*k0 - ang0*k1 + phi0 - phi1) - 6*a0*k0**2*k1**3*ny*y0*np.sin(ang0*k0 - ang0*k1 + phi0 - phi1) + 4*a0*k0*k1**4*ny*y0*np.sin(ang0*k0 - ang0*k1 + phi0 - phi1) + 2*a0**2*k0**4*k1*nx*np.sin(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1) - 4*a0**2*k0**3*k1**2*nx*np.sin(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1) - 2*a0**2*k0**2*k1**3*nx*np.sin(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1) + 4*a0**2*k0*k1**4*nx*np.sin(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1) + 4*a0**2*k0**5*nx*np.sin(ang0*k1 + phi1) + 8*a1**2*k0**5*nx*np.sin(ang0*k1 + phi1) - 10*a0**2*k0**4*k1*nx*np.sin(ang0*k1 + phi1) - 20*a1**2*k0**4*k1*nx*np.sin(ang0*k1 + phi1) + 10*a0**2*k0**2*k1**3*nx*np.sin(ang0*k1 + phi1) + 20*a1**2*k0**2*k1**3*nx*np.sin(ang0*k1 + phi1) - 4*a0**2*k0*k1**4*nx*np.sin(ang0*k1 + phi1) - 8*a1**2*k0*k1**4*nx*np.sin(ang0*k1 + phi1) + 8*a1*H*k0**5*nz*np.sin(ang0*k1 + phi1) - 20*a1*H*k0**4*k1*nz*np.sin(ang0*k1 + phi1) + 20*a1*H*k0**2*k1**3*nz*np.sin(ang0*k1 + phi1) - 8*a1*H*k0*k1**4*nz*np.sin(ang0*k1 + phi1) + 2*a1*k0**5*nx*x0*np.sin(2*(ang0*k1 + phi1)) - 5*a1*k0**4*k1*nx*x0*np.sin(2*(ang0*k1 + phi1)) + 5*a1*k0**2*k1**3*nx*x0*np.sin(2*(ang0*k1 + phi1)) - 2*a1*k0*k1**4*nx*x0*np.sin(2*(ang0*k1 + phi1)) + 2*a1*k0**5*ny*y0*np.sin(2*(ang0*k1 + phi1)) - 5*a1*k0**4*k1*ny*y0*np.sin(2*(ang0*k1 + phi1)) + 5*a1*k0**2*k1**3*ny*y0*np.sin(2*(ang0*k1 + phi1)) - 2*a1*k0*k1**4*ny*y0*np.sin(2*(ang0*k1 + phi1)) + 4*a0*k0**4*k1*nx*x0*np.sin(ang0*(k0 + k1) + phi0 + phi1) - 14*a0*k0**3*k1**2*nx*x0*np.sin(ang0*(k0 + k1) + phi0 + phi1) + 14*a0*k0**2*k1**3*nx*x0*np.sin(ang0*(k0 + k1) + phi0 + phi1) - 4*a0*k0*k1**4*nx*x0*np.sin(ang0*(k0 + k1) + phi0 + phi1) + 4*a0*k0**4*k1*ny*y0*np.sin(ang0*(k0 + k1) + phi0 + phi1) - 14*a0*k0**3*k1**2*ny*y0*np.sin(ang0*(k0 + k1) + phi0 + phi1) + 14*a0*k0**2*k1**3*ny*y0*np.sin(ang0*(k0 + k1) + phi0 + phi1) - 4*a0*k0*k1**4*ny*y0*np.sin(ang0*(k0 + k1) + phi0 + phi1))/(4.*k0*(k0 - 2*k1)*(k0 - k1)*(2*k0 - k1)*k1*(k0 + k1)))
            res1 = ((4*a1*ang1*k0**5*k1*nx*x0 - 10*a1*ang1*k0**4*k1**2*nx*x0 + 10*a1*ang1*k0**2*k1**4*nx*x0 - 4*a1*ang1*k0*k1**5*nx*x0 + 4*a1*ang1*k0**5*k1*ny*y0 - 10*a1*ang1*k0**4*k1**2*ny*y0 + 10*a1*ang1*k0**2*k1**4*ny*y0 - 4*a1*ang1*k0*k1**5*ny*y0 + 2*a0*a1*k1*(-2*k0**4 + 5*k0**3*k1 - 5*k0*k1**3 + 2*k1**4)*ny*np.cos(ang1*k0 + phi0) - 2*a0*a1*k0*k1*(2*k0**3 - k0**2*k1 - 2*k0*k1**2 + k1**3)*ny*np.cos(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1) - 4*a0*k0**4*k1*ny*x0*np.cos(ang1*k0 - ang1*k1 + phi0 - phi1) + 6*a0*k0**3*k1**2*ny*x0*np.cos(ang1*k0 - ang1*k1 + phi0 - phi1) + 6*a0*k0**2*k1**3*ny*x0*np.cos(ang1*k0 - ang1*k1 + phi0 - phi1) - 4*a0*k0*k1**4*ny*x0*np.cos(ang1*k0 - ang1*k1 + phi0 - phi1) + 4*a0*k0**4*k1*nx*y0*np.cos(ang1*k0 - ang1*k1 + phi0 - phi1) - 6*a0*k0**3*k1**2*nx*y0*np.cos(ang1*k0 - ang1*k1 + phi0 - phi1) - 6*a0*k0**2*k1**3*nx*y0*np.cos(ang1*k0 - ang1*k1 + phi0 - phi1) + 4*a0*k0*k1**4*nx*y0*np.cos(ang1*k0 - ang1*k1 + phi0 - phi1) - 2*a0**2*k0**4*k1*ny*np.cos(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1) + 4*a0**2*k0**3*k1**2*ny*np.cos(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1) + 2*a0**2*k0**2*k1**3*ny*np.cos(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1) - 4*a0**2*k0*k1**4*ny*np.cos(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1) + 4*a0**2*k0**5*ny*np.cos(ang1*k1 + phi1) - 10*a0**2*k0**4*k1*ny*np.cos(ang1*k1 + phi1) + 10*a0**2*k0**2*k1**3*ny*np.cos(ang1*k1 + phi1) - 4*a0**2*k0*k1**4*ny*np.cos(ang1*k1 + phi1) + 2*a1*k0**5*ny*x0*np.cos(2*(ang1*k1 + phi1)) - 5*a1*k0**4*k1*ny*x0*np.cos(2*(ang1*k1 + phi1)) + 5*a1*k0**2*k1**3*ny*x0*np.cos(2*(ang1*k1 + phi1)) - 2*a1*k0*k1**4*ny*x0*np.cos(2*(ang1*k1 + phi1)) - 2*a1*k0**5*nx*y0*np.cos(2*(ang1*k1 + phi1)) + 5*a1*k0**4*k1*nx*y0*np.cos(2*(ang1*k1 + phi1)) - 5*a1*k0**2*k1**3*nx*y0*np.cos(2*(ang1*k1 + phi1)) + 2*a1*k0*k1**4*nx*y0*np.cos(2*(ang1*k1 + phi1)) + 4*a0*k0**4*k1*ny*x0*np.cos(ang1*(k0 + k1) + phi0 + phi1) - 14*a0*k0**3*k1**2*ny*x0*np.cos(ang1*(k0 + k1) + phi0 + phi1) + 14*a0*k0**2*k1**3*ny*x0*np.cos(ang1*(k0 + k1) + phi0 + phi1) - 4*a0*k0*k1**4*ny*x0*np.cos(ang1*(k0 + k1) + phi0 + phi1) - 4*a0*k0**4*k1*nx*y0*np.cos(ang1*(k0 + k1) + phi0 + phi1) + 14*a0*k0**3*k1**2*nx*y0*np.cos(ang1*(k0 + k1) + phi0 + phi1) - 14*a0*k0**2*k1**3*nx*y0*np.cos(ang1*(k0 + k1) + phi0 + phi1) + 4*a0*k0*k1**4*nx*y0*np.cos(ang1*(k0 + k1) + phi0 + phi1) + 12*a0*a1*k0**4*k1*nx*np.sin(ang1*k0 + phi0) - 30*a0*a1*k0**3*k1**2*nx*np.sin(ang1*k0 + phi0) + 30*a0*a1*k0*k1**4*nx*np.sin(ang1*k0 + phi0) - 12*a0*a1*k1**5*nx*np.sin(ang1*k0 + phi0) + 8*a0*H*k0**4*k1*nz*np.sin(ang1*k0 + phi0) - 20*a0*H*k0**3*k1**2*nz*np.sin(ang1*k0 + phi0) + 20*a0*H*k0*k1**4*nz*np.sin(ang1*k0 + phi0) - 8*a0*H*k1**5*nz*np.sin(ang1*k0 + phi0) + 4*a0*a1*k0**4*k1*nx*np.sin(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1) - 2*a0*a1*k0**3*k1**2*nx*np.sin(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1) - 4*a0*a1*k0**2*k1**3*nx*np.sin(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1) + 2*a0*a1*k0*k1**4*nx*np.sin(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1) + 4*a0*k0**4*k1*nx*x0*np.sin(ang1*k0 - ang1*k1 + phi0 - phi1) - 6*a0*k0**3*k1**2*nx*x0*np.sin(ang1*k0 - ang1*k1 + phi0 - phi1) - 6*a0*k0**2*k1**3*nx*x0*np.sin(ang1*k0 - ang1*k1 + phi0 - phi1) + 4*a0*k0*k1**4*nx*x0*np.sin(ang1*k0 - ang1*k1 + phi0 - phi1) + 4*a0*k0**4*k1*ny*y0*np.sin(ang1*k0 - ang1*k1 + phi0 - phi1) - 6*a0*k0**3*k1**2*ny*y0*np.sin(ang1*k0 - ang1*k1 + phi0 - phi1) - 6*a0*k0**2*k1**3*ny*y0*np.sin(ang1*k0 - ang1*k1 + phi0 - phi1) + 4*a0*k0*k1**4*ny*y0*np.sin(ang1*k0 - ang1*k1 + phi0 - phi1) + 2*a0**2*k0**4*k1*nx*np.sin(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1) - 4*a0**2*k0**3*k1**2*nx*np.sin(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1) - 2*a0**2*k0**2*k1**3*nx*np.sin(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1) + 4*a0**2*k0*k1**4*nx*np.sin(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1) + 4*a0**2*k0**5*nx*np.sin(ang1*k1 + phi1) + 8*a1**2*k0**5*nx*np.sin(ang1*k1 + phi1) - 10*a0**2*k0**4*k1*nx*np.sin(ang1*k1 + phi1) - 20*a1**2*k0**4*k1*nx*np.sin(ang1*k1 + phi1) + 10*a0**2*k0**2*k1**3*nx*np.sin(ang1*k1 + phi1) + 20*a1**2*k0**2*k1**3*nx*np.sin(ang1*k1 + phi1) - 4*a0**2*k0*k1**4*nx*np.sin(ang1*k1 + phi1) - 8*a1**2*k0*k1**4*nx*np.sin(ang1*k1 + phi1) + 8*a1*H*k0**5*nz*np.sin(ang1*k1 + phi1) - 20*a1*H*k0**4*k1*nz*np.sin(ang1*k1 + phi1) + 20*a1*H*k0**2*k1**3*nz*np.sin(ang1*k1 + phi1) - 8*a1*H*k0*k1**4*nz*np.sin(ang1*k1 + phi1) + 2*a1*k0**5*nx*x0*np.sin(2*(ang1*k1 + phi1)) - 5*a1*k0**4*k1*nx*x0*np.sin(2*(ang1*k1 + phi1)) + 5*a1*k0**2*k1**3*nx*x0*np.sin(2*(ang1*k1 + phi1)) - 2*a1*k0*k1**4*nx*x0*np.sin(2*(ang1*k1 + phi1)) + 2*a1*k0**5*ny*y0*np.sin(2*(ang1*k1 + phi1)) - 5*a1*k0**4*k1*ny*y0*np.sin(2*(ang1*k1 + phi1)) + 5*a1*k0**2*k1**3*ny*y0*np.sin(2*(ang1*k1 + phi1)) - 2*a1*k0*k1**4*ny*y0*np.sin(2*(ang1*k1 + phi1)) + 4*a0*k0**4*k1*nx*x0*np.sin(ang1*(k0 + k1) + phi0 + phi1) - 14*a0*k0**3*k1**2*nx*x0*np.sin(ang1*(k0 + k1) + phi0 + phi1) + 14*a0*k0**2*k1**3*nx*x0*np.sin(ang1*(k0 + k1) + phi0 + phi1) - 4*a0*k0*k1**4*nx*x0*np.sin(ang1*(k0 + k1) + phi0 + phi1) + 4*a0*k0**4*k1*ny*y0*np.sin(ang1*(k0 + k1) + phi0 + phi1) - 14*a0*k0**3*k1**2*ny*y0*np.sin(ang1*(k0 + k1) + phi0 + phi1) + 14*a0*k0**2*k1**3*ny*y0*np.sin(ang1*(k0 + k1) + phi0 + phi1) - 4*a0*k0*k1**4*ny*y0*np.sin(ang1*(k0 + k1) + phi0 + phi1))/(4.*k0*(k0 - 2*k1)*(k0 - k1)*(2*k0 - k1)*k1*(k0 + k1)))
            return res1 - res0
        elif j == 1:
            res0 = (((-6*a0*a1*(nx*x0 + ny*y0)*np.cos(ang0*k0 + phi0))/k0 - (3*a0**2*(2*a1*nx + H*nz)*np.cos(2*(ang0*k0 + phi0)))/k0 + (3*a0*a1**2*nx*np.cos(ang0*k0 - 3*ang0*k1 + phi0 - 3*phi1))/(k0 - 3*k1) - (3*a0**2*nx*x0*np.cos(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1))/(2*k0 - k1) - (3*a0**2*ny*y0*np.cos(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1))/(2*k0 - k1) - (3*a0**3*k0*nx*np.cos(3*ang0*k0 - ang0*k1 + 3*phi0 - phi1))/((3*k0 - k1)*(k0 + k1)) - (3*a0**3*k1*nx*np.cos(3*ang0*k0 - ang0*k1 + 3*phi0 - phi1))/((3*k0 - k1)*(k0 + k1)) - (3*a1**2*nx*x0*np.cos(ang0*k1 + phi1))/k1 - (3*a1**2*ny*y0*np.cos(ang0*k1 + phi1))/k1 - (3*a0**2*a1*nx*np.cos(2*(ang0*k1 + phi1)))/k1 - (3*a1**3*nx*np.cos(2*(ang0*k1 + phi1)))/k1 - (3*a1**2*H*nz*np.cos(2*(ang0*k1 + phi1)))/k1 - (a1**2*nx*x0*np.cos(3*(ang0*k1 + phi1)))/k1 - (a1**2*ny*y0*np.cos(3*(ang0*k1 + phi1)))/k1 - (15*a0*a1**2*nx*np.cos(ang0*(k0 + k1) + phi0 + phi1))/(k0 + k1) - (9*a0**3*k0*nx*np.cos(ang0*(k0 + k1) + phi0 + phi1))/((3*k0 - k1)*(k0 + k1)) + (3*a0**3*k1*nx*np.cos(ang0*(k0 + k1) + phi0 + phi1))/((3*k0 - k1)*(k0 + k1)) - (12*a0*a1*H*nz*np.cos(ang0*(k0 + k1) + phi0 + phi1))/(k0 + k1) - (3*a0**2*nx*x0*np.cos(2*ang0*k0 + ang0*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (3*a0**2*ny*y0*np.cos(2*ang0*k0 + ang0*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (6*a0*a1*nx*x0*np.cos(ang0*k0 + 2*ang0*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (6*a0*a1*ny*y0*np.cos(ang0*k0 + 2*ang0*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (6*a0*a1*ny*x0*np.sin(ang0*k0 + phi0))/k0 + (6*a0*a1*nx*y0*np.sin(ang0*k0 + phi0))/k0 - (3*a0**2*a1*ny*np.sin(2*(ang0*k0 + phi0)))/k0 + (3*a0*a1**2*ny*np.sin(ang0*k0 - 3*ang0*k1 + phi0 - 3*phi1))/(k0 - 3*k1) - (3*a0**2*ny*x0*np.sin(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1))/(2*k0 - k1) + (3*a0**2*nx*y0*np.sin(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1))/(2*k0 - k1) - (3*a0**3*k0*ny*np.sin(3*ang0*k0 - ang0*k1 + 3*phi0 - phi1))/((3*k0 - k1)*(k0 + k1)) - (3*a0**3*k1*ny*np.sin(3*ang0*k0 - ang0*k1 + 3*phi0 - phi1))/((3*k0 - k1)*(k0 + k1)) - (3*a1**2*ny*x0*np.sin(ang0*k1 + phi1))/k1 + (3*a1**2*nx*y0*np.sin(ang0*k1 + phi1))/k1 + (3*a0**2*a1*ny*np.sin(2*(ang0*k1 + phi1)))/k1 + (a1**2*ny*x0*np.sin(3*(ang0*k1 + phi1)))/k1 - (a1**2*nx*y0*np.sin(3*(ang0*k1 + phi1)))/k1 - (3*a0*a1**2*ny*np.sin(ang0*(k0 + k1) + phi0 + phi1))/(k0 + k1) + (9*a0**3*k0*ny*np.sin(ang0*(k0 + k1) + phi0 + phi1))/((3*k0 - k1)*(k0 + k1)) - (3*a0**3*k1*ny*np.sin(ang0*(k0 + k1) + phi0 + phi1))/((3*k0 - k1)*(k0 + k1)) + (3*a0**2*ny*x0*np.sin(2*ang0*k0 + ang0*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (3*a0**2*nx*y0*np.sin(2*ang0*k0 + ang0*k1 + 2*phi0 + phi1))/(2*k0 + k1) + (6*a0*a1*ny*x0*np.sin(ang0*k0 + 2*ang0*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (6*a0*a1*nx*y0*np.sin(ang0*k0 + 2*ang0*k1 + phi0 + 2*phi1))/(k0 + 2*k1))/12.)
            res1 = (((-6*a0*a1*(nx*x0 + ny*y0)*np.cos(ang1*k0 + phi0))/k0 - (3*a0**2*(2*a1*nx + H*nz)*np.cos(2*(ang1*k0 + phi0)))/k0 + (3*a0*a1**2*nx*np.cos(ang1*k0 - 3*ang1*k1 + phi0 - 3*phi1))/(k0 - 3*k1) - (3*a0**2*nx*x0*np.cos(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1))/(2*k0 - k1) - (3*a0**2*ny*y0*np.cos(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1))/(2*k0 - k1) - (3*a0**3*k0*nx*np.cos(3*ang1*k0 - ang1*k1 + 3*phi0 - phi1))/((3*k0 - k1)*(k0 + k1)) - (3*a0**3*k1*nx*np.cos(3*ang1*k0 - ang1*k1 + 3*phi0 - phi1))/((3*k0 - k1)*(k0 + k1)) - (3*a1**2*nx*x0*np.cos(ang1*k1 + phi1))/k1 - (3*a1**2*ny*y0*np.cos(ang1*k1 + phi1))/k1 - (3*a0**2*a1*nx*np.cos(2*(ang1*k1 + phi1)))/k1 - (3*a1**3*nx*np.cos(2*(ang1*k1 + phi1)))/k1 - (3*a1**2*H*nz*np.cos(2*(ang1*k1 + phi1)))/k1 - (a1**2*nx*x0*np.cos(3*(ang1*k1 + phi1)))/k1 - (a1**2*ny*y0*np.cos(3*(ang1*k1 + phi1)))/k1 - (15*a0*a1**2*nx*np.cos(ang1*(k0 + k1) + phi0 + phi1))/(k0 + k1) - (9*a0**3*k0*nx*np.cos(ang1*(k0 + k1) + phi0 + phi1))/((3*k0 - k1)*(k0 + k1)) + (3*a0**3*k1*nx*np.cos(ang1*(k0 + k1) + phi0 + phi1))/((3*k0 - k1)*(k0 + k1)) - (12*a0*a1*H*nz*np.cos(ang1*(k0 + k1) + phi0 + phi1))/(k0 + k1) - (3*a0**2*nx*x0*np.cos(2*ang1*k0 + ang1*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (3*a0**2*ny*y0*np.cos(2*ang1*k0 + ang1*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (6*a0*a1*nx*x0*np.cos(ang1*k0 + 2*ang1*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (6*a0*a1*ny*y0*np.cos(ang1*k0 + 2*ang1*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (6*a0*a1*ny*x0*np.sin(ang1*k0 + phi0))/k0 + (6*a0*a1*nx*y0*np.sin(ang1*k0 + phi0))/k0 - (3*a0**2*a1*ny*np.sin(2*(ang1*k0 + phi0)))/k0 + (3*a0*a1**2*ny*np.sin(ang1*k0 - 3*ang1*k1 + phi0 - 3*phi1))/(k0 - 3*k1) - (3*a0**2*ny*x0*np.sin(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1))/(2*k0 - k1) + (3*a0**2*nx*y0*np.sin(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1))/(2*k0 - k1) - (3*a0**3*k0*ny*np.sin(3*ang1*k0 - ang1*k1 + 3*phi0 - phi1))/((3*k0 - k1)*(k0 + k1)) - (3*a0**3*k1*ny*np.sin(3*ang1*k0 - ang1*k1 + 3*phi0 - phi1))/((3*k0 - k1)*(k0 + k1)) - (3*a1**2*ny*x0*np.sin(ang1*k1 + phi1))/k1 + (3*a1**2*nx*y0*np.sin(ang1*k1 + phi1))/k1 + (3*a0**2*a1*ny*np.sin(2*(ang1*k1 + phi1)))/k1 + (a1**2*ny*x0*np.sin(3*(ang1*k1 + phi1)))/k1 - (a1**2*nx*y0*np.sin(3*(ang1*k1 + phi1)))/k1 - (3*a0*a1**2*ny*np.sin(ang1*(k0 + k1) + phi0 + phi1))/(k0 + k1) + (9*a0**3*k0*ny*np.sin(ang1*(k0 + k1) + phi0 + phi1))/((3*k0 - k1)*(k0 + k1)) - (3*a0**3*k1*ny*np.sin(ang1*(k0 + k1) + phi0 + phi1))/((3*k0 - k1)*(k0 + k1)) + (3*a0**2*ny*x0*np.sin(2*ang1*k0 + ang1*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (3*a0**2*nx*y0*np.sin(2*ang1*k0 + ang1*k1 + 2*phi0 + phi1))/(2*k0 + k1) + (6*a0*a1*ny*x0*np.sin(ang1*k0 + 2*ang1*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (6*a0*a1*nx*y0*np.sin(ang1*k0 + 2*ang1*k1 + phi0 + 2*phi1))/(k0 + 2*k1))/12.)
            return res1 - res0
        else:
            print('invalid index j = ' + str(j) + ' in Iij_c')
            return 0.0
    else:
        print('invalid index i = ' + str(i) + ' in Iij_c')
        return 0.0
    
@njit(cache=True)
def integrate(F, z, a0, a1): #F = F_matrix[p, q]
    res = 0
    for i in range(F.shape[0]):
        for j in range(F.shape[1]):
            if F[i, j] != 0:
                res += F[i,j]*Iij(i, j, z, a0, a1)
    return res  
    
@njit(cache=True)
def Iij(i, j, z_matrix, ang0, ang1):
    [a0, k0, phi0], [a1, k1, phi1] = z_matrix
    if k0 == 0 or k1 == 0:
        print('case of zero k0 or k1 has not implemented yet')
        return 0.0
    elif i == 0:
        if j == 0:
            return ang1-ang0
        elif j == 1:
            res1 = -a0/k0 * np.cos(k0*ang1 + phi0) - a1/k1 * np.cos(k1*ang1 + phi1)
            res0 = -a0/k0 * np.cos(k0*ang0 + phi0) - a1/k1 * np.cos(k1*ang0 + phi1)
            return res1 - res0
        elif j == 2:
            if k0 == k1:
                res0 = (-(-2*a0**2*ang0*k0 - 2*a1**2*ang0*k0 - 4*a0*a1*ang0*k0*np.cos(phi0 - phi1) + a0**2*np.sin(2*(ang0*k0 + phi0)) + a1**2*np.sin(2*(ang0*k0 + phi1)) + 2*a0*a1*np.sin(2*ang0*k0 + phi0 + phi1))/(4.*k0))
                res1 = (-(-2*a0**2*ang1*k0 - 2*a1**2*ang1*k0 - 4*a0*a1*ang1*k0*np.cos(phi0 - phi1) + a0**2*np.sin(2*(ang1*k0 + phi0)) + a1**2*np.sin(2*(ang1*k0 + phi1)) + 2*a0*a1*np.sin(2*ang1*k0 + phi0 + phi1))/(4.*k0))
                return res1 - res0
            else:
                res0 = ((a0**2*k1*(-k0**2 + k1**2)*np.sin(2*(ang0*k0 + phi0)) + k0*(4*a0*a1*k1*(k0 + k1)*np.sin(ang0*(k0 - k1) + phi0 - phi1) + (k0 - k1)*(-(a1**2*(k0 + k1)*np.sin(2*(ang0*k1 + phi1))) + 2*k1*((a0**2 + a1**2)*ang0*(k0 + k1) - 2*a0*a1*np.sin(ang0*(k0 + k1) + phi0 + phi1)))))/(4*k0*k1*(k0**2 - k1**2)))
                res1 = ((a0**2*k1*(-k0**2 + k1**2)*np.sin(2*(ang1*k0 + phi0)) + k0*(4*a0*a1*k1*(k0 + k1)*np.sin(ang1*(k0 - k1) + phi0 - phi1) + (k0 - k1)*(-(a1**2*(k0 + k1)*np.sin(2*(ang1*k1 + phi1))) + 2*k1*((a0**2 + a1**2)*ang1*(k0 + k1) - 2*a0*a1*np.sin(ang1*(k0 + k1) + phi0 + phi1)))))/(4*k0*k1*(k0**2 - k1**2)))
                return res1 - res0
        elif j == 3:
            if k0 == 2*k1:
                print('case of k0 == 2*k1 has not implemented yet')
                return 0.0
            elif k0 == -2*k1:
                print('case of k0 == -2*k1 has not implemented yet')
                return 0.0
            elif 2*k0 == -k1:
                print('case of 2*k0 == -k1 has not implemented yet')
                return 0.0
            elif 2*k0 == k1:
                res0 = ((-180*a0*(a0**2 + 2*a1**2)*np.cos(ang0*k0 + phi0) + 20*a0**3*np.cos(3*(ang0*k0 + phi0)) + a1*(-90*(2*a0**2 + a1**2)*np.cos(2*ang0*k0 + phi1) + 10*a1**2*np.cos(3*(2*ang0*k0 + phi1)) + 3*a0*(15*a0*np.cos(4*ang0*k0 + 2*phi0 + phi1) - 20*a1*np.cos(3*ang0*k0 - phi0 + 2*phi1) + 12*a1*np.cos(5*ang0*k0 + phi0 + 2*phi1) + 60*a0*ang0*k0*np.sin(2*phi0 - phi1))))/(240.*k0))
                res1 = ((-180*a0*(a0**2 + 2*a1**2)*np.cos(ang1*k0 + phi0) + 20*a0**3*np.cos(3*(ang1*k0 + phi0)) + a1*(-90*(2*a0**2 + a1**2)*np.cos(2*ang1*k0 + phi1) + 10*a1**2*np.cos(3*(2*ang1*k0 + phi1)) + 3*a0*(15*a0*np.cos(4*ang1*k0 + 2*phi0 + phi1) - 20*a1*np.cos(3*ang1*k0 - phi0 + 2*phi1) + 12*a1*np.cos(5*ang1*k0 + phi0 + 2*phi1) + 60*a0*ang1*k0*np.sin(2*phi0 - phi1))))/(240.*k0))
                return res1 - res0
            else:
                res0 = ((-9*a0*(a0**2 + 2*a1**2)*k1*(4*k0**4 - 17*k0**2*k1**2 + 4*k1**4)*np.cos(ang0*k0 + phi0) + a0**3*k1*(4*k0**4 - 17*k0**2*k1**2 + 4*k1**4)*np.cos(3*(ang0*k0 + phi0)) + a1*k0*(-9*a0*a1*k1*(-4*k0**3 - 8*k0**2*k1 + k0*k1**2 + 2*k1**3)*np.cos(ang0*(k0 - 2*k1) + phi0 - 2*phi1) - (k0 - 2*k1)*(9*a0**2*k1*(2*k0**2 + 5*k0*k1 + 2*k1**2)*np.cos(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1) + (2*k0 - k1)*(9*(2*a0**2 + a1**2)*(2*k0**2 + 5*k0*k1 + 2*k1**2)*np.cos(ang0*k1 + phi1) - a1**2*(2*k0**2 + 5*k0*k1 + 2*k1**2)*np.cos(3*(ang0*k1 + phi1)) - 9*a0*k1*(a0*(k0 + 2*k1)*np.cos(ang0*(2*k0 + k1) + 2*phi0 + phi1) + a1*(2*k0 + k1)*np.cos(ang0*(k0 + 2*k1) + phi0 + 2*phi1))))))/(12.*(4*k0**5*k1 - 17*k0**3*k1**3 + 4*k0*k1**5)))
                res1 = ((-9*a0*(a0**2 + 2*a1**2)*k1*(4*k0**4 - 17*k0**2*k1**2 + 4*k1**4)*np.cos(ang1*k0 + phi0) + a0**3*k1*(4*k0**4 - 17*k0**2*k1**2 + 4*k1**4)*np.cos(3*(ang1*k0 + phi0)) + a1*k0*(-9*a0*a1*k1*(-4*k0**3 - 8*k0**2*k1 + k0*k1**2 + 2*k1**3)*np.cos(ang1*(k0 - 2*k1) + phi0 - 2*phi1) - (k0 - 2*k1)*(9*a0**2*k1*(2*k0**2 + 5*k0*k1 + 2*k1**2)*np.cos(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1) + (2*k0 - k1)*(9*(2*a0**2 + a1**2)*(2*k0**2 + 5*k0*k1 + 2*k1**2)*np.cos(ang1*k1 + phi1) - a1**2*(2*k0**2 + 5*k0*k1 + 2*k1**2)*np.cos(3*(ang1*k1 + phi1)) - 9*a0*k1*(a0*(k0 + 2*k1)*np.cos(ang1*(2*k0 + k1) + 2*phi0 + phi1) + a1*(2*k0 + k1)*np.cos(ang1*(k0 + 2*k1) + phi0 + 2*phi1))))))/(12.*(4*k0**5*k1 - 17*k0**3*k1**3 + 4*k0*k1**5)))
                return res1 - res0
        elif j == 4:
            res0 = ((8*a0**2*(a0**2 + 3*a1**2)*k1*(-9*k0**6 + 91*k0**4*k1**2 - 91*k0**2*k1**4 + 9*k1**6)*np.sin(2*(ang0*k0 + phi0)) + a0**4*k1*(9*k0**6 - 91*k0**4*k1**2 + 91*k0**2*k1**4 - 9*k1**6)*np.sin(4*(ang0*k0 + phi0)) + k0*(-16*a0*a1**3*k1*(9*k0**5 + 27*k0**4*k1 - 10*k0**3*k1**2 - 30*k0**2*k1**3 + k0*k1**4 + 3*k1**5)*np.sin(ang0*(k0 - 3*k1) + phi0 - 3*phi1) + (k0 - 3*k1)*(48*a0*a1*(a0**2 + a1**2)*k1*(9*k0**4 + 36*k0**3*k1 + 26*k0**2*k1**2 - 4*k0*k1**3 - 3*k1**4)*np.sin(ang0*(k0 - k1) + phi0 - phi1) + 12*a0**2*a1**2*k1*(9*k0**4 + 36*k0**3*k1 + 26*k0**2*k1**2 - 4*k0*k1**3 - 3*k1**4)*np.sin(2*(ang0*(k0 - k1) + phi0 - phi1)) + (k0 - k1)*(-16*a0**3*a1*k1*(3*k0**3 + 13*k0**2*k1 + 13*k0*k1**2 + 3*k1**3)*np.sin(3*ang0*k0 - ang0*k1 + 3*phi0 - phi1) + (3*k0 - k1)*(-8*a1**2*(3*a0**2 + a1**2)*(3*k0**3 + 13*k0**2*k1 + 13*k0*k1**2 + 3*k1**3)*np.sin(2*(ang0*k1 + phi1)) + a1**4*(3*k0**3 + 13*k0**2*k1 + 13*k0*k1**2 + 3*k1**3)*np.sin(4*(ang0*k1 + phi1)) + 4*k1*(-12*a0*a1*(a0**2 + a1**2)*(3*k0**2 + 10*k0*k1 + 3*k1**2)*np.sin(ang0*(k0 + k1) + phi0 + phi1) + 3*a0**2*a1**2*(3*k0**2 + 10*k0*k1 + 3*k1**2)*np.sin(2*(ang0*(k0 + k1) + phi0 + phi1)) + (k0 + k1)*(4*a0**3*a1*(k0 + 3*k1)*np.sin(ang0*(3*k0 + k1) + 3*phi0 + phi1) + (3*k0 + k1)*(3*(a0**4 + 4*a0**2*a1**2 + a1**4)*ang0*(k0 + 3*k1) + 4*a0*a1**3*np.sin(ang0*(k0 + 3*k1) + phi0 + 3*phi1)))))))))/(32.*(9*k0**7*k1 - 91*k0**5*k1**3 + 91*k0**3*k1**5 - 9*k0*k1**7)))
            res1 = ((8*a0**2*(a0**2 + 3*a1**2)*k1*(-9*k0**6 + 91*k0**4*k1**2 - 91*k0**2*k1**4 + 9*k1**6)*np.sin(2*(ang1*k0 + phi0)) + a0**4*k1*(9*k0**6 - 91*k0**4*k1**2 + 91*k0**2*k1**4 - 9*k1**6)*np.sin(4*(ang1*k0 + phi0)) + k0*(-16*a0*a1**3*k1*(9*k0**5 + 27*k0**4*k1 - 10*k0**3*k1**2 - 30*k0**2*k1**3 + k0*k1**4 + 3*k1**5)*np.sin(ang1*(k0 - 3*k1) + phi0 - 3*phi1) + (k0 - 3*k1)*(48*a0*a1*(a0**2 + a1**2)*k1*(9*k0**4 + 36*k0**3*k1 + 26*k0**2*k1**2 - 4*k0*k1**3 - 3*k1**4)*np.sin(ang1*(k0 - k1) + phi0 - phi1) + 12*a0**2*a1**2*k1*(9*k0**4 + 36*k0**3*k1 + 26*k0**2*k1**2 - 4*k0*k1**3 - 3*k1**4)*np.sin(2*(ang1*(k0 - k1) + phi0 - phi1)) + (k0 - k1)*(-16*a0**3*a1*k1*(3*k0**3 + 13*k0**2*k1 + 13*k0*k1**2 + 3*k1**3)*np.sin(3*ang1*k0 - ang1*k1 + 3*phi0 - phi1) + (3*k0 - k1)*(-8*a1**2*(3*a0**2 + a1**2)*(3*k0**3 + 13*k0**2*k1 + 13*k0*k1**2 + 3*k1**3)*np.sin(2*(ang1*k1 + phi1)) + a1**4*(3*k0**3 + 13*k0**2*k1 + 13*k0*k1**2 + 3*k1**3)*np.sin(4*(ang1*k1 + phi1)) + 4*k1*(-12*a0*a1*(a0**2 + a1**2)*(3*k0**2 + 10*k0*k1 + 3*k1**2)*np.sin(ang1*(k0 + k1) + phi0 + phi1) + 3*a0**2*a1**2*(3*k0**2 + 10*k0*k1 + 3*k1**2)*np.sin(2*(ang1*(k0 + k1) + phi0 + phi1)) + (k0 + k1)*(4*a0**3*a1*(k0 + 3*k1)*np.sin(ang1*(3*k0 + k1) + 3*phi0 + phi1) + (3*k0 + k1)*(3*(a0**4 + 4*a0**2*a1**2 + a1**4)*ang1*(k0 + 3*k1) + 4*a0*a1**3*np.sin(ang1*(k0 + 3*k1) + phi0 + 3*phi1)))))))))/(32.*(9*k0**7*k1 - 91*k0**5*k1**3 + 91*k0**3*k1**5 - 9*k0*k1**7)))
            return res1 - res0
        elif j == 5:
            res0 = (((-150*a0*(a0**4 + 6*a0**2*a1**2 + 3*a1**4)*np.cos(ang0*k0 + phi0))/k0 + (25*(a0**5 + 4*a0**3*a1**2)*np.cos(3*(ang0*k0 + phi0)))/k0 - (3*a0**5*np.cos(5*(ang0*k0 + phi0)))/k0 - (75*a0*a1**4*np.cos(ang0*k0 - 4*ang0*k1 + phi0 - 4*phi1))/(k0 - 4*k1) + (150*a0**2*a1**3*np.cos(2*ang0*k0 - 3*ang0*k1 + 2*phi0 - 3*phi1))/(2*k0 - 3*k1) + (450*a0**3*a1**2*np.cos(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1))/(k0 - 2*k1) + (300*a0*a1**4*np.cos(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1))/(k0 - 2*k1) - (150*a0**3*a1**2*np.cos(3*ang0*k0 - 2*ang0*k1 + 3*phi0 - 2*phi1))/(3*k0 - 2*k1) - (300*a0**4*a1*np.cos(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1))/(2*k0 - k1) - (450*a0**2*a1**3*np.cos(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1))/(2*k0 - k1) + (75*a0**4*a1*np.cos(4*ang0*k0 - ang0*k1 + 4*phi0 - phi1))/(4*k0 - k1) - (450*a0**4*a1*np.cos(ang0*k1 + phi1))/k1 - (900*a0**2*a1**3*np.cos(ang0*k1 + phi1))/k1 - (150*a1**5*np.cos(ang0*k1 + phi1))/k1 + (100*a0**2*a1**3*np.cos(3*(ang0*k1 + phi1)))/k1 + (25*a1**5*np.cos(3*(ang0*k1 + phi1)))/k1 - (3*a1**5*np.cos(5*(ang0*k1 + phi1)))/k1 + (300*a0**4*a1*np.cos(2*ang0*k0 + ang0*k1 + 2*phi0 + phi1))/(2*k0 + k1) + (450*a0**2*a1**3*np.cos(2*ang0*k0 + ang0*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (75*a0**4*a1*np.cos(4*ang0*k0 + ang0*k1 + 4*phi0 + phi1))/(4*k0 + k1) + (450*a0**3*a1**2*np.cos(ang0*k0 + 2*ang0*k1 + phi0 + 2*phi1))/(k0 + 2*k1) + (300*a0*a1**4*np.cos(ang0*k0 + 2*ang0*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (150*a0**3*a1**2*np.cos(3*ang0*k0 + 2*ang0*k1 + 3*phi0 + 2*phi1))/(3*k0 + 2*k1) - (150*a0**2*a1**3*np.cos(2*ang0*k0 + 3*ang0*k1 + 2*phi0 + 3*phi1))/(2*k0 + 3*k1) - (75*a0*a1**4*np.cos(ang0*k0 + 4*ang0*k1 + phi0 + 4*phi1))/(k0 + 4*k1))/240.)
            res1 = (((-150*a0*(a0**4 + 6*a0**2*a1**2 + 3*a1**4)*np.cos(ang1*k0 + phi0))/k0 + (25*(a0**5 + 4*a0**3*a1**2)*np.cos(3*(ang1*k0 + phi0)))/k0 - (3*a0**5*np.cos(5*(ang1*k0 + phi0)))/k0 - (75*a0*a1**4*np.cos(ang1*k0 - 4*ang1*k1 + phi0 - 4*phi1))/(k0 - 4*k1) + (150*a0**2*a1**3*np.cos(2*ang1*k0 - 3*ang1*k1 + 2*phi0 - 3*phi1))/(2*k0 - 3*k1) + (450*a0**3*a1**2*np.cos(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1))/(k0 - 2*k1) + (300*a0*a1**4*np.cos(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1))/(k0 - 2*k1) - (150*a0**3*a1**2*np.cos(3*ang1*k0 - 2*ang1*k1 + 3*phi0 - 2*phi1))/(3*k0 - 2*k1) - (300*a0**4*a1*np.cos(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1))/(2*k0 - k1) - (450*a0**2*a1**3*np.cos(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1))/(2*k0 - k1) + (75*a0**4*a1*np.cos(4*ang1*k0 - ang1*k1 + 4*phi0 - phi1))/(4*k0 - k1) - (450*a0**4*a1*np.cos(ang1*k1 + phi1))/k1 - (900*a0**2*a1**3*np.cos(ang1*k1 + phi1))/k1 - (150*a1**5*np.cos(ang1*k1 + phi1))/k1 + (100*a0**2*a1**3*np.cos(3*(ang1*k1 + phi1)))/k1 + (25*a1**5*np.cos(3*(ang1*k1 + phi1)))/k1 - (3*a1**5*np.cos(5*(ang1*k1 + phi1)))/k1 + (300*a0**4*a1*np.cos(2*ang1*k0 + ang1*k1 + 2*phi0 + phi1))/(2*k0 + k1) + (450*a0**2*a1**3*np.cos(2*ang1*k0 + ang1*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (75*a0**4*a1*np.cos(4*ang1*k0 + ang1*k1 + 4*phi0 + phi1))/(4*k0 + k1) + (450*a0**3*a1**2*np.cos(ang1*k0 + 2*ang1*k1 + phi0 + 2*phi1))/(k0 + 2*k1) + (300*a0*a1**4*np.cos(ang1*k0 + 2*ang1*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (150*a0**3*a1**2*np.cos(3*ang1*k0 + 2*ang1*k1 + 3*phi0 + 2*phi1))/(3*k0 + 2*k1) - (150*a0**2*a1**3*np.cos(2*ang1*k0 + 3*ang1*k1 + 2*phi0 + 3*phi1))/(2*k0 + 3*k1) - (75*a0*a1**4*np.cos(ang1*k0 + 4*ang1*k1 + phi0 + 4*phi1))/(k0 + 4*k1))/240.)
            return res1 - res0
        else:
            print('invalid index j = ' + str(j) + ' in Iij')
            return 0.0
    elif i == 1:
        if j == 0:
            res1 = a0/k0 * np.sin(k0*ang1 + phi0) + a1/k1 * np.sin(k1*ang1 + phi1)
            res0 = a0/k0 * np.sin(k0*ang0 + phi0) + a1/k1 * np.sin(k1*ang0 + phi1)
            return res1 - res0
        elif j == 1:   
            res1_1 = -a0*a0/(4*k0) * np.cos(2*k0*ang1 + 2*phi0)
            res1_2 = -a1*a1/(4*k1) * np.cos(2*k1*ang1 + 2*phi1)
            res0_1 = -a0*a0/(4*k0) * np.cos(2*k0*ang0 + 2*phi0)
            res0_2 = -a1*a1/(4*k1) * np.cos(2*k1*ang0 + 2*phi1)
            res1_3 = 0
            res0_3 = 0
            if k1==-k0:
                print('case of k0 == -k1 has not implemented yet')
                return 0.0
            else:
                res1_4 = -a1*a0/(k0+k1) * np.cos((k0+k1)*ang1 + phi0+phi1)
                res0_4 = -a1*a0/(k0+k1) * np.cos((k0+k1)*ang0 + phi0+phi1)
            res1 = res1_1 + res1_2 + res1_3 + res1_4
            res0 = res0_1 + res0_2 + res0_3 + res0_4
            return res1  - res0
        elif j == 2:
            if k0 == 2*k1:
                print('case of k0 == 2*k1 has not implemented yet')
                return 0.0
            elif 2*k0 == k1:
                res0 = ((60*a0**2*a1*ang0*k0*np.cos(2*phi0 - phi1) + 60*a0*(a0**2 + 2*a1**2)*np.sin(ang0*k0 + phi0) - 20*a0**3*np.sin(3*(ang0*k0 + phi0)) + 60*a0**2*a1*np.sin(2*ang0*k0 + phi1) + 30*a1**3*np.sin(2*ang0*k0 + phi1) - 10*a1**3*np.sin(3*(2*ang0*k0 + phi1)) - 45*a0**2*a1*np.sin(4*ang0*k0 + 2*phi0 + phi1) + 20*a0*a1**2*np.sin(3*ang0*k0 - phi0 + 2*phi1) - 36*a0*a1**2*np.sin(5*ang0*k0 + phi0 + 2*phi1))/(240.*k0))
                res1 = ((60*a0**2*a1*ang1*k0*np.cos(2*phi0 - phi1) + 60*a0*(a0**2 + 2*a1**2)*np.sin(ang1*k0 + phi0) - 20*a0**3*np.sin(3*(ang1*k0 + phi0)) + 60*a0**2*a1*np.sin(2*ang1*k0 + phi1) + 30*a1**3*np.sin(2*ang1*k0 + phi1) - 10*a1**3*np.sin(3*(2*ang1*k0 + phi1)) - 45*a0**2*a1*np.sin(4*ang1*k0 + 2*phi0 + phi1) + 20*a0*a1**2*np.sin(3*ang1*k0 - phi0 + 2*phi1) - 36*a0*a1**2*np.sin(5*ang1*k0 + phi0 + 2*phi1))/(240.*k0))
                return res1 - res0
            elif 2*k0 == -k1:
                print('case of 2*k0 == -k1 has not implemented yet')
                return 0.0
            elif k0 == -2*k1:
                print('case of k0 == -2*k1 has not implemented yet')
                return 0.0
            else:
                res0 = ((3*a0*(a0**2 + 2*a1**2)*k1*(4*k0**4 - 17*k0**2*k1**2 + 4*k1**4)*np.sin(ang0*k0 + phi0) - a0**3*k1*(4*k0**4 - 17*k0**2*k1**2 + 4*k1**4)*np.sin(3*(ang0*k0 + phi0)) + a1*k0*(-3*a0*a1*k1*(-4*k0**3 - 8*k0**2*k1 + k0*k1**2 + 2*k1**3)*np.sin(ang0*(k0 - 2*k1) + phi0 - 2*phi1) + (k0 - 2*k1)*(3*a0**2*k1*(2*k0**2 + 5*k0*k1 + 2*k1**2)*np.sin(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1) + (2*k0 - k1)*(3*(2*a0**2 + a1**2)*(2*k0**2 + 5*k0*k1 + 2*k1**2)*np.sin(ang0*k1 + phi1) - a1**2*(2*k0**2 + 5*k0*k1 + 2*k1**2)*np.sin(3*(ang0*k1 + phi1)) - 9*a0*k1*(a0*(k0 + 2*k1)*np.sin(ang0*(2*k0 + k1) + 2*phi0 + phi1) + a1*(2*k0 + k1)*np.sin(ang0*(k0 + 2*k1) + phi0 + 2*phi1))))))/(12*(4*k0**5*k1 - 17*k0**3*k1**3 + 4*k0*k1**5)))
                res1 = ((3*a0*(a0**2 + 2*a1**2)*k1*(4*k0**4 - 17*k0**2*k1**2 + 4*k1**4)*np.sin(ang1*k0 + phi0) - a0**3*k1*(4*k0**4 - 17*k0**2*k1**2 + 4*k1**4)*np.sin(3*(ang1*k0 + phi0)) + a1*k0*(-3*a0*a1*k1*(-4*k0**3 - 8*k0**2*k1 + k0*k1**2 + 2*k1**3)*np.sin(ang1*(k0 - 2*k1) + phi0 - 2*phi1) + (k0 - 2*k1)*(3*a0**2*k1*(2*k0**2 + 5*k0*k1 + 2*k1**2)*np.sin(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1) + (2*k0 - k1)*(3*(2*a0**2 + a1**2)*(2*k0**2 + 5*k0*k1 + 2*k1**2)*np.sin(ang1*k1 + phi1) - a1**2*(2*k0**2 + 5*k0*k1 + 2*k1**2)*np.sin(3*(ang1*k1 + phi1)) - 9*a0*k1*(a0*(k0 + 2*k1)*np.sin(ang1*(2*k0 + k1) + 2*phi0 + phi1) + a1*(2*k0 + k1)*np.sin(ang1*(k0 + 2*k1) + phi0 + 2*phi1))))))/(12*(4*k0**5*k1 - 17*k0**3*k1**3 + 4*k0*k1**5)))
                return res1 - res0
        elif j == 3:
            if k0 == 3*k1:
                print('case of k0 == 3*k1 has not implemented yet')
                return 0.0
            elif 3*k0 == -k1:
                print('case of 3*k0 == -k1 has not implemented yet')
                return 0.0
            elif k0 == -3*k1:
                print('case of k0 == -3*k1 has not implemented yet')
                return 0.0
            elif k0 == -k1:
                print('case of k0 == -k1 has not implemented yet')
                return 0.0
            elif 3*k0 == k1:
                res0 = ((-60*(a0**4 + 3*a0**2*a1**2)*np.cos(2*(ang0*k0 + phi0)) + 15*a0**4*np.cos(4*(ang0*k0 + phi0)) + a1*(-20*a1*(3*a0**2 + a1**2)*np.cos(2*(3*ang0*k0 + phi1)) + 5*a1**3*np.cos(4*(3*ang0*k0 + phi1)) + a0*(-90*(a0**2 + a1**2)*np.cos(4*ang0*k0 + phi0 + phi1) + 45*a0*a1*np.cos(2*(4*ang0*k0 + phi0 + phi1)) + 40*a0**2*np.cos(6*ang0*k0 + 3*phi0 + phi1) - 15*a1**2*np.cos(8*ang0*k0 - phi0 + 3*phi1) + 24*a1**2*np.cos(10*ang0*k0 + phi0 + 3*phi1) + 120*a0**2*ang0*k0*np.sin(3*phi0 - phi1))))/(480.*k0))
                res1 = ((-60*(a0**4 + 3*a0**2*a1**2)*np.cos(2*(ang1*k0 + phi0)) + 15*a0**4*np.cos(4*(ang1*k0 + phi0)) + a1*(-20*a1*(3*a0**2 + a1**2)*np.cos(2*(3*ang1*k0 + phi1)) + 5*a1**3*np.cos(4*(3*ang1*k0 + phi1)) + a0*(-90*(a0**2 + a1**2)*np.cos(4*ang1*k0 + phi0 + phi1) + 45*a0*a1*np.cos(2*(4*ang1*k0 + phi0 + phi1)) + 40*a0**2*np.cos(6*ang1*k0 + 3*phi0 + phi1) - 15*a1**2*np.cos(8*ang1*k0 - phi0 + 3*phi1) + 24*a1**2*np.cos(10*ang1*k0 + phi0 + 3*phi1) + 120*a0**2*ang1*k0*np.sin(3*phi0 - phi1))))/(480.*k0))
                return res1 - res0
            else:
                res0 = ((-4*a0**2*(a0**2 + 3*a1**2)*k1*(9*k0**5 + 9*k0**4*k1 - 82*k0**3*k1**2 - 82*k0**2*k1**3 + 9*k0*k1**4 + 9*k1**5)*np.cos(2*(ang0*k0 + phi0)) + a0**4*k1*(9*k0**5 + 9*k0**4*k1 - 82*k0**3*k1**2 - 82*k0**2*k1**3 + 9*k0*k1**4 + 9*k1**5)*np.cos(4*(ang0*k0 + phi0)) + a1*k0*(8*a0*a1**2*k1*(9*k0**4 + 36*k0**3*k1 + 26*k0**2*k1**2 - 4*k0*k1**3 - 3*k1**4)*np.cos(ang0*(k0 - 3*k1) + phi0 - 3*phi1) - (k0 - 3*k1)*(8*a0**3*k1*(3*k0**3 + 13*k0**2*k1 + 13*k0*k1**2 + 3*k1**3)*np.cos(3*ang0*k0 - ang0*k1 + 3*phi0 - phi1) + (3*k0 - k1)*(4*a1*(3*a0**2 + a1**2)*(3*k0**3 + 13*k0**2*k1 + 13*k0*k1**2 + 3*k1**3)*np.cos(2*(ang0*k1 + phi1)) - a1**3*(3*k0**3 + 13*k0**2*k1 + 13*k0*k1**2 + 3*k1**3)*np.cos(4*(ang0*k1 + phi1)) - 4*a0*k1*(-6*(a0**2 + a1**2)*(3*k0**2 + 10*k0*k1 + 3*k1**2)*np.cos(ang0*(k0 + k1) + phi0 + phi1) + 3*a0*a1*(3*k0**2 + 10*k0*k1 + 3*k1**2)*np.cos(2*(ang0*(k0 + k1) + phi0 + phi1)) + 4*(k0 + k1)*(a0**2*(k0 + 3*k1)*np.cos(ang0*(3*k0 + k1) + 3*phi0 + phi1) + a1**2*(3*k0 + k1)*np.cos(ang0*(k0 + 3*k1) + phi0 + 3*phi1)))))))/(32.*k0*k1*(9*k0**5 + 9*k0**4*k1 - 82*k0**3*k1**2 - 82*k0**2*k1**3 + 9*k0*k1**4 + 9*k1**5)))
                res1 = ((-4*a0**2*(a0**2 + 3*a1**2)*k1*(9*k0**5 + 9*k0**4*k1 - 82*k0**3*k1**2 - 82*k0**2*k1**3 + 9*k0*k1**4 + 9*k1**5)*np.cos(2*(ang1*k0 + phi0)) + a0**4*k1*(9*k0**5 + 9*k0**4*k1 - 82*k0**3*k1**2 - 82*k0**2*k1**3 + 9*k0*k1**4 + 9*k1**5)*np.cos(4*(ang1*k0 + phi0)) + a1*k0*(8*a0*a1**2*k1*(9*k0**4 + 36*k0**3*k1 + 26*k0**2*k1**2 - 4*k0*k1**3 - 3*k1**4)*np.cos(ang1*(k0 - 3*k1) + phi0 - 3*phi1) - (k0 - 3*k1)*(8*a0**3*k1*(3*k0**3 + 13*k0**2*k1 + 13*k0*k1**2 + 3*k1**3)*np.cos(3*ang1*k0 - ang1*k1 + 3*phi0 - phi1) + (3*k0 - k1)*(4*a1*(3*a0**2 + a1**2)*(3*k0**3 + 13*k0**2*k1 + 13*k0*k1**2 + 3*k1**3)*np.cos(2*(ang1*k1 + phi1)) - a1**3*(3*k0**3 + 13*k0**2*k1 + 13*k0*k1**2 + 3*k1**3)*np.cos(4*(ang1*k1 + phi1)) - 4*a0*k1*(-6*(a0**2 + a1**2)*(3*k0**2 + 10*k0*k1 + 3*k1**2)*np.cos(ang1*(k0 + k1) + phi0 + phi1) + 3*a0*a1*(3*k0**2 + 10*k0*k1 + 3*k1**2)*np.cos(2*(ang1*(k0 + k1) + phi0 + phi1)) + 4*(k0 + k1)*(a0**2*(k0 + 3*k1)*np.cos(ang1*(3*k0 + k1) + 3*phi0 + phi1) + a1**2*(3*k0 + k1)*np.cos(ang1*(k0 + 3*k1) + phi0 + 3*phi1)))))))/(32.*k0*k1*(9*k0**5 + 9*k0**4*k1 - 82*k0**3*k1**2 - 82*k0**2*k1**3 + 9*k0*k1**4 + 9*k1**5)))
                return res1 - res0
        elif j == 4:
            res0 = (((10*a0*(a0**4 + 6*a0**2*a1**2 + 3*a1**4)*np.sin(ang0*k0 + phi0))/k0 - (5*(a0**5 + 4*a0**3*a1**2)*np.sin(3*(ang0*k0 + phi0)))/k0 + (a0**5*np.sin(5*(ang0*k0 + phi0)))/k0 - (15*a0*a1**4*np.sin(ang0*k0 - 4*ang0*k1 + phi0 - 4*phi1))/(k0 - 4*k1) + (10*a0**2*a1**3*np.sin(2*ang0*k0 - 3*ang0*k1 + 2*phi0 - 3*phi1))/(2*k0 - 3*k1) + (30*a0**3*a1**2*np.sin(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1))/(k0 - 2*k1) + (20*a0*a1**4*np.sin(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1))/(k0 - 2*k1) + (10*a0**3*a1**2*np.sin(3*ang0*k0 - 2*ang0*k1 + 3*phi0 - 2*phi1))/(3*k0 - 2*k1) + (20*a0**4*a1*np.sin(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1))/(2*k0 - k1) + (30*a0**2*a1**3*np.sin(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1))/(2*k0 - k1) - (15*a0**4*a1*np.sin(4*ang0*k0 - ang0*k1 + 4*phi0 - phi1))/(4*k0 - k1) + (30*a0**4*a1*np.sin(ang0*k1 + phi1))/k1 + (60*a0**2*a1**3*np.sin(ang0*k1 + phi1))/k1 + (10*a1**5*np.sin(ang0*k1 + phi1))/k1 - (20*a0**2*a1**3*np.sin(3*(ang0*k1 + phi1)))/k1 - (5*a1**5*np.sin(3*(ang0*k1 + phi1)))/k1 + (a1**5*np.sin(5*(ang0*k1 + phi1)))/k1 - (60*a0**4*a1*np.sin(2*ang0*k0 + ang0*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (90*a0**2*a1**3*np.sin(2*ang0*k0 + ang0*k1 + 2*phi0 + phi1))/(2*k0 + k1) + (25*a0**4*a1*np.sin(4*ang0*k0 + ang0*k1 + 4*phi0 + phi1))/(4*k0 + k1) - (90*a0**3*a1**2*np.sin(ang0*k0 + 2*ang0*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (60*a0*a1**4*np.sin(ang0*k0 + 2*ang0*k1 + phi0 + 2*phi1))/(k0 + 2*k1) + (50*a0**3*a1**2*np.sin(3*ang0*k0 + 2*ang0*k1 + 3*phi0 + 2*phi1))/(3*k0 + 2*k1) + (50*a0**2*a1**3*np.sin(2*ang0*k0 + 3*ang0*k1 + 2*phi0 + 3*phi1))/(2*k0 + 3*k1) + (25*a0*a1**4*np.sin(ang0*k0 + 4*ang0*k1 + phi0 + 4*phi1))/(k0 + 4*k1))/80.)
            res1 = (((10*a0*(a0**4 + 6*a0**2*a1**2 + 3*a1**4)*np.sin(ang1*k0 + phi0))/k0 - (5*(a0**5 + 4*a0**3*a1**2)*np.sin(3*(ang1*k0 + phi0)))/k0 + (a0**5*np.sin(5*(ang1*k0 + phi0)))/k0 - (15*a0*a1**4*np.sin(ang1*k0 - 4*ang1*k1 + phi0 - 4*phi1))/(k0 - 4*k1) + (10*a0**2*a1**3*np.sin(2*ang1*k0 - 3*ang1*k1 + 2*phi0 - 3*phi1))/(2*k0 - 3*k1) + (30*a0**3*a1**2*np.sin(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1))/(k0 - 2*k1) + (20*a0*a1**4*np.sin(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1))/(k0 - 2*k1) + (10*a0**3*a1**2*np.sin(3*ang1*k0 - 2*ang1*k1 + 3*phi0 - 2*phi1))/(3*k0 - 2*k1) + (20*a0**4*a1*np.sin(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1))/(2*k0 - k1) + (30*a0**2*a1**3*np.sin(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1))/(2*k0 - k1) - (15*a0**4*a1*np.sin(4*ang1*k0 - ang1*k1 + 4*phi0 - phi1))/(4*k0 - k1) + (30*a0**4*a1*np.sin(ang1*k1 + phi1))/k1 + (60*a0**2*a1**3*np.sin(ang1*k1 + phi1))/k1 + (10*a1**5*np.sin(ang1*k1 + phi1))/k1 - (20*a0**2*a1**3*np.sin(3*(ang1*k1 + phi1)))/k1 - (5*a1**5*np.sin(3*(ang1*k1 + phi1)))/k1 + (a1**5*np.sin(5*(ang1*k1 + phi1)))/k1 - (60*a0**4*a1*np.sin(2*ang1*k0 + ang1*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (90*a0**2*a1**3*np.sin(2*ang1*k0 + ang1*k1 + 2*phi0 + phi1))/(2*k0 + k1) + (25*a0**4*a1*np.sin(4*ang1*k0 + ang1*k1 + 4*phi0 + phi1))/(4*k0 + k1) - (90*a0**3*a1**2*np.sin(ang1*k0 + 2*ang1*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (60*a0*a1**4*np.sin(ang1*k0 + 2*ang1*k1 + phi0 + 2*phi1))/(k0 + 2*k1) + (50*a0**3*a1**2*np.sin(3*ang1*k0 + 2*ang1*k1 + 3*phi0 + 2*phi1))/(3*k0 + 2*k1) + (50*a0**2*a1**3*np.sin(2*ang1*k0 + 3*ang1*k1 + 2*phi0 + 3*phi1))/(2*k0 + 3*k1) + (25*a0*a1**4*np.sin(ang1*k0 + 4*ang1*k1 + phi0 + 4*phi1))/(k0 + 4*k1))/80.)
            return res1 - res0
        elif j == 5:
            res0 = (((-15*(a0**6 + 8*a0**4*a1**2 + 6*a0**2*a1**4)*np.cos(2*(ang0*k0 + phi0)))/k0 + (6*(a0**6 + 5*a0**4*a1**2)*np.cos(4*(ang0*k0 + phi0)))/k0 - (a0**6*np.cos(6*(ang0*k0 + phi0)))/k0 - (24*a0*a1**5*np.cos(ang0*k0 - 5*ang0*k1 + phi0 - 5*phi1))/(k0 - 5*k1) + (120*a0**3*a1**3*np.cos(ang0*k0 - 3*ang0*k1 + phi0 - 3*phi1))/(k0 - 3*k1) + (60*a0*a1**5*np.cos(ang0*k0 - 3*ang0*k1 + phi0 - 3*phi1))/(k0 - 3*k1) + (15*a0**2*a1**4*np.cos(2*(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1)))/(k0 - 2*k1) - (15*a0**4*a1**2*np.cos(4*ang0*k0 - 2*ang0*k1 + 4*phi0 - 2*phi1))/(2*k0 - k1) - (60*a0**5*a1*np.cos(3*ang0*k0 - ang0*k1 + 3*phi0 - phi1))/(3*k0 - k1) - (120*a0**3*a1**3*np.cos(3*ang0*k0 - ang0*k1 + 3*phi0 - phi1))/(3*k0 - k1) + (24*a0**5*a1*np.cos(5*ang0*k0 - ang0*k1 + 5*phi0 - phi1))/(5*k0 - k1) - (90*a0**4*a1**2*np.cos(2*(ang0*k1 + phi1)))/k1 - (120*a0**2*a1**4*np.cos(2*(ang0*k1 + phi1)))/k1 - (15*a1**6*np.cos(2*(ang0*k1 + phi1)))/k1 + (30*a0**2*a1**4*np.cos(4*(ang0*k1 + phi1)))/k1 + (6*a1**6*np.cos(4*(ang0*k1 + phi1)))/k1 - (a1**6*np.cos(6*(ang0*k1 + phi1)))/k1 - (120*a0**5*a1*np.cos(ang0*(k0 + k1) + phi0 + phi1))/(k0 + k1) - (360*a0**3*a1**3*np.cos(ang0*(k0 + k1) + phi0 + phi1))/(k0 + k1) - (120*a0*a1**5*np.cos(ang0*(k0 + k1) + phi0 + phi1))/(k0 + k1) + (120*a0**4*a1**2*np.cos(2*(ang0*(k0 + k1) + phi0 + phi1)))/(k0 + k1) + (120*a0**2*a1**4*np.cos(2*(ang0*(k0 + k1) + phi0 + phi1)))/(k0 + k1) - (40*a0**3*a1**3*np.cos(3*(ang0*(k0 + k1) + phi0 + phi1)))/(k0 + k1) - (45*a0**4*a1**2*np.cos(2*(2*ang0*k0 + ang0*k1 + 2*phi0 + phi1)))/(2*k0 + k1) + (120*a0**5*a1*np.cos(3*ang0*k0 + ang0*k1 + 3*phi0 + phi1))/(3*k0 + k1) + (240*a0**3*a1**3*np.cos(3*ang0*k0 + ang0*k1 + 3*phi0 + phi1))/(3*k0 + k1) - (36*a0**5*a1*np.cos(5*ang0*k0 + ang0*k1 + 5*phi0 + phi1))/(5*k0 + k1) - (45*a0**2*a1**4*np.cos(2*(ang0*k0 + 2*ang0*k1 + phi0 + 2*phi1)))/(k0 + 2*k1) + (240*a0**3*a1**3*np.cos(ang0*k0 + 3*ang0*k1 + phi0 + 3*phi1))/(k0 + 3*k1) + (120*a0*a1**5*np.cos(ang0*k0 + 3*ang0*k1 + phi0 + 3*phi1))/(k0 + 3*k1) - (36*a0*a1**5*np.cos(ang0*k0 + 5*ang0*k1 + phi0 + 5*phi1))/(k0 + 5*k1))/192.)
            res1 = (((-15*(a0**6 + 8*a0**4*a1**2 + 6*a0**2*a1**4)*np.cos(2*(ang1*k0 + phi0)))/k0 + (6*(a0**6 + 5*a0**4*a1**2)*np.cos(4*(ang1*k0 + phi0)))/k0 - (a0**6*np.cos(6*(ang1*k0 + phi0)))/k0 - (24*a0*a1**5*np.cos(ang1*k0 - 5*ang1*k1 + phi0 - 5*phi1))/(k0 - 5*k1) + (120*a0**3*a1**3*np.cos(ang1*k0 - 3*ang1*k1 + phi0 - 3*phi1))/(k0 - 3*k1) + (60*a0*a1**5*np.cos(ang1*k0 - 3*ang1*k1 + phi0 - 3*phi1))/(k0 - 3*k1) + (15*a0**2*a1**4*np.cos(2*(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1)))/(k0 - 2*k1) - (15*a0**4*a1**2*np.cos(4*ang1*k0 - 2*ang1*k1 + 4*phi0 - 2*phi1))/(2*k0 - k1) - (60*a0**5*a1*np.cos(3*ang1*k0 - ang1*k1 + 3*phi0 - phi1))/(3*k0 - k1) - (120*a0**3*a1**3*np.cos(3*ang1*k0 - ang1*k1 + 3*phi0 - phi1))/(3*k0 - k1) + (24*a0**5*a1*np.cos(5*ang1*k0 - ang1*k1 + 5*phi0 - phi1))/(5*k0 - k1) - (90*a0**4*a1**2*np.cos(2*(ang1*k1 + phi1)))/k1 - (120*a0**2*a1**4*np.cos(2*(ang1*k1 + phi1)))/k1 - (15*a1**6*np.cos(2*(ang1*k1 + phi1)))/k1 + (30*a0**2*a1**4*np.cos(4*(ang1*k1 + phi1)))/k1 + (6*a1**6*np.cos(4*(ang1*k1 + phi1)))/k1 - (a1**6*np.cos(6*(ang1*k1 + phi1)))/k1 - (120*a0**5*a1*np.cos(ang1*(k0 + k1) + phi0 + phi1))/(k0 + k1) - (360*a0**3*a1**3*np.cos(ang1*(k0 + k1) + phi0 + phi1))/(k0 + k1) - (120*a0*a1**5*np.cos(ang1*(k0 + k1) + phi0 + phi1))/(k0 + k1) + (120*a0**4*a1**2*np.cos(2*(ang1*(k0 + k1) + phi0 + phi1)))/(k0 + k1) + (120*a0**2*a1**4*np.cos(2*(ang1*(k0 + k1) + phi0 + phi1)))/(k0 + k1) - (40*a0**3*a1**3*np.cos(3*(ang1*(k0 + k1) + phi0 + phi1)))/(k0 + k1) - (45*a0**4*a1**2*np.cos(2*(2*ang1*k0 + ang1*k1 + 2*phi0 + phi1)))/(2*k0 + k1) + (120*a0**5*a1*np.cos(3*ang1*k0 + ang1*k1 + 3*phi0 + phi1))/(3*k0 + k1) + (240*a0**3*a1**3*np.cos(3*ang1*k0 + ang1*k1 + 3*phi0 + phi1))/(3*k0 + k1) - (36*a0**5*a1*np.cos(5*ang1*k0 + ang1*k1 + 5*phi0 + phi1))/(5*k0 + k1) - (45*a0**2*a1**4*np.cos(2*(ang1*k0 + 2*ang1*k1 + phi0 + 2*phi1)))/(k0 + 2*k1) + (240*a0**3*a1**3*np.cos(ang1*k0 + 3*ang1*k1 + phi0 + 3*phi1))/(k0 + 3*k1) + (120*a0*a1**5*np.cos(ang1*k0 + 3*ang1*k1 + phi0 + 3*phi1))/(k0 + 3*k1) - (36*a0*a1**5*np.cos(ang1*k0 + 5*ang1*k1 + phi0 + 5*phi1))/(k0 + 5*k1))/192.)
            return res1 - res0
        else:
            print('invalid index j = ' + str(j) + ' in Iij')
            return 0.0
    elif i == 2:
        if j == 0:
            if k0 == k1:
                res0 = ((2*a0**2*ang0*k0 + 2*a1**2*ang0*k0 + 4*a0*a1*ang0*k0*np.cos(phi0 - phi1) + a0**2*np.sin(2*(ang0*k0 + phi0)) + a1**2*np.sin(2*(ang0*k0 + phi1)) + 2*a0*a1*np.sin(2*ang0*k0 + phi0 + phi1))/(4.*k0))
                res1 = ((2*a0**2*ang1*k0 + 2*a1**2*ang1*k0 + 4*a0*a1*ang1*k0*np.cos(phi0 - phi1) + a0**2*np.sin(2*(ang1*k0 + phi0)) + a1**2*np.sin(2*(ang1*k0 + phi1)) + 2*a0*a1*np.sin(2*ang1*k0 + phi0 + phi1))/(4.*k0))
                return res1 - res0
            elif k0 == -k1:
                print('case of k0 == -k1 has not implemented yet')
                return 0.0
            else:
                res0 = ((a0**2*k1*(k0**2 - k1**2)*np.sin(2*(ang0*k0 + phi0)) + k0*(4*a0*a1*k1*(k0 + k1)*np.sin(ang0*(k0 - k1) + phi0 - phi1) + (k0 - k1)*(a1**2*(k0 + k1)*np.sin(2*(ang0*k1 + phi1)) + 2*k1*((a0**2 + a1**2)*ang0*(k0 + k1) + 2*a0*a1*np.sin(ang0*(k0 + k1) + phi0 + phi1)))))/(4*k0*k1*(k0**2 - k1**2)))
                res1 = ((a0**2*k1*(k0**2 - k1**2)*np.sin(2*(ang1*k0 + phi0)) + k0*(4*a0*a1*k1*(k0 + k1)*np.sin(ang1*(k0 - k1) + phi0 - phi1) + (k0 - k1)*(a1**2*(k0 + k1)*np.sin(2*(ang1*k1 + phi1)) + 2*k1*((a0**2 + a1**2)*ang1*(k0 + k1) + 2*a0*a1*np.sin(ang1*(k0 + k1) + phi0 + phi1)))))/(4*k0*k1*(k0**2 - k1**2)))
                return res1 - res0
        elif j == 1:
            if k0 == 2*k1:
                print('case of k0 == 2*k1 has not implemented yet')
                return 0.0
            elif k0 == -2*k1:
                print('case of k0 == -2*k1 has not implemented yet')
                return 0.0
            elif 2*k0 == -k1:
                print('case of 2*k0 == -k1 has not implemented yet')
                return 0.0
            elif 2*k0 == k1:
                res0 = (-(60*a0*(a0**2 + 2*a1**2)*np.cos(ang0*k0 + phi0) + 20*a0**3*np.cos(3*(ang0*k0 + phi0)) + a1*(30*(2*a0**2 + a1**2)*np.cos(2*ang0*k0 + phi1) + 10*a1**2*np.cos(3*(2*ang0*k0 + phi1)) + a0*(45*a0*np.cos(4*ang0*k0 + 2*phi0 + phi1) + 20*a1*np.cos(3*ang0*k0 - phi0 + 2*phi1) + 36*a1*np.cos(5*ang0*k0 + phi0 + 2*phi1) - 60*a0*ang0*k0*np.sin(2*phi0 - phi1))))/(240.*k0))
                res1 = (-(60*a0*(a0**2 + 2*a1**2)*np.cos(ang1*k0 + phi0) + 20*a0**3*np.cos(3*(ang1*k0 + phi0)) + a1*(30*(2*a0**2 + a1**2)*np.cos(2*ang1*k0 + phi1) + 10*a1**2*np.cos(3*(2*ang1*k0 + phi1)) + a0*(45*a0*np.cos(4*ang1*k0 + 2*phi0 + phi1) + 20*a1*np.cos(3*ang1*k0 - phi0 + 2*phi1) + 36*a1*np.cos(5*ang1*k0 + phi0 + 2*phi1) - 60*a0*ang1*k0*np.sin(2*phi0 - phi1))))/(240.*k0))
                return res1 - res0
            else:
                res0 = (-(3*a0*(a0**2 + 2*a1**2)*k1*(4*k0**4 - 17*k0**2*k1**2 + 4*k1**4)*np.cos(ang0*k0 + phi0) + a0**3*k1*(4*k0**4 - 17*k0**2*k1**2 + 4*k1**4)*np.cos(3*(ang0*k0 + phi0)) + a1*k0*(3*a0*a1*k1*(-4*k0**3 - 8*k0**2*k1 + k0*k1**2 + 2*k1**3)*np.cos(ang0*(k0 - 2*k1) + phi0 - 2*phi1) + (k0 - 2*k1)*(3*a0**2*k1*(2*k0**2 + 5*k0*k1 + 2*k1**2)*np.cos(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1) + (2*k0 - k1)*(3*(2*a0**2 + a1**2)*(2*k0**2 + 5*k0*k1 + 2*k1**2)*np.cos(ang0*k1 + phi1) + a1**2*(2*k0**2 + 5*k0*k1 + 2*k1**2)*np.cos(3*(ang0*k1 + phi1)) + 9*a0*k1*(a0*(k0 + 2*k1)*np.cos(ang0*(2*k0 + k1) + 2*phi0 + phi1) + a1*(2*k0 + k1)*np.cos(ang0*(k0 + 2*k1) + phi0 + 2*phi1))))))/(12.*(4*k0**5*k1 - 17*k0**3*k1**3 + 4*k0*k1**5)))
                res1 = (-(3*a0*(a0**2 + 2*a1**2)*k1*(4*k0**4 - 17*k0**2*k1**2 + 4*k1**4)*np.cos(ang1*k0 + phi0) + a0**3*k1*(4*k0**4 - 17*k0**2*k1**2 + 4*k1**4)*np.cos(3*(ang1*k0 + phi0)) + a1*k0*(3*a0*a1*k1*(-4*k0**3 - 8*k0**2*k1 + k0*k1**2 + 2*k1**3)*np.cos(ang1*(k0 - 2*k1) + phi0 - 2*phi1) + (k0 - 2*k1)*(3*a0**2*k1*(2*k0**2 + 5*k0*k1 + 2*k1**2)*np.cos(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1) + (2*k0 - k1)*(3*(2*a0**2 + a1**2)*(2*k0**2 + 5*k0*k1 + 2*k1**2)*np.cos(ang1*k1 + phi1) + a1**2*(2*k0**2 + 5*k0*k1 + 2*k1**2)*np.cos(3*(ang1*k1 + phi1)) + 9*a0*k1*(a0*(k0 + 2*k1)*np.cos(ang1*(2*k0 + k1) + 2*phi0 + phi1) + a1*(2*k0 + k1)*np.cos(ang1*(k0 + 2*k1) + phi0 + 2*phi1))))))/(12.*(4*k0**5*k1 - 17*k0**3*k1**3 + 4*k0*k1**5)))
                return res1 - res0
        elif j == 2:
            if k0 == -3*k1:
                print('case of k0 == -3*k1 has not implemented yet')
                return 0.0
            elif 3*k0 == -k1:
                print('case of 3*k0 == -k1 has not implemented yet')
                return 0.0
            elif k0 == -k1:
                print('case of k0 == -k1 has not implemented yet')
                return 0.0
            elif k0 == k1:
                res0 = (-(-4*a0**4*ang0*k0 - 16*a0**2*a1**2*ang0*k0 - 4*a1**4*ang0*k0 - 16*a0*a1*(a0**2 + a1**2)*ang0*k0*np.cos(phi0 - phi1) - 8*a0**2*a1**2*ang0*k0*np.cos(2*(phi0 - phi1)) + a0**4*np.sin(4*(ang0*k0 + phi0)) + a1**4*np.sin(4*(ang0*k0 + phi1)) + 6*a0**2*a1**2*np.sin(2*(2*ang0*k0 + phi0 + phi1)) + 4*a0**3*a1*np.sin(4*ang0*k0 + 3*phi0 + phi1) + 4*a0*a1**3*np.sin(4*ang0*k0 + phi0 + 3*phi1))/(32.*k0))
                res1 = (-(-4*a0**4*ang1*k0 - 16*a0**2*a1**2*ang1*k0 - 4*a1**4*ang1*k0 - 16*a0*a1*(a0**2 + a1**2)*ang1*k0*np.cos(phi0 - phi1) - 8*a0**2*a1**2*ang1*k0*np.cos(2*(phi0 - phi1)) + a0**4*np.sin(4*(ang1*k0 + phi0)) + a1**4*np.sin(4*(ang1*k0 + phi1)) + 6*a0**2*a1**2*np.sin(2*(2*ang1*k0 + phi0 + phi1)) + 4*a0**3*a1*np.sin(4*ang1*k0 + 3*phi0 + phi1) + 4*a0*a1**3*np.sin(4*ang1*k0 + phi0 + 3*phi1))/(32.*k0))
                return res1 - res0
            else:
                res0 = ((a0**4*k1*(-3*k0**4 - 10*k0**3*k1 + 10*k0*k1**3 + 3*k1**4)*np.sin(4*(ang0*k0 + phi0)) + k0*(12*a0**4*ang0*k0**4*k1 + 48*a0**2*a1**2*ang0*k0**4*k1 + 12*a1**4*ang0*k0**4*k1 + 40*a0**4*ang0*k0**3*k1**2 + 160*a0**2*a1**2*ang0*k0**3*k1**2 + 40*a1**4*ang0*k0**3*k1**2 - 40*a0**4*ang0*k0*k1**4 - 160*a0**2*a1**2*ang0*k0*k1**4 - 40*a1**4*ang0*k0*k1**4 - 12*a0**4*ang0*k1**5 - 48*a0**2*a1**2*ang0*k1**5 - 12*a1**4*ang0*k1**5 + 16*a0**3*a1*k1*(3*k0**3 + 13*k0**2*k1 + 13*k0*k1**2 + 3*k1**3)*np.sin(ang0*(k0 - k1) + phi0 - phi1) + 4*a0**2*a1**2*k1*(3*k0**3 + 13*k0**2*k1 + 13*k0*k1**2 + 3*k1**3)*np.sin(2*(ang0*k0 - ang0*k1 + phi0 - phi1)) - 3*a1**4*k0**4*np.sin(4*(ang0*k1 + phi1)) - 10*a1**4*k0**3*k1*np.sin(4*(ang0*k1 + phi1)) + 10*a1**4*k0*k1**3*np.sin(4*(ang0*k1 + phi1)) + 3*a1**4*k1**4*np.sin(4*(ang0*k1 + phi1)) - 48*a0*a1**3*k0**3*k1*np.sin(ang0*(-k0 + k1) - phi0 + phi1) - 208*a0*a1**3*k0**2*k1**2*np.sin(ang0*(-k0 + k1) - phi0 + phi1) - 208*a0*a1**3*k0*k1**3*np.sin(ang0*(-k0 + k1) - phi0 + phi1) - 48*a0*a1**3*k1**4*np.sin(ang0*(-k0 + k1) - phi0 + phi1) - 36*a0**2*a1**2*k0**3*k1*np.sin(2*(ang0*(k0 + k1) + phi0 + phi1)) - 84*a0**2*a1**2*k0**2*k1**2*np.sin(2*(ang0*(k0 + k1) + phi0 + phi1)) + 84*a0**2*a1**2*k0*k1**3*np.sin(2*(ang0*(k0 + k1) + phi0 + phi1)) + 36*a0**2*a1**2*k1**4*np.sin(2*(ang0*(k0 + k1) + phi0 + phi1)) - 16*a0**3*a1*k0**3*k1*np.sin(3*ang0*k0 + ang0*k1 + 3*phi0 + phi1) - 48*a0**3*a1*k0**2*k1**2*np.sin(3*ang0*k0 + ang0*k1 + 3*phi0 + phi1) + 16*a0**3*a1*k0*k1**3*np.sin(3*ang0*k0 + ang0*k1 + 3*phi0 + phi1) + 48*a0**3*a1*k1**4*np.sin(3*ang0*k0 + ang0*k1 + 3*phi0 + phi1) - 48*a0*a1**3*k0**3*k1*np.sin(ang0*k0 + 3*ang0*k1 + phi0 + 3*phi1) - 16*a0*a1**3*k0**2*k1**2*np.sin(ang0*k0 + 3*ang0*k1 + phi0 + 3*phi1) + 48*a0*a1**3*k0*k1**3*np.sin(ang0*k0 + 3*ang0*k1 + phi0 + 3*phi1) + 16*a0*a1**3*k1**4*np.sin(ang0*k0 + 3*ang0*k1 + phi0 + 3*phi1)))/(32.*k0*(k0 - k1)*k1*(k0 + k1)*(3*k0 + k1)*(k0 + 3*k1)))
                res1 = ((a0**4*k1*(-3*k0**4 - 10*k0**3*k1 + 10*k0*k1**3 + 3*k1**4)*np.sin(4*(ang1*k0 + phi0)) + k0*(12*a0**4*ang1*k0**4*k1 + 48*a0**2*a1**2*ang1*k0**4*k1 + 12*a1**4*ang1*k0**4*k1 + 40*a0**4*ang1*k0**3*k1**2 + 160*a0**2*a1**2*ang1*k0**3*k1**2 + 40*a1**4*ang1*k0**3*k1**2 - 40*a0**4*ang1*k0*k1**4 - 160*a0**2*a1**2*ang1*k0*k1**4 - 40*a1**4*ang1*k0*k1**4 - 12*a0**4*ang1*k1**5 - 48*a0**2*a1**2*ang1*k1**5 - 12*a1**4*ang1*k1**5 + 16*a0**3*a1*k1*(3*k0**3 + 13*k0**2*k1 + 13*k0*k1**2 + 3*k1**3)*np.sin(ang1*(k0 - k1) + phi0 - phi1) + 4*a0**2*a1**2*k1*(3*k0**3 + 13*k0**2*k1 + 13*k0*k1**2 + 3*k1**3)*np.sin(2*(ang1*k0 - ang1*k1 + phi0 - phi1)) - 3*a1**4*k0**4*np.sin(4*(ang1*k1 + phi1)) - 10*a1**4*k0**3*k1*np.sin(4*(ang1*k1 + phi1)) + 10*a1**4*k0*k1**3*np.sin(4*(ang1*k1 + phi1)) + 3*a1**4*k1**4*np.sin(4*(ang1*k1 + phi1)) - 48*a0*a1**3*k0**3*k1*np.sin(ang1*(-k0 + k1) - phi0 + phi1) - 208*a0*a1**3*k0**2*k1**2*np.sin(ang1*(-k0 + k1) - phi0 + phi1) - 208*a0*a1**3*k0*k1**3*np.sin(ang1*(-k0 + k1) - phi0 + phi1) - 48*a0*a1**3*k1**4*np.sin(ang1*(-k0 + k1) - phi0 + phi1) - 36*a0**2*a1**2*k0**3*k1*np.sin(2*(ang1*(k0 + k1) + phi0 + phi1)) - 84*a0**2*a1**2*k0**2*k1**2*np.sin(2*(ang1*(k0 + k1) + phi0 + phi1)) + 84*a0**2*a1**2*k0*k1**3*np.sin(2*(ang1*(k0 + k1) + phi0 + phi1)) + 36*a0**2*a1**2*k1**4*np.sin(2*(ang1*(k0 + k1) + phi0 + phi1)) - 16*a0**3*a1*k0**3*k1*np.sin(3*ang1*k0 + ang1*k1 + 3*phi0 + phi1) - 48*a0**3*a1*k0**2*k1**2*np.sin(3*ang1*k0 + ang1*k1 + 3*phi0 + phi1) + 16*a0**3*a1*k0*k1**3*np.sin(3*ang1*k0 + ang1*k1 + 3*phi0 + phi1) + 48*a0**3*a1*k1**4*np.sin(3*ang1*k0 + ang1*k1 + 3*phi0 + phi1) - 48*a0*a1**3*k0**3*k1*np.sin(ang1*k0 + 3*ang1*k1 + phi0 + 3*phi1) - 16*a0*a1**3*k0**2*k1**2*np.sin(ang1*k0 + 3*ang1*k1 + phi0 + 3*phi1) + 48*a0*a1**3*k0*k1**3*np.sin(ang1*k0 + 3*ang1*k1 + phi0 + 3*phi1) + 16*a0*a1**3*k1**4*np.sin(ang1*k0 + 3*ang1*k1 + phi0 + 3*phi1)))/(32.*k0*(k0 - k1)*k1*(k0 + k1)*(3*k0 + k1)*(k0 + 3*k1)))
                return res1 - res0
        elif j == 3:
            if k0 == 4*k1:
                print('case of k0 == 4*k1 has not implemented yet')
                return 0.0
            elif 2*k0 == 3*k1:
                print('case of 2*k0 == 3*k1 has not implemented yet')
                return 0.0
            elif k0 == 2*k1:
                print('case of k0 == 2*k1 has not implemented yet')
                return 0.0
            elif 2*k0 == -k1:
                print('case of 2*k0 == -k1 has not implemented yet')
                return 0.0
            elif 4*k0 == -k1:
                print('case of 4*k0 == -k1 has not implemented yet')
                return 0.0
            elif k0 == -2*k1:
                print('case of k0 == -2*k1 has not implemented yet')
                return 0.0
            elif 3*k0 == -2*k1:
                print('case of 3*k0 == -2*k1 has not implemented yet')
                return 0.0
            elif 2*k0 == -3*k1:
                print('case of 3*k0 == -k1 has not implemented yet')
                return 0.0
            elif k0 == -4*k1:
                print('case of k0 == -4*k1 has not implemented yet')
                return 0.0
            elif 3*k0 == 2*k1:
                res0 = (-(180180*a0*(a0**4 + 6*a0**2*a1**2 + 3*a1**4)*np.cos(ang0*k0 + phi0) + 30030*(a0**5 + 4*a0**3*a1**2)*np.cos(3*(ang0*k0 + phi0)) - 18018*a0**5*np.cos(5*(ang0*k0 + phi0)) + 720720*a0**4*a1*np.cos((ang0*k0)/2. + 2*phi0 - phi1) + 1081080*a0**2*a1**3*np.cos((ang0*k0)/2. + 2*phi0 - phi1) + 36036*a0**4*a1*np.cos((5*ang0*k0)/2. + 4*phi0 - phi1) + 360360*a0**4*a1*np.cos((3*ang0*k0)/2. + phi1) + 720720*a0**2*a1**3*np.cos((3*ang0*k0)/2. + phi1) + 120120*a1**5*np.cos((3*ang0*k0)/2. + phi1) + 102960*a0**4*a1*np.cos((7*ang0*k0)/2. + 2*phi0 + phi1) + 154440*a0**2*a1**3*np.cos((7*ang0*k0)/2. + 2*phi0 + phi1) - 81900*a0**4*a1*np.cos((11*ang0*k0)/2. + 4*phi0 + phi1) + 270270*a0**3*a1**2*np.cos(2*ang0*k0 - phi0 + 2*phi1) + 180180*a0*a1**4*np.cos(2*ang0*k0 - phi0 + 2*phi1) + 135135*a0**3*a1**2*np.cos(4*ang0*k0 + phi0 + 2*phi1) + 90090*a0*a1**4*np.cos(4*ang0*k0 + phi0 + 2*phi1) - 150150*a0**3*a1**2*np.cos(6*ang0*k0 + 3*phi0 + 2*phi1) + 80080*a0**2*a1**3*np.cos((9*ang0*k0)/2. + 3*phi1) + 20020*a1**5*np.cos((9*ang0*k0)/2. + 3*phi1) + 72072*a0**2*a1**3*np.cos((5*ang0*k0)/2. - 2*phi0 + 3*phi1) - 138600*a0**2*a1**3*np.cos((13*ang0*k0)/2. + 2*phi0 + 3*phi1) + 18018*a0*a1**4*np.cos(5*ang0*k0 - phi0 + 4*phi1) - 64350*a0*a1**4*np.cos(7*ang0*k0 + phi0 + 4*phi1) - 12012*a1**5*np.cos((15*ang0*k0)/2. + 5*phi1) - 180180*a0**3*a1**2*ang0*k0*np.sin(3*phi0 - 2*phi1))/(1.44144e6*k0))
                res1 = (-(180180*a0*(a0**4 + 6*a0**2*a1**2 + 3*a1**4)*np.cos(ang1*k0 + phi0) + 30030*(a0**5 + 4*a0**3*a1**2)*np.cos(3*(ang1*k0 + phi0)) - 18018*a0**5*np.cos(5*(ang1*k0 + phi0)) + 720720*a0**4*a1*np.cos((ang1*k0)/2. + 2*phi0 - phi1) + 1081080*a0**2*a1**3*np.cos((ang1*k0)/2. + 2*phi0 - phi1) + 36036*a0**4*a1*np.cos((5*ang1*k0)/2. + 4*phi0 - phi1) + 360360*a0**4*a1*np.cos((3*ang1*k0)/2. + phi1) + 720720*a0**2*a1**3*np.cos((3*ang1*k0)/2. + phi1) + 120120*a1**5*np.cos((3*ang1*k0)/2. + phi1) + 102960*a0**4*a1*np.cos((7*ang1*k0)/2. + 2*phi0 + phi1) + 154440*a0**2*a1**3*np.cos((7*ang1*k0)/2. + 2*phi0 + phi1) - 81900*a0**4*a1*np.cos((11*ang1*k0)/2. + 4*phi0 + phi1) + 270270*a0**3*a1**2*np.cos(2*ang1*k0 - phi0 + 2*phi1) + 180180*a0*a1**4*np.cos(2*ang1*k0 - phi0 + 2*phi1) + 135135*a0**3*a1**2*np.cos(4*ang1*k0 + phi0 + 2*phi1) + 90090*a0*a1**4*np.cos(4*ang1*k0 + phi0 + 2*phi1) - 150150*a0**3*a1**2*np.cos(6*ang1*k0 + 3*phi0 + 2*phi1) + 80080*a0**2*a1**3*np.cos((9*ang1*k0)/2. + 3*phi1) + 20020*a1**5*np.cos((9*ang1*k0)/2. + 3*phi1) + 72072*a0**2*a1**3*np.cos((5*ang1*k0)/2. - 2*phi0 + 3*phi1) - 138600*a0**2*a1**3*np.cos((13*ang1*k0)/2. + 2*phi0 + 3*phi1) + 18018*a0*a1**4*np.cos(5*ang1*k0 - phi0 + 4*phi1) - 64350*a0*a1**4*np.cos(7*ang1*k0 + phi0 + 4*phi1) - 12012*a1**5*np.cos((15*ang1*k0)/2. + 5*phi1) - 180180*a0**3*a1**2*ang1*k0*np.sin(3*phi0 - 2*phi1))/(1.44144e6*k0))
                return res1 - res0
            elif 2*k0 == k1:
                res0 = (-(2520*a0*(a0**4 + 6*a0**2*a1**2 + 3*a1**4)*np.cos(ang0*k0 + phi0) + 420*(a0**5 + 4*a0**3*a1**2)*np.cos(3*(ang0*k0 + phi0)) - 252*a0**5*np.cos(5*(ang0*k0 + phi0)) + 630*a0**4*a1*np.cos(2*ang0*k0 + 4*phi0 - phi1) + 3780*a0**4*a1*np.cos(2*ang0*k0 + phi1) + 7560*a0**2*a1**3*np.cos(2*ang0*k0 + phi1) + 1260*a1**5*np.cos(2*ang0*k0 + phi1) + 840*a0**2*a1**3*np.cos(3*(2*ang0*k0 + phi1)) + 210*a1**5*np.cos(3*(2*ang0*k0 + phi1)) - 126*a1**5*np.cos(5*(2*ang0*k0 + phi1)) + 1260*a0**4*a1*np.cos(4*ang0*k0 + 2*phi0 + phi1) + 1890*a0**2*a1**3*np.cos(4*ang0*k0 + 2*phi0 + phi1) - 1050*a0**4*a1*np.cos(6*ang0*k0 + 4*phi0 + phi1) - 2520*a0**3*a1**2*np.cos(ang0*k0 - 3*phi0 + 2*phi1) + 2520*a0**3*a1**2*np.cos(3*ang0*k0 - phi0 + 2*phi1) + 1680*a0*a1**4*np.cos(3*ang0*k0 - phi0 + 2*phi1) + 1512*a0**3*a1**2*np.cos(5*ang0*k0 + phi0 + 2*phi1) + 1008*a0*a1**4*np.cos(5*ang0*k0 + phi0 + 2*phi1) - 1800*a0**3*a1**2*np.cos(7*ang0*k0 + 3*phi0 + 2*phi1) + 630*a0**2*a1**3*np.cos(4*ang0*k0 - 2*phi0 + 3*phi1) - 1575*a0**2*a1**3*np.cos(8*ang0*k0 + 2*phi0 + 3*phi1) + 180*a0*a1**4*np.cos(7*ang0*k0 - phi0 + 4*phi1) - 700*a0*a1**4*np.cos(9*ang0*k0 + phi0 + 4*phi1) - 5040*a0**4*a1*ang0*k0*np.sin(2*phi0 - phi1) - 7560*a0**2*a1**3*ang0*k0*np.sin(2*phi0 - phi1))/(20160.*k0))
                res1 = (-(2520*a0*(a0**4 + 6*a0**2*a1**2 + 3*a1**4)*np.cos(ang1*k0 + phi0) + 420*(a0**5 + 4*a0**3*a1**2)*np.cos(3*(ang1*k0 + phi0)) - 252*a0**5*np.cos(5*(ang1*k0 + phi0)) + 630*a0**4*a1*np.cos(2*ang1*k0 + 4*phi0 - phi1) + 3780*a0**4*a1*np.cos(2*ang1*k0 + phi1) + 7560*a0**2*a1**3*np.cos(2*ang1*k0 + phi1) + 1260*a1**5*np.cos(2*ang1*k0 + phi1) + 840*a0**2*a1**3*np.cos(3*(2*ang1*k0 + phi1)) + 210*a1**5*np.cos(3*(2*ang1*k0 + phi1)) - 126*a1**5*np.cos(5*(2*ang1*k0 + phi1)) + 1260*a0**4*a1*np.cos(4*ang1*k0 + 2*phi0 + phi1) + 1890*a0**2*a1**3*np.cos(4*ang1*k0 + 2*phi0 + phi1) - 1050*a0**4*a1*np.cos(6*ang1*k0 + 4*phi0 + phi1) - 2520*a0**3*a1**2*np.cos(ang1*k0 - 3*phi0 + 2*phi1) + 2520*a0**3*a1**2*np.cos(3*ang1*k0 - phi0 + 2*phi1) + 1680*a0*a1**4*np.cos(3*ang1*k0 - phi0 + 2*phi1) + 1512*a0**3*a1**2*np.cos(5*ang1*k0 + phi0 + 2*phi1) + 1008*a0*a1**4*np.cos(5*ang1*k0 + phi0 + 2*phi1) - 1800*a0**3*a1**2*np.cos(7*ang1*k0 + 3*phi0 + 2*phi1) + 630*a0**2*a1**3*np.cos(4*ang1*k0 - 2*phi0 + 3*phi1) - 1575*a0**2*a1**3*np.cos(8*ang1*k0 + 2*phi0 + 3*phi1) + 180*a0*a1**4*np.cos(7*ang1*k0 - phi0 + 4*phi1) - 700*a0*a1**4*np.cos(9*ang1*k0 + phi0 + 4*phi1) - 5040*a0**4*a1*ang1*k0*np.sin(2*phi0 - phi1) - 7560*a0**2*a1**3*ang1*k0*np.sin(2*phi0 - phi1))/(20160.*k0))
                return res1 - res0
            elif 4*k0 == k1:
                res0 = (-(942480*a0*(a0**4 + 6*a0**2*a1**2 + 3*a1**4)*np.cos(ang0*k0 + phi0) + 157080*(a0**5 + 4*a0**3*a1**2)*np.cos(3*(ang0*k0 + phi0)) - 94248*a0**5*np.cos(5*(ang0*k0 + phi0)) + 706860*a0**4*a1*np.cos(4*ang0*k0 + phi1) + 1413720*a0**2*a1**3*np.cos(4*ang0*k0 + phi1) + 235620*a1**5*np.cos(4*ang0*k0 + phi1) + 157080*a0**2*a1**3*np.cos(3*(4*ang0*k0 + phi1)) + 39270*a1**5*np.cos(3*(4*ang0*k0 + phi1)) - 23562*a1**5*np.cos(5*(4*ang0*k0 + phi1)) - 942480*a0**4*a1*np.cos(2*ang0*k0 - 2*phi0 + phi1) - 1413720*a0**2*a1**3*np.cos(2*ang0*k0 - 2*phi0 + phi1) + 314160*a0**4*a1*np.cos(6*ang0*k0 + 2*phi0 + phi1) + 471240*a0**2*a1**3*np.cos(6*ang0*k0 + 2*phi0 + phi1) - 294525*a0**4*a1*np.cos(8*ang0*k0 + 4*phi0 + phi1) - 188496*a0**3*a1**2*np.cos(5*ang0*k0 - 3*phi0 + 2*phi1) + 403920*a0**3*a1**2*np.cos(7*ang0*k0 - phi0 + 2*phi1) + 269280*a0*a1**4*np.cos(7*ang0*k0 - phi0 + 2*phi1) + 314160*a0**3*a1**2*np.cos(9*ang0*k0 + phi0 + 2*phi1) + 209440*a0*a1**4*np.cos(9*ang0*k0 + phi0 + 2*phi1) - 428400*a0**3*a1**2*np.cos(11*ang0*k0 + 3*phi0 + 2*phi1) + 94248*a0**2*a1**3*np.cos(10*ang0*k0 - 2*phi0 + 3*phi1) - 336600*a0**2*a1**3*np.cos(14*ang0*k0 + 2*phi0 + 3*phi1) + 31416*a0*a1**4*np.cos(15*ang0*k0 - phi0 + 4*phi1) - 138600*a0*a1**4*np.cos(17*ang0*k0 + phi0 + 4*phi1) - 471240*a0**4*a1*ang0*k0*np.sin(4*phi0 - phi1))/(7.53984e6*k0))
                res1 = (-(942480*a0*(a0**4 + 6*a0**2*a1**2 + 3*a1**4)*np.cos(ang1*k0 + phi0) + 157080*(a0**5 + 4*a0**3*a1**2)*np.cos(3*(ang1*k0 + phi0)) - 94248*a0**5*np.cos(5*(ang1*k0 + phi0)) + 706860*a0**4*a1*np.cos(4*ang1*k0 + phi1) + 1413720*a0**2*a1**3*np.cos(4*ang1*k0 + phi1) + 235620*a1**5*np.cos(4*ang1*k0 + phi1) + 157080*a0**2*a1**3*np.cos(3*(4*ang1*k0 + phi1)) + 39270*a1**5*np.cos(3*(4*ang1*k0 + phi1)) - 23562*a1**5*np.cos(5*(4*ang1*k0 + phi1)) - 942480*a0**4*a1*np.cos(2*ang1*k0 - 2*phi0 + phi1) - 1413720*a0**2*a1**3*np.cos(2*ang1*k0 - 2*phi0 + phi1) + 314160*a0**4*a1*np.cos(6*ang1*k0 + 2*phi0 + phi1) + 471240*a0**2*a1**3*np.cos(6*ang1*k0 + 2*phi0 + phi1) - 294525*a0**4*a1*np.cos(8*ang1*k0 + 4*phi0 + phi1) - 188496*a0**3*a1**2*np.cos(5*ang1*k0 - 3*phi0 + 2*phi1) + 403920*a0**3*a1**2*np.cos(7*ang1*k0 - phi0 + 2*phi1) + 269280*a0*a1**4*np.cos(7*ang1*k0 - phi0 + 2*phi1) + 314160*a0**3*a1**2*np.cos(9*ang1*k0 + phi0 + 2*phi1) + 209440*a0*a1**4*np.cos(9*ang1*k0 + phi0 + 2*phi1) - 428400*a0**3*a1**2*np.cos(11*ang1*k0 + 3*phi0 + 2*phi1) + 94248*a0**2*a1**3*np.cos(10*ang1*k0 - 2*phi0 + 3*phi1) - 336600*a0**2*a1**3*np.cos(14*ang1*k0 + 2*phi0 + 3*phi1) + 31416*a0*a1**4*np.cos(15*ang1*k0 - phi0 + 4*phi1) - 138600*a0*a1**4*np.cos(17*ang1*k0 + phi0 + 4*phi1) - 471240*a0**4*a1*ang1*k0*np.sin(4*phi0 - phi1))/(7.53984e6*k0))
                return res1 - res0
            else:
                res0 = (((-30*a0*(a0**4 + 6*a0**2*a1**2 + 3*a1**4)*np.cos(ang0*k0 + phi0))/k0 - (5*(a0**5 + 4*a0**3*a1**2)*np.cos(3*(ang0*k0 + phi0)))/k0 + (3*a0**5*np.cos(5*(ang0*k0 + phi0)))/k0 + (15*a0*a1**4*np.cos(ang0*k0 - 4*ang0*k1 + phi0 - 4*phi1))/(k0 - 4*k1) + (30*a0**2*a1**3*np.cos(2*ang0*k0 - 3*ang0*k1 + 2*phi0 - 3*phi1))/(2*k0 - 3*k1) + (90*a0**3*a1**2*np.cos(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1))/(k0 - 2*k1) + (60*a0*a1**4*np.cos(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1))/(k0 - 2*k1) - (30*a0**3*a1**2*np.cos(3*ang0*k0 - 2*ang0*k1 + 3*phi0 - 2*phi1))/(3*k0 - 2*k1) - (60*a0**4*a1*np.cos(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1))/(2*k0 - k1) - (90*a0**2*a1**3*np.cos(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1))/(2*k0 - k1) - (15*a0**4*a1*np.cos(4*ang0*k0 - ang0*k1 + 4*phi0 - phi1))/(4*k0 - k1) - (90*a0**4*a1*np.cos(ang0*k1 + phi1))/k1 - (180*a0**2*a1**3*np.cos(ang0*k1 + phi1))/k1 - (30*a1**5*np.cos(ang0*k1 + phi1))/k1 - (20*a0**2*a1**3*np.cos(3*(ang0*k1 + phi1)))/k1 - (5*a1**5*np.cos(3*(ang0*k1 + phi1)))/k1 + (3*a1**5*np.cos(5*(ang0*k1 + phi1)))/k1 - (60*a0**4*a1*np.cos(2*ang0*k0 + ang0*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (90*a0**2*a1**3*np.cos(2*ang0*k0 + ang0*k1 + 2*phi0 + phi1))/(2*k0 + k1) + (75*a0**4*a1*np.cos(4*ang0*k0 + ang0*k1 + 4*phi0 + phi1))/(4*k0 + k1) - (90*a0**3*a1**2*np.cos(ang0*k0 + 2*ang0*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (60*a0*a1**4*np.cos(ang0*k0 + 2*ang0*k1 + phi0 + 2*phi1))/(k0 + 2*k1) + (150*a0**3*a1**2*np.cos(3*ang0*k0 + 2*ang0*k1 + 3*phi0 + 2*phi1))/(3*k0 + 2*k1) + (150*a0**2*a1**3*np.cos(2*ang0*k0 + 3*ang0*k1 + 2*phi0 + 3*phi1))/(2*k0 + 3*k1) + (75*a0*a1**4*np.cos(ang0*k0 + 4*ang0*k1 + phi0 + 4*phi1))/(k0 + 4*k1))/240.)
                res1 = (((-30*a0*(a0**4 + 6*a0**2*a1**2 + 3*a1**4)*np.cos(ang1*k0 + phi0))/k0 - (5*(a0**5 + 4*a0**3*a1**2)*np.cos(3*(ang1*k0 + phi0)))/k0 + (3*a0**5*np.cos(5*(ang1*k0 + phi0)))/k0 + (15*a0*a1**4*np.cos(ang1*k0 - 4*ang1*k1 + phi0 - 4*phi1))/(k0 - 4*k1) + (30*a0**2*a1**3*np.cos(2*ang1*k0 - 3*ang1*k1 + 2*phi0 - 3*phi1))/(2*k0 - 3*k1) + (90*a0**3*a1**2*np.cos(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1))/(k0 - 2*k1) + (60*a0*a1**4*np.cos(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1))/(k0 - 2*k1) - (30*a0**3*a1**2*np.cos(3*ang1*k0 - 2*ang1*k1 + 3*phi0 - 2*phi1))/(3*k0 - 2*k1) - (60*a0**4*a1*np.cos(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1))/(2*k0 - k1) - (90*a0**2*a1**3*np.cos(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1))/(2*k0 - k1) - (15*a0**4*a1*np.cos(4*ang1*k0 - ang1*k1 + 4*phi0 - phi1))/(4*k0 - k1) - (90*a0**4*a1*np.cos(ang1*k1 + phi1))/k1 - (180*a0**2*a1**3*np.cos(ang1*k1 + phi1))/k1 - (30*a1**5*np.cos(ang1*k1 + phi1))/k1 - (20*a0**2*a1**3*np.cos(3*(ang1*k1 + phi1)))/k1 - (5*a1**5*np.cos(3*(ang1*k1 + phi1)))/k1 + (3*a1**5*np.cos(5*(ang1*k1 + phi1)))/k1 - (60*a0**4*a1*np.cos(2*ang1*k0 + ang1*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (90*a0**2*a1**3*np.cos(2*ang1*k0 + ang1*k1 + 2*phi0 + phi1))/(2*k0 + k1) + (75*a0**4*a1*np.cos(4*ang1*k0 + ang1*k1 + 4*phi0 + phi1))/(4*k0 + k1) - (90*a0**3*a1**2*np.cos(ang1*k0 + 2*ang1*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (60*a0*a1**4*np.cos(ang1*k0 + 2*ang1*k1 + phi0 + 2*phi1))/(k0 + 2*k1) + (150*a0**3*a1**2*np.cos(3*ang1*k0 + 2*ang1*k1 + 3*phi0 + 2*phi1))/(3*k0 + 2*k1) + (150*a0**2*a1**3*np.cos(2*ang1*k0 + 3*ang1*k1 + 2*phi0 + 3*phi1))/(2*k0 + 3*k1) + (75*a0*a1**4*np.cos(ang1*k0 + 4*ang1*k1 + phi0 + 4*phi1))/(k0 + 4*k1))/240.)
                return res1 - res0
        elif j == 4:
            res0 = ((12*a0**6*ang0 + 108*a0**4*a1**2*ang0 + 108*a0**2*a1**4*ang0 + 12*a1**6*ang0 - (3*(a0**6 + 8*a0**4*a1**2 + 6*a0**2*a1**4)*np.sin(2*(ang0*k0 + phi0)))/k0 - (3*(a0**6 + 5*a0**4*a1**2)*np.sin(4*(ang0*k0 + phi0)))/k0 + (a0**6*np.sin(6*(ang0*k0 + phi0)))/k0 - (12*a0*a1**5*np.sin(ang0*k0 - 5*ang0*k1 + phi0 - 5*phi1))/(k0 - 5*k1) - (24*a0**3*a1**3*np.sin(ang0*k0 - 3*ang0*k1 + phi0 - 3*phi1))/(k0 - 3*k1) - (12*a0*a1**5*np.sin(ang0*k0 - 3*ang0*k1 + phi0 - 3*phi1))/(k0 - 3*k1) - (3*a0**2*a1**4*np.sin(2*(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1)))/(k0 - 2*k1) - (3*a0**4*a1**2*np.sin(4*ang0*k0 - 2*ang0*k1 + 4*phi0 - 2*phi1))/(2*k0 - k1) + (72*a0**5*a1*np.sin(ang0*k0 - ang0*k1 + phi0 - phi1))/(k0 - k1) + (216*a0**3*a1**3*np.sin(ang0*k0 - ang0*k1 + phi0 - phi1))/(k0 - k1) + (72*a0*a1**5*np.sin(ang0*k0 - ang0*k1 + phi0 - phi1))/(k0 - k1) + (36*a0**4*a1**2*np.sin(2*(ang0*k0 - ang0*k1 + phi0 - phi1)))/(k0 - k1) + (36*a0**2*a1**4*np.sin(2*(ang0*k0 - ang0*k1 + phi0 - phi1)))/(k0 - k1) + (8*a0**3*a1**3*np.sin(3*(ang0*k0 - ang0*k1 + phi0 - phi1)))/(k0 - k1) - (12*a0**5*a1*np.sin(3*ang0*k0 - ang0*k1 + 3*phi0 - phi1))/(3*k0 - k1) - (24*a0**3*a1**3*np.sin(3*ang0*k0 - ang0*k1 + 3*phi0 - phi1))/(3*k0 - k1) - (12*a0**5*a1*np.sin(5*ang0*k0 - ang0*k1 + 5*phi0 - phi1))/(5*k0 - k1) - (18*a0**4*a1**2*np.sin(2*(ang0*k1 + phi1)))/k1 - (24*a0**2*a1**4*np.sin(2*(ang0*k1 + phi1)))/k1 - (3*a1**6*np.sin(2*(ang0*k1 + phi1)))/k1 - (15*a0**2*a1**4*np.sin(4*(ang0*k1 + phi1)))/k1 - (3*a1**6*np.sin(4*(ang0*k1 + phi1)))/k1 + (a1**6*np.sin(6*(ang0*k1 + phi1)))/k1 - (24*a0**5*a1*np.sin(ang0*(k0 + k1) + phi0 + phi1))/(k0 + k1) - (72*a0**3*a1**3*np.sin(ang0*(k0 + k1) + phi0 + phi1))/(k0 + k1) - (24*a0*a1**5*np.sin(ang0*(k0 + k1) + phi0 + phi1))/(k0 + k1) - (60*a0**4*a1**2*np.sin(2*(ang0*(k0 + k1) + phi0 + phi1)))/(k0 + k1) - (60*a0**2*a1**4*np.sin(2*(ang0*(k0 + k1) + phi0 + phi1)))/(k0 + k1) + (40*a0**3*a1**3*np.sin(3*(ang0*(k0 + k1) + phi0 + phi1)))/(k0 + k1) + (45*a0**4*a1**2*np.sin(2*(2*ang0*k0 + ang0*k1 + 2*phi0 + phi1)))/(2*k0 + k1) - (60*a0**5*a1*np.sin(3*ang0*k0 + ang0*k1 + 3*phi0 + phi1))/(3*k0 + k1) - (120*a0**3*a1**3*np.sin(3*ang0*k0 + ang0*k1 + 3*phi0 + phi1))/(3*k0 + k1) + (36*a0**5*a1*np.sin(5*ang0*k0 + ang0*k1 + 5*phi0 + phi1))/(5*k0 + k1) + (45*a0**2*a1**4*np.sin(2*(ang0*k0 + 2*ang0*k1 + phi0 + 2*phi1)))/(k0 + 2*k1) - (120*a0**3*a1**3*np.sin(ang0*k0 + 3*ang0*k1 + phi0 + 3*phi1))/(k0 + 3*k1) - (60*a0*a1**5*np.sin(ang0*k0 + 3*ang0*k1 + phi0 + 3*phi1))/(k0 + 3*k1) + (36*a0*a1**5*np.sin(ang0*k0 + 5*ang0*k1 + phi0 + 5*phi1))/(k0 + 5*k1))/192.)
            res1 = ((12*a0**6*ang1 + 108*a0**4*a1**2*ang1 + 108*a0**2*a1**4*ang1 + 12*a1**6*ang1 - (3*(a0**6 + 8*a0**4*a1**2 + 6*a0**2*a1**4)*np.sin(2*(ang1*k0 + phi0)))/k0 - (3*(a0**6 + 5*a0**4*a1**2)*np.sin(4*(ang1*k0 + phi0)))/k0 + (a0**6*np.sin(6*(ang1*k0 + phi0)))/k0 - (12*a0*a1**5*np.sin(ang1*k0 - 5*ang1*k1 + phi0 - 5*phi1))/(k0 - 5*k1) - (24*a0**3*a1**3*np.sin(ang1*k0 - 3*ang1*k1 + phi0 - 3*phi1))/(k0 - 3*k1) - (12*a0*a1**5*np.sin(ang1*k0 - 3*ang1*k1 + phi0 - 3*phi1))/(k0 - 3*k1) - (3*a0**2*a1**4*np.sin(2*(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1)))/(k0 - 2*k1) - (3*a0**4*a1**2*np.sin(4*ang1*k0 - 2*ang1*k1 + 4*phi0 - 2*phi1))/(2*k0 - k1) + (72*a0**5*a1*np.sin(ang1*k0 - ang1*k1 + phi0 - phi1))/(k0 - k1) + (216*a0**3*a1**3*np.sin(ang1*k0 - ang1*k1 + phi0 - phi1))/(k0 - k1) + (72*a0*a1**5*np.sin(ang1*k0 - ang1*k1 + phi0 - phi1))/(k0 - k1) + (36*a0**4*a1**2*np.sin(2*(ang1*k0 - ang1*k1 + phi0 - phi1)))/(k0 - k1) + (36*a0**2*a1**4*np.sin(2*(ang1*k0 - ang1*k1 + phi0 - phi1)))/(k0 - k1) + (8*a0**3*a1**3*np.sin(3*(ang1*k0 - ang1*k1 + phi0 - phi1)))/(k0 - k1) - (12*a0**5*a1*np.sin(3*ang1*k0 - ang1*k1 + 3*phi0 - phi1))/(3*k0 - k1) - (24*a0**3*a1**3*np.sin(3*ang1*k0 - ang1*k1 + 3*phi0 - phi1))/(3*k0 - k1) - (12*a0**5*a1*np.sin(5*ang1*k0 - ang1*k1 + 5*phi0 - phi1))/(5*k0 - k1) - (18*a0**4*a1**2*np.sin(2*(ang1*k1 + phi1)))/k1 - (24*a0**2*a1**4*np.sin(2*(ang1*k1 + phi1)))/k1 - (3*a1**6*np.sin(2*(ang1*k1 + phi1)))/k1 - (15*a0**2*a1**4*np.sin(4*(ang1*k1 + phi1)))/k1 - (3*a1**6*np.sin(4*(ang1*k1 + phi1)))/k1 + (a1**6*np.sin(6*(ang1*k1 + phi1)))/k1 - (24*a0**5*a1*np.sin(ang1*(k0 + k1) + phi0 + phi1))/(k0 + k1) - (72*a0**3*a1**3*np.sin(ang1*(k0 + k1) + phi0 + phi1))/(k0 + k1) - (24*a0*a1**5*np.sin(ang1*(k0 + k1) + phi0 + phi1))/(k0 + k1) - (60*a0**4*a1**2*np.sin(2*(ang1*(k0 + k1) + phi0 + phi1)))/(k0 + k1) - (60*a0**2*a1**4*np.sin(2*(ang1*(k0 + k1) + phi0 + phi1)))/(k0 + k1) + (40*a0**3*a1**3*np.sin(3*(ang1*(k0 + k1) + phi0 + phi1)))/(k0 + k1) + (45*a0**4*a1**2*np.sin(2*(2*ang1*k0 + ang1*k1 + 2*phi0 + phi1)))/(2*k0 + k1) - (60*a0**5*a1*np.sin(3*ang1*k0 + ang1*k1 + 3*phi0 + phi1))/(3*k0 + k1) - (120*a0**3*a1**3*np.sin(3*ang1*k0 + ang1*k1 + 3*phi0 + phi1))/(3*k0 + k1) + (36*a0**5*a1*np.sin(5*ang1*k0 + ang1*k1 + 5*phi0 + phi1))/(5*k0 + k1) + (45*a0**2*a1**4*np.sin(2*(ang1*k0 + 2*ang1*k1 + phi0 + 2*phi1)))/(k0 + 2*k1) - (120*a0**3*a1**3*np.sin(ang1*k0 + 3*ang1*k1 + phi0 + 3*phi1))/(k0 + 3*k1) - (60*a0*a1**5*np.sin(ang1*k0 + 3*ang1*k1 + phi0 + 3*phi1))/(k0 + 3*k1) + (36*a0*a1**5*np.sin(ang1*k0 + 5*ang1*k1 + phi0 + 5*phi1))/(k0 + 5*k1))/192.)
            return res1 - res0
        elif j == 5:
            res0 = (((-525*a0*(a0**6 + 12*a0**4*a1**2 + 18*a0**2*a1**4 + 4*a1**6)*np.cos(ang0*k0 + phi0))/k0 - (35*(a0**7 + 10*a0**5*a1**2 + 10*a0**3*a1**4)*np.cos(3*(ang0*k0 + phi0)))/k0 + (63*a0**7*np.cos(5*(ang0*k0 + phi0)))/k0 + (378*a0**5*a1**2*np.cos(5*(ang0*k0 + phi0)))/k0 - (15*a0**7*np.cos(7*(ang0*k0 + phi0)))/k0 - (315*a0*a1**6*np.cos(ang0*k0 - 6*ang0*k1 + phi0 - 6*phi1))/(k0 - 6*k1) + (105*a0**2*a1**5*np.cos(2*ang0*k0 - 5*ang0*k1 + 2*phi0 - 5*phi1))/(2*k0 - 5*k1) + (525*a0**3*a1**4*np.cos(ang0*k0 - 4*ang0*k1 + phi0 - 4*phi1))/(k0 - 4*k1) + (210*a0*a1**6*np.cos(ang0*k0 - 4*ang0*k1 + phi0 - 4*phi1))/(k0 - 4*k1) + (525*a0**3*a1**4*np.cos(3*ang0*k0 - 4*ang0*k1 + 3*phi0 - 4*phi1))/(3*k0 - 4*k1) + (2100*a0**4*a1**3*np.cos(2*ang0*k0 - 3*ang0*k1 + 2*phi0 - 3*phi1))/(2*k0 - 3*k1) + (1575*a0**2*a1**5*np.cos(2*ang0*k0 - 3*ang0*k1 + 2*phi0 - 3*phi1))/(2*k0 - 3*k1) - (525*a0**4*a1**3*np.cos(4*ang0*k0 - 3*ang0*k1 + 4*phi0 - 3*phi1))/(4*k0 - 3*k1) + (3150*a0**5*a1**2*np.cos(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1))/(k0 - 2*k1) + (6300*a0**3*a1**4*np.cos(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1))/(k0 - 2*k1) + (1575*a0*a1**6*np.cos(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1))/(k0 - 2*k1) - (1575*a0**5*a1**2*np.cos(3*ang0*k0 - 2*ang0*k1 + 3*phi0 - 2*phi1))/(3*k0 - 2*k1) - (2100*a0**3*a1**4*np.cos(3*ang0*k0 - 2*ang0*k1 + 3*phi0 - 2*phi1))/(3*k0 - 2*k1) - (105*a0**5*a1**2*np.cos(5*ang0*k0 - 2*ang0*k1 + 5*phi0 - 2*phi1))/(5*k0 - 2*k1) - (1575*a0**6*a1*np.cos(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1))/(2*k0 - k1) - (6300*a0**4*a1**3*np.cos(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1))/(2*k0 - k1) - (3150*a0**2*a1**5*np.cos(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1))/(2*k0 - k1) - (210*a0**6*a1*np.cos(4*ang0*k0 - ang0*k1 + 4*phi0 - phi1))/(4*k0 - k1) - (525*a0**4*a1**3*np.cos(4*ang0*k0 - ang0*k1 + 4*phi0 - phi1))/(4*k0 - k1) + (315*a0**6*a1*np.cos(6*ang0*k0 - ang0*k1 + 6*phi0 - phi1))/(6*k0 - k1) - (2100*a0**6*a1*np.cos(ang0*k1 + phi1))/k1 - (9450*a0**4*a1**3*np.cos(ang0*k1 + phi1))/k1 - (6300*a0**2*a1**5*np.cos(ang0*k1 + phi1))/k1 - (525*a1**7*np.cos(ang0*k1 + phi1))/k1 - (350*a0**4*a1**3*np.cos(3*(ang0*k1 + phi1)))/k1 - (350*a0**2*a1**5*np.cos(3*(ang0*k1 + phi1)))/k1 - (35*a1**7*np.cos(3*(ang0*k1 + phi1)))/k1 + (378*a0**2*a1**5*np.cos(5*(ang0*k1 + phi1)))/k1 + (63*a1**7*np.cos(5*(ang0*k1 + phi1)))/k1 - (15*a1**7*np.cos(7*(ang0*k1 + phi1)))/k1 - (525*a0**6*a1*np.cos(2*ang0*k0 + ang0*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (2100*a0**4*a1**3*np.cos(2*ang0*k0 + ang0*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (1050*a0**2*a1**5*np.cos(2*ang0*k0 + ang0*k1 + 2*phi0 + phi1))/(2*k0 + k1) + (1890*a0**6*a1*np.cos(4*ang0*k0 + ang0*k1 + 4*phi0 + phi1))/(4*k0 + k1) + (4725*a0**4*a1**3*np.cos(4*ang0*k0 + ang0*k1 + 4*phi0 + phi1))/(4*k0 + k1) - (735*a0**6*a1*np.cos(6*ang0*k0 + ang0*k1 + 6*phi0 + phi1))/(6*k0 + k1) - (1050*a0**5*a1**2*np.cos(ang0*k0 + 2*ang0*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (2100*a0**3*a1**4*np.cos(ang0*k0 + 2*ang0*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (525*a0*a1**6*np.cos(ang0*k0 + 2*ang0*k1 + phi0 + 2*phi1))/(k0 + 2*k1) + (4725*a0**5*a1**2*np.cos(3*ang0*k0 + 2*ang0*k1 + 3*phi0 + 2*phi1))/(3*k0 + 2*k1) + (6300*a0**3*a1**4*np.cos(3*ang0*k0 + 2*ang0*k1 + 3*phi0 + 2*phi1))/(3*k0 + 2*k1) - (2205*a0**5*a1**2*np.cos(5*ang0*k0 + 2*ang0*k1 + 5*phi0 + 2*phi1))/(5*k0 + 2*k1) + (6300*a0**4*a1**3*np.cos(2*ang0*k0 + 3*ang0*k1 + 2*phi0 + 3*phi1))/(2*k0 + 3*k1) + (4725*a0**2*a1**5*np.cos(2*ang0*k0 + 3*ang0*k1 + 2*phi0 + 3*phi1))/(2*k0 + 3*k1) - (3675*a0**4*a1**3*np.cos(4*ang0*k0 + 3*ang0*k1 + 4*phi0 + 3*phi1))/(4*k0 + 3*k1) + (4725*a0**3*a1**4*np.cos(ang0*k0 + 4*ang0*k1 + phi0 + 4*phi1))/(k0 + 4*k1) + (1890*a0*a1**6*np.cos(ang0*k0 + 4*ang0*k1 + phi0 + 4*phi1))/(k0 + 4*k1) - (3675*a0**3*a1**4*np.cos(3*ang0*k0 + 4*ang0*k1 + 3*phi0 + 4*phi1))/(3*k0 + 4*k1) - (2205*a0**2*a1**5*np.cos(2*ang0*k0 + 5*ang0*k1 + 2*phi0 + 5*phi1))/(2*k0 + 5*k1) - (735*a0*a1**6*np.cos(ang0*k0 + 6*ang0*k1 + phi0 + 6*phi1))/(k0 + 6*k1))/6720.)
            res1 = (((-525*a0*(a0**6 + 12*a0**4*a1**2 + 18*a0**2*a1**4 + 4*a1**6)*np.cos(ang1*k0 + phi0))/k0 - (35*(a0**7 + 10*a0**5*a1**2 + 10*a0**3*a1**4)*np.cos(3*(ang1*k0 + phi0)))/k0 + (63*a0**7*np.cos(5*(ang1*k0 + phi0)))/k0 + (378*a0**5*a1**2*np.cos(5*(ang1*k0 + phi0)))/k0 - (15*a0**7*np.cos(7*(ang1*k0 + phi0)))/k0 - (315*a0*a1**6*np.cos(ang1*k0 - 6*ang1*k1 + phi0 - 6*phi1))/(k0 - 6*k1) + (105*a0**2*a1**5*np.cos(2*ang1*k0 - 5*ang1*k1 + 2*phi0 - 5*phi1))/(2*k0 - 5*k1) + (525*a0**3*a1**4*np.cos(ang1*k0 - 4*ang1*k1 + phi0 - 4*phi1))/(k0 - 4*k1) + (210*a0*a1**6*np.cos(ang1*k0 - 4*ang1*k1 + phi0 - 4*phi1))/(k0 - 4*k1) + (525*a0**3*a1**4*np.cos(3*ang1*k0 - 4*ang1*k1 + 3*phi0 - 4*phi1))/(3*k0 - 4*k1) + (2100*a0**4*a1**3*np.cos(2*ang1*k0 - 3*ang1*k1 + 2*phi0 - 3*phi1))/(2*k0 - 3*k1) + (1575*a0**2*a1**5*np.cos(2*ang1*k0 - 3*ang1*k1 + 2*phi0 - 3*phi1))/(2*k0 - 3*k1) - (525*a0**4*a1**3*np.cos(4*ang1*k0 - 3*ang1*k1 + 4*phi0 - 3*phi1))/(4*k0 - 3*k1) + (3150*a0**5*a1**2*np.cos(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1))/(k0 - 2*k1) + (6300*a0**3*a1**4*np.cos(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1))/(k0 - 2*k1) + (1575*a0*a1**6*np.cos(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1))/(k0 - 2*k1) - (1575*a0**5*a1**2*np.cos(3*ang1*k0 - 2*ang1*k1 + 3*phi0 - 2*phi1))/(3*k0 - 2*k1) - (2100*a0**3*a1**4*np.cos(3*ang1*k0 - 2*ang1*k1 + 3*phi0 - 2*phi1))/(3*k0 - 2*k1) - (105*a0**5*a1**2*np.cos(5*ang1*k0 - 2*ang1*k1 + 5*phi0 - 2*phi1))/(5*k0 - 2*k1) - (1575*a0**6*a1*np.cos(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1))/(2*k0 - k1) - (6300*a0**4*a1**3*np.cos(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1))/(2*k0 - k1) - (3150*a0**2*a1**5*np.cos(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1))/(2*k0 - k1) - (210*a0**6*a1*np.cos(4*ang1*k0 - ang1*k1 + 4*phi0 - phi1))/(4*k0 - k1) - (525*a0**4*a1**3*np.cos(4*ang1*k0 - ang1*k1 + 4*phi0 - phi1))/(4*k0 - k1) + (315*a0**6*a1*np.cos(6*ang1*k0 - ang1*k1 + 6*phi0 - phi1))/(6*k0 - k1) - (2100*a0**6*a1*np.cos(ang1*k1 + phi1))/k1 - (9450*a0**4*a1**3*np.cos(ang1*k1 + phi1))/k1 - (6300*a0**2*a1**5*np.cos(ang1*k1 + phi1))/k1 - (525*a1**7*np.cos(ang1*k1 + phi1))/k1 - (350*a0**4*a1**3*np.cos(3*(ang1*k1 + phi1)))/k1 - (350*a0**2*a1**5*np.cos(3*(ang1*k1 + phi1)))/k1 - (35*a1**7*np.cos(3*(ang1*k1 + phi1)))/k1 + (378*a0**2*a1**5*np.cos(5*(ang1*k1 + phi1)))/k1 + (63*a1**7*np.cos(5*(ang1*k1 + phi1)))/k1 - (15*a1**7*np.cos(7*(ang1*k1 + phi1)))/k1 - (525*a0**6*a1*np.cos(2*ang1*k0 + ang1*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (2100*a0**4*a1**3*np.cos(2*ang1*k0 + ang1*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (1050*a0**2*a1**5*np.cos(2*ang1*k0 + ang1*k1 + 2*phi0 + phi1))/(2*k0 + k1) + (1890*a0**6*a1*np.cos(4*ang1*k0 + ang1*k1 + 4*phi0 + phi1))/(4*k0 + k1) + (4725*a0**4*a1**3*np.cos(4*ang1*k0 + ang1*k1 + 4*phi0 + phi1))/(4*k0 + k1) - (735*a0**6*a1*np.cos(6*ang1*k0 + ang1*k1 + 6*phi0 + phi1))/(6*k0 + k1) - (1050*a0**5*a1**2*np.cos(ang1*k0 + 2*ang1*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (2100*a0**3*a1**4*np.cos(ang1*k0 + 2*ang1*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (525*a0*a1**6*np.cos(ang1*k0 + 2*ang1*k1 + phi0 + 2*phi1))/(k0 + 2*k1) + (4725*a0**5*a1**2*np.cos(3*ang1*k0 + 2*ang1*k1 + 3*phi0 + 2*phi1))/(3*k0 + 2*k1) + (6300*a0**3*a1**4*np.cos(3*ang1*k0 + 2*ang1*k1 + 3*phi0 + 2*phi1))/(3*k0 + 2*k1) - (2205*a0**5*a1**2*np.cos(5*ang1*k0 + 2*ang1*k1 + 5*phi0 + 2*phi1))/(5*k0 + 2*k1) + (6300*a0**4*a1**3*np.cos(2*ang1*k0 + 3*ang1*k1 + 2*phi0 + 3*phi1))/(2*k0 + 3*k1) + (4725*a0**2*a1**5*np.cos(2*ang1*k0 + 3*ang1*k1 + 2*phi0 + 3*phi1))/(2*k0 + 3*k1) - (3675*a0**4*a1**3*np.cos(4*ang1*k0 + 3*ang1*k1 + 4*phi0 + 3*phi1))/(4*k0 + 3*k1) + (4725*a0**3*a1**4*np.cos(ang1*k0 + 4*ang1*k1 + phi0 + 4*phi1))/(k0 + 4*k1) + (1890*a0*a1**6*np.cos(ang1*k0 + 4*ang1*k1 + phi0 + 4*phi1))/(k0 + 4*k1) - (3675*a0**3*a1**4*np.cos(3*ang1*k0 + 4*ang1*k1 + 3*phi0 + 4*phi1))/(3*k0 + 4*k1) - (2205*a0**2*a1**5*np.cos(2*ang1*k0 + 5*ang1*k1 + 2*phi0 + 5*phi1))/(2*k0 + 5*k1) - (735*a0*a1**6*np.cos(ang1*k0 + 6*ang1*k1 + phi0 + 6*phi1))/(k0 + 6*k1))/6720.)
            return res1 - res0
        else:
            print('invalid index j = ' + str(j) + ' in Iij')
            return 0.0
    elif i == 3:
        if j == 0:
            if k0 == -2*k1:
                print('case of k0 == -2*k1 has not implemented yet')
                return 0.0
            elif k0 == 2*k1:
                print('case of k0 == 2*k1 has not implemented yet')
                return 0.0
            elif 2*k0 == -k1:
                print('case of 2*k0 == -k1 has not implemented yet')
                return 0.0
            elif 2*k0 == k1:
                res0 = ((180*a0**2*a1*ang0*k0*np.cos(2*phi0 - phi1) + 180*a0*(a0**2 + 2*a1**2)*np.sin(ang0*k0 + phi0) + 20*a0**3*np.sin(3*(ang0*k0 + phi0)) + 180*a0**2*a1*np.sin(2*ang0*k0 + phi1) + 90*a1**3*np.sin(2*ang0*k0 + phi1) + 10*a1**3*np.sin(3*(2*ang0*k0 + phi1)) + 45*a0**2*a1*np.sin(4*ang0*k0 + 2*phi0 + phi1) + 60*a0*a1**2*np.sin(3*ang0*k0 - phi0 + 2*phi1) + 36*a0*a1**2*np.sin(5*ang0*k0 + phi0 + 2*phi1))/(240.*k0))
                res1 = ((180*a0**2*a1*ang1*k0*np.cos(2*phi0 - phi1) + 180*a0*(a0**2 + 2*a1**2)*np.sin(ang1*k0 + phi0) + 20*a0**3*np.sin(3*(ang1*k0 + phi0)) + 180*a0**2*a1*np.sin(2*ang1*k0 + phi1) + 90*a1**3*np.sin(2*ang1*k0 + phi1) + 10*a1**3*np.sin(3*(2*ang1*k0 + phi1)) + 45*a0**2*a1*np.sin(4*ang1*k0 + 2*phi0 + phi1) + 60*a0*a1**2*np.sin(3*ang1*k0 - phi0 + 2*phi1) + 36*a0*a1**2*np.sin(5*ang1*k0 + phi0 + 2*phi1))/(240.*k0))
                return res1 - res0
            else:
                res0 = ((9*a0*(a0**2 + 2*a1**2)*k1*(4*k0**4 - 17*k0**2*k1**2 + 4*k1**4)*np.sin(ang0*k0 + phi0) + a0**3*k1*(4*k0**4 - 17*k0**2*k1**2 + 4*k1**4)*np.sin(3*(ang0*k0 + phi0)) + a1*k0*(-9*a0*a1*k1*(-4*k0**3 - 8*k0**2*k1 + k0*k1**2 + 2*k1**3)*np.sin(ang0*(k0 - 2*k1) + phi0 - 2*phi1) + (k0 - 2*k1)*(9*a0**2*k1*(2*k0**2 + 5*k0*k1 + 2*k1**2)*np.sin(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1) + (2*k0 - k1)*(9*(2*a0**2 + a1**2)*(2*k0**2 + 5*k0*k1 + 2*k1**2)*np.sin(ang0*k1 + phi1) + a1**2*(2*k0**2 + 5*k0*k1 + 2*k1**2)*np.sin(3*(ang0*k1 + phi1)) + 9*a0*k1*(a0*(k0 + 2*k1)*np.sin(ang0*(2*k0 + k1) + 2*phi0 + phi1) + a1*(2*k0 + k1)*np.sin(ang0*(k0 + 2*k1) + phi0 + 2*phi1))))))/(12.*(4*k0**5*k1 - 17*k0**3*k1**3 + 4*k0*k1**5)))   
                res1 = ((9*a0*(a0**2 + 2*a1**2)*k1*(4*k0**4 - 17*k0**2*k1**2 + 4*k1**4)*np.sin(ang1*k0 + phi0) + a0**3*k1*(4*k0**4 - 17*k0**2*k1**2 + 4*k1**4)*np.sin(3*(ang1*k0 + phi0)) + a1*k0*(-9*a0*a1*k1*(-4*k0**3 - 8*k0**2*k1 + k0*k1**2 + 2*k1**3)*np.sin(ang1*(k0 - 2*k1) + phi0 - 2*phi1) + (k0 - 2*k1)*(9*a0**2*k1*(2*k0**2 + 5*k0*k1 + 2*k1**2)*np.sin(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1) + (2*k0 - k1)*(9*(2*a0**2 + a1**2)*(2*k0**2 + 5*k0*k1 + 2*k1**2)*np.sin(ang1*k1 + phi1) + a1**2*(2*k0**2 + 5*k0*k1 + 2*k1**2)*np.sin(3*(ang1*k1 + phi1)) + 9*a0*k1*(a0*(k0 + 2*k1)*np.sin(ang1*(2*k0 + k1) + 2*phi0 + phi1) + a1*(2*k0 + k1)*np.sin(ang1*(k0 + 2*k1) + phi0 + 2*phi1))))))/(12.*(4*k0**5*k1 - 17*k0**3*k1**3 + 4*k0*k1**5)))
                return res1 - res0
        elif j == 1:
            if k0 == 3*k1:
                print('case of k0 == 3*k1 has not implemented yet')
                return 0.0
            elif k0 == -k1:
                print('case of k0 == -k1 has not implemented yet')
                return 0.0
            elif 3*k0 == -k1:
                print('case of 3*k0 == -k1 has not implemented yet')
                return 0.0
            elif 3*k0 == k1:
                res0 = (-(60*(a0**4 + 3*a0**2*a1**2)*np.cos(2*(ang0*k0 + phi0)) + 15*a0**4*np.cos(4*(ang0*k0 + phi0)) + a1*(20*a1*(3*a0**2 + a1**2)*np.cos(2*(3*ang0*k0 + phi1)) + 5*a1**3*np.cos(4*(3*ang0*k0 + phi1)) + a0*(90*(a0**2 + a1**2)*np.cos(4*ang0*k0 + phi0 + phi1) + 45*a0*a1*np.cos(2*(4*ang0*k0 + phi0 + phi1)) + 40*a0**2*np.cos(6*ang0*k0 + 3*phi0 + phi1) + 15*a1**2*np.cos(8*ang0*k0 - phi0 + 3*phi1) + 24*a1**2*np.cos(10*ang0*k0 + phi0 + 3*phi1) - 120*a0**2*ang0*k0*np.sin(3*phi0 - phi1))))/(480.*k0))
                res1 = (-(60*(a0**4 + 3*a0**2*a1**2)*np.cos(2*(ang1*k0 + phi0)) + 15*a0**4*np.cos(4*(ang1*k0 + phi0)) + a1*(20*a1*(3*a0**2 + a1**2)*np.cos(2*(3*ang1*k0 + phi1)) + 5*a1**3*np.cos(4*(3*ang1*k0 + phi1)) + a0*(90*(a0**2 + a1**2)*np.cos(4*ang1*k0 + phi0 + phi1) + 45*a0*a1*np.cos(2*(4*ang1*k0 + phi0 + phi1)) + 40*a0**2*np.cos(6*ang1*k0 + 3*phi0 + phi1) + 15*a1**2*np.cos(8*ang1*k0 - phi0 + 3*phi1) + 24*a1**2*np.cos(10*ang1*k0 + phi0 + 3*phi1) - 120*a0**2*ang1*k0*np.sin(3*phi0 - phi1))))/(480.*k0))
                return res1 - res0
            else:
                res0 = (-(4*a0**2*(a0**2 + 3*a1**2)*k1*(9*k0**5 + 9*k0**4*k1 - 82*k0**3*k1**2 - 82*k0**2*k1**3 + 9*k0*k1**4 + 9*k1**5)*np.cos(2*(ang0*k0 + phi0)) + a0**4*k1*(9*k0**5 + 9*k0**4*k1 - 82*k0**3*k1**2 - 82*k0**2*k1**3 + 9*k0*k1**4 + 9*k1**5)*np.cos(4*(ang0*k0 + phi0)) + a1*k0*(8*a0*a1**2*k1*(-9*k0**4 - 36*k0**3*k1 - 26*k0**2*k1**2 + 4*k0*k1**3 + 3*k1**4)*np.cos(ang0*(k0 - 3*k1) + phi0 - 3*phi1) + (k0 - 3*k1)*(8*a0**3*k1*(3*k0**3 + 13*k0**2*k1 + 13*k0*k1**2 + 3*k1**3)*np.cos(3*ang0*k0 - ang0*k1 + 3*phi0 - phi1) + (3*k0 - k1)*(4*a1*(3*a0**2 + a1**2)*(3*k0**3 + 13*k0**2*k1 + 13*k0*k1**2 + 3*k1**3)*np.cos(2*(ang0*k1 + phi1)) + a1**3*(3*k0**3 + 13*k0**2*k1 + 13*k0*k1**2 + 3*k1**3)*np.cos(4*(ang0*k1 + phi1)) + 4*a0*k1*(6*(a0**2 + a1**2)*(3*k0**2 + 10*k0*k1 + 3*k1**2)*np.cos(ang0*(k0 + k1) + phi0 + phi1) + 3*a0*a1*(3*k0**2 + 10*k0*k1 + 3*k1**2)*np.cos(2*(ang0*(k0 + k1) + phi0 + phi1)) + 4*(k0 + k1)*(a0**2*(k0 + 3*k1)*np.cos(ang0*(3*k0 + k1) + 3*phi0 + phi1) + a1**2*(3*k0 + k1)*np.cos(ang0*(k0 + 3*k1) + phi0 + 3*phi1)))))))/(32.*k0*k1*(9*k0**5 + 9*k0**4*k1 - 82*k0**3*k1**2 - 82*k0**2*k1**3 + 9*k0*k1**4 + 9*k1**5)))
                res1 = (-(4*a0**2*(a0**2 + 3*a1**2)*k1*(9*k0**5 + 9*k0**4*k1 - 82*k0**3*k1**2 - 82*k0**2*k1**3 + 9*k0*k1**4 + 9*k1**5)*np.cos(2*(ang1*k0 + phi0)) + a0**4*k1*(9*k0**5 + 9*k0**4*k1 - 82*k0**3*k1**2 - 82*k0**2*k1**3 + 9*k0*k1**4 + 9*k1**5)*np.cos(4*(ang1*k0 + phi0)) + a1*k0*(8*a0*a1**2*k1*(-9*k0**4 - 36*k0**3*k1 - 26*k0**2*k1**2 + 4*k0*k1**3 + 3*k1**4)*np.cos(ang1*(k0 - 3*k1) + phi0 - 3*phi1) + (k0 - 3*k1)*(8*a0**3*k1*(3*k0**3 + 13*k0**2*k1 + 13*k0*k1**2 + 3*k1**3)*np.cos(3*ang1*k0 - ang1*k1 + 3*phi0 - phi1) + (3*k0 - k1)*(4*a1*(3*a0**2 + a1**2)*(3*k0**3 + 13*k0**2*k1 + 13*k0*k1**2 + 3*k1**3)*np.cos(2*(ang1*k1 + phi1)) + a1**3*(3*k0**3 + 13*k0**2*k1 + 13*k0*k1**2 + 3*k1**3)*np.cos(4*(ang1*k1 + phi1)) + 4*a0*k1*(6*(a0**2 + a1**2)*(3*k0**2 + 10*k0*k1 + 3*k1**2)*np.cos(ang1*(k0 + k1) + phi0 + phi1) + 3*a0*a1*(3*k0**2 + 10*k0*k1 + 3*k1**2)*np.cos(2*(ang1*(k0 + k1) + phi0 + phi1)) + 4*(k0 + k1)*(a0**2*(k0 + 3*k1)*np.cos(ang1*(3*k0 + k1) + 3*phi0 + phi1) + a1**2*(3*k0 + k1)*np.cos(ang1*(k0 + 3*k1) + phi0 + 3*phi1)))))))/(32.*k0*k1*(9*k0**5 + 9*k0**4*k1 - 82*k0**3*k1**2 - 82*k0**2*k1**3 + 9*k0*k1**4 + 9*k1**5)))
                return res1 - res0
        elif j == 2:
            if k0 == 4*k1:
                print('case of k0 == 4*k1 has not implemented yet')
                return 0.0
            elif 2*k0 == 3*k1:
                print('case of 2*k0 == 3*k1 has not implemented yet')
                return 0.0
            elif k0 == 2*k1:
                print('case of k0 == 2*k1 has not implemented yet')
                return 0.0
            elif 2*k0 == -k1:
                print('case of 2*k0 == -k1 has not implemented yet')
                return 0.0
            elif 4*k0 == -k1:
                print('case of 4*k0 == -k1 has not implemented yet')
                return 0.0
            elif k0 == -2*k1:
                print('case of k0 == -2*k1 has not implemented yet')
                return 0.0
            elif 3*k0 == -2*k1:
                print('case of 3*k0 == -2*k1 has not implemented yet')
                return 0.0
            elif 2*k0 == -3*k1:
                print('case of 3*k0 == -k1 has not implemented yet')
                return 0.0
            elif k0 == -4*k1:
                print('case of k0 == -4*k1 has not implemented yet')
                return 0.0
            elif 3*k0 == 2*k1:
                res0 = (-(-180180*a0**3*a1**2*ang0*k0*np.cos(3*phi0 - 2*phi1) - 180180*a0*(a0**4 + 6*a0**2*a1**2 + 3*a1**4)*np.sin(ang0*k0 + phi0) + 30030*a0**5*np.sin(3*(ang0*k0 + phi0)) + 120120*a0**3*a1**2*np.sin(3*(ang0*k0 + phi0)) + 18018*a0**5*np.sin(5*(ang0*k0 + phi0)) - 720720*a0**4*a1*np.sin((ang0*k0)/2. + 2*phi0 - phi1) - 1081080*a0**2*a1**3*np.sin((ang0*k0)/2. + 2*phi0 - phi1) + 36036*a0**4*a1*np.sin((5*ang0*k0)/2. + 4*phi0 - phi1) - 360360*a0**4*a1*np.sin((3*ang0*k0)/2. + phi1) - 720720*a0**2*a1**3*np.sin((3*ang0*k0)/2. + phi1) - 120120*a1**5*np.sin((3*ang0*k0)/2. + phi1) + 102960*a0**4*a1*np.sin((7*ang0*k0)/2. + 2*phi0 + phi1) + 154440*a0**2*a1**3*np.sin((7*ang0*k0)/2. + 2*phi0 + phi1) + 81900*a0**4*a1*np.sin((11*ang0*k0)/2. + 4*phi0 + phi1) - 270270*a0**3*a1**2*np.sin(2*ang0*k0 - phi0 + 2*phi1) - 180180*a0*a1**4*np.sin(2*ang0*k0 - phi0 + 2*phi1) + 135135*a0**3*a1**2*np.sin(4*ang0*k0 + phi0 + 2*phi1) + 90090*a0*a1**4*np.sin(4*ang0*k0 + phi0 + 2*phi1) + 150150*a0**3*a1**2*np.sin(6*ang0*k0 + 3*phi0 + 2*phi1) + 80080*a0**2*a1**3*np.sin((9*ang0*k0)/2. + 3*phi1) + 20020*a1**5*np.sin((9*ang0*k0)/2. + 3*phi1) - 72072*a0**2*a1**3*np.sin((5*ang0*k0)/2. - 2*phi0 + 3*phi1) + 138600*a0**2*a1**3*np.sin((13*ang0*k0)/2. + 2*phi0 + 3*phi1) + 18018*a0*a1**4*np.sin(5*ang0*k0 - phi0 + 4*phi1) + 64350*a0*a1**4*np.sin(7*ang0*k0 + phi0 + 4*phi1) + 12012*a1**5*np.sin((15*ang0*k0)/2. + 5*phi1))/(1.44144e6*k0))
                res1 = (-(-180180*a0**3*a1**2*ang1*k0*np.cos(3*phi0 - 2*phi1) - 180180*a0*(a0**4 + 6*a0**2*a1**2 + 3*a1**4)*np.sin(ang1*k0 + phi0) + 30030*a0**5*np.sin(3*(ang1*k0 + phi0)) + 120120*a0**3*a1**2*np.sin(3*(ang1*k0 + phi0)) + 18018*a0**5*np.sin(5*(ang1*k0 + phi0)) - 720720*a0**4*a1*np.sin((ang1*k0)/2. + 2*phi0 - phi1) - 1081080*a0**2*a1**3*np.sin((ang1*k0)/2. + 2*phi0 - phi1) + 36036*a0**4*a1*np.sin((5*ang1*k0)/2. + 4*phi0 - phi1) - 360360*a0**4*a1*np.sin((3*ang1*k0)/2. + phi1) - 720720*a0**2*a1**3*np.sin((3*ang1*k0)/2. + phi1) - 120120*a1**5*np.sin((3*ang1*k0)/2. + phi1) + 102960*a0**4*a1*np.sin((7*ang1*k0)/2. + 2*phi0 + phi1) + 154440*a0**2*a1**3*np.sin((7*ang1*k0)/2. + 2*phi0 + phi1) + 81900*a0**4*a1*np.sin((11*ang1*k0)/2. + 4*phi0 + phi1) - 270270*a0**3*a1**2*np.sin(2*ang1*k0 - phi0 + 2*phi1) - 180180*a0*a1**4*np.sin(2*ang1*k0 - phi0 + 2*phi1) + 135135*a0**3*a1**2*np.sin(4*ang1*k0 + phi0 + 2*phi1) + 90090*a0*a1**4*np.sin(4*ang1*k0 + phi0 + 2*phi1) + 150150*a0**3*a1**2*np.sin(6*ang1*k0 + 3*phi0 + 2*phi1) + 80080*a0**2*a1**3*np.sin((9*ang1*k0)/2. + 3*phi1) + 20020*a1**5*np.sin((9*ang1*k0)/2. + 3*phi1) - 72072*a0**2*a1**3*np.sin((5*ang1*k0)/2. - 2*phi0 + 3*phi1) + 138600*a0**2*a1**3*np.sin((13*ang1*k0)/2. + 2*phi0 + 3*phi1) + 18018*a0*a1**4*np.sin(5*ang1*k0 - phi0 + 4*phi1) + 64350*a0*a1**4*np.sin(7*ang1*k0 + phi0 + 4*phi1) + 12012*a1**5*np.sin((15*ang1*k0)/2. + 5*phi1))/(1.44144e6*k0))
                return res1 - res0
            elif 2*k0 == k1:
                res0 = (-(-2520*a0**2*a1*(2*a0**2 + 3*a1**2)*ang0*k0*np.cos(2*phi0 - phi1) - 2520*a0*(a0**4 + 6*a0**2*a1**2 + 3*a1**4)*np.sin(ang0*k0 + phi0) + 420*a0**5*np.sin(3*(ang0*k0 + phi0)) + 1680*a0**3*a1**2*np.sin(3*(ang0*k0 + phi0)) + 252*a0**5*np.sin(5*(ang0*k0 + phi0)) + 630*a0**4*a1*np.sin(2*ang0*k0 + 4*phi0 - phi1) - 3780*a0**4*a1*np.sin(2*ang0*k0 + phi1) - 7560*a0**2*a1**3*np.sin(2*ang0*k0 + phi1) - 1260*a1**5*np.sin(2*ang0*k0 + phi1) + 840*a0**2*a1**3*np.sin(3*(2*ang0*k0 + phi1)) + 210*a1**5*np.sin(3*(2*ang0*k0 + phi1)) + 126*a1**5*np.sin(5*(2*ang0*k0 + phi1)) + 1260*a0**4*a1*np.sin(4*ang0*k0 + 2*phi0 + phi1) + 1890*a0**2*a1**3*np.sin(4*ang0*k0 + 2*phi0 + phi1) + 1050*a0**4*a1*np.sin(6*ang0*k0 + 4*phi0 + phi1) - 2520*a0**3*a1**2*np.sin(ang0*k0 - 3*phi0 + 2*phi1) - 2520*a0**3*a1**2*np.sin(3*ang0*k0 - phi0 + 2*phi1) - 1680*a0*a1**4*np.sin(3*ang0*k0 - phi0 + 2*phi1) + 1512*a0**3*a1**2*np.sin(5*ang0*k0 + phi0 + 2*phi1) + 1008*a0*a1**4*np.sin(5*ang0*k0 + phi0 + 2*phi1) + 1800*a0**3*a1**2*np.sin(7*ang0*k0 + 3*phi0 + 2*phi1) - 630*a0**2*a1**3*np.sin(4*ang0*k0 - 2*phi0 + 3*phi1) + 1575*a0**2*a1**3*np.sin(8*ang0*k0 + 2*phi0 + 3*phi1) + 180*a0*a1**4*np.sin(7*ang0*k0 - phi0 + 4*phi1) + 700*a0*a1**4*np.sin(9*ang0*k0 + phi0 + 4*phi1))/(20160.*k0))
                res1 = (-(-2520*a0**2*a1*(2*a0**2 + 3*a1**2)*ang1*k0*np.cos(2*phi0 - phi1) - 2520*a0*(a0**4 + 6*a0**2*a1**2 + 3*a1**4)*np.sin(ang1*k0 + phi0) + 420*a0**5*np.sin(3*(ang1*k0 + phi0)) + 1680*a0**3*a1**2*np.sin(3*(ang1*k0 + phi0)) + 252*a0**5*np.sin(5*(ang1*k0 + phi0)) + 630*a0**4*a1*np.sin(2*ang1*k0 + 4*phi0 - phi1) - 3780*a0**4*a1*np.sin(2*ang1*k0 + phi1) - 7560*a0**2*a1**3*np.sin(2*ang1*k0 + phi1) - 1260*a1**5*np.sin(2*ang1*k0 + phi1) + 840*a0**2*a1**3*np.sin(3*(2*ang1*k0 + phi1)) + 210*a1**5*np.sin(3*(2*ang1*k0 + phi1)) + 126*a1**5*np.sin(5*(2*ang1*k0 + phi1)) + 1260*a0**4*a1*np.sin(4*ang1*k0 + 2*phi0 + phi1) + 1890*a0**2*a1**3*np.sin(4*ang1*k0 + 2*phi0 + phi1) + 1050*a0**4*a1*np.sin(6*ang1*k0 + 4*phi0 + phi1) - 2520*a0**3*a1**2*np.sin(ang1*k0 - 3*phi0 + 2*phi1) - 2520*a0**3*a1**2*np.sin(3*ang1*k0 - phi0 + 2*phi1) - 1680*a0*a1**4*np.sin(3*ang1*k0 - phi0 + 2*phi1) + 1512*a0**3*a1**2*np.sin(5*ang1*k0 + phi0 + 2*phi1) + 1008*a0*a1**4*np.sin(5*ang1*k0 + phi0 + 2*phi1) + 1800*a0**3*a1**2*np.sin(7*ang1*k0 + 3*phi0 + 2*phi1) - 630*a0**2*a1**3*np.sin(4*ang1*k0 - 2*phi0 + 3*phi1) + 1575*a0**2*a1**3*np.sin(8*ang1*k0 + 2*phi0 + 3*phi1) + 180*a0*a1**4*np.sin(7*ang1*k0 - phi0 + 4*phi1) + 700*a0*a1**4*np.sin(9*ang1*k0 + phi0 + 4*phi1))/(20160.*k0))
                return res1 - res0
            elif 4*k0 == k1:
                res0 = (-(471240*a0**4*a1*ang0*k0*np.cos(4*phi0 - phi1) - 942480*a0*(a0**4 + 6*a0**2*a1**2 + 3*a1**4)*np.sin(ang0*k0 + phi0) + 157080*a0**5*np.sin(3*(ang0*k0 + phi0)) + 628320*a0**3*a1**2*np.sin(3*(ang0*k0 + phi0)) + 94248*a0**5*np.sin(5*(ang0*k0 + phi0)) - 706860*a0**4*a1*np.sin(4*ang0*k0 + phi1) - 1413720*a0**2*a1**3*np.sin(4*ang0*k0 + phi1) - 235620*a1**5*np.sin(4*ang0*k0 + phi1) + 157080*a0**2*a1**3*np.sin(3*(4*ang0*k0 + phi1)) + 39270*a1**5*np.sin(3*(4*ang0*k0 + phi1)) + 23562*a1**5*np.sin(5*(4*ang0*k0 + phi1)) - 942480*a0**4*a1*np.sin(2*ang0*k0 - 2*phi0 + phi1) - 1413720*a0**2*a1**3*np.sin(2*ang0*k0 - 2*phi0 + phi1) + 314160*a0**4*a1*np.sin(6*ang0*k0 + 2*phi0 + phi1) + 471240*a0**2*a1**3*np.sin(6*ang0*k0 + 2*phi0 + phi1) + 294525*a0**4*a1*np.sin(8*ang0*k0 + 4*phi0 + phi1) - 188496*a0**3*a1**2*np.sin(5*ang0*k0 - 3*phi0 + 2*phi1) - 403920*a0**3*a1**2*np.sin(7*ang0*k0 - phi0 + 2*phi1) - 269280*a0*a1**4*np.sin(7*ang0*k0 - phi0 + 2*phi1) + 314160*a0**3*a1**2*np.sin(9*ang0*k0 + phi0 + 2*phi1) + 209440*a0*a1**4*np.sin(9*ang0*k0 + phi0 + 2*phi1) + 428400*a0**3*a1**2*np.sin(11*ang0*k0 + 3*phi0 + 2*phi1) - 94248*a0**2*a1**3*np.sin(10*ang0*k0 - 2*phi0 + 3*phi1) + 336600*a0**2*a1**3*np.sin(14*ang0*k0 + 2*phi0 + 3*phi1) + 31416*a0*a1**4*np.sin(15*ang0*k0 - phi0 + 4*phi1) + 138600*a0*a1**4*np.sin(17*ang0*k0 + phi0 + 4*phi1))/(7.53984e6*k0))
                res1 = (-(471240*a0**4*a1*ang1*k0*np.cos(4*phi0 - phi1) - 942480*a0*(a0**4 + 6*a0**2*a1**2 + 3*a1**4)*np.sin(ang1*k0 + phi0) + 157080*a0**5*np.sin(3*(ang1*k0 + phi0)) + 628320*a0**3*a1**2*np.sin(3*(ang1*k0 + phi0)) + 94248*a0**5*np.sin(5*(ang1*k0 + phi0)) - 706860*a0**4*a1*np.sin(4*ang1*k0 + phi1) - 1413720*a0**2*a1**3*np.sin(4*ang1*k0 + phi1) - 235620*a1**5*np.sin(4*ang1*k0 + phi1) + 157080*a0**2*a1**3*np.sin(3*(4*ang1*k0 + phi1)) + 39270*a1**5*np.sin(3*(4*ang1*k0 + phi1)) + 23562*a1**5*np.sin(5*(4*ang1*k0 + phi1)) - 942480*a0**4*a1*np.sin(2*ang1*k0 - 2*phi0 + phi1) - 1413720*a0**2*a1**3*np.sin(2*ang1*k0 - 2*phi0 + phi1) + 314160*a0**4*a1*np.sin(6*ang1*k0 + 2*phi0 + phi1) + 471240*a0**2*a1**3*np.sin(6*ang1*k0 + 2*phi0 + phi1) + 294525*a0**4*a1*np.sin(8*ang1*k0 + 4*phi0 + phi1) - 188496*a0**3*a1**2*np.sin(5*ang1*k0 - 3*phi0 + 2*phi1) - 403920*a0**3*a1**2*np.sin(7*ang1*k0 - phi0 + 2*phi1) - 269280*a0*a1**4*np.sin(7*ang1*k0 - phi0 + 2*phi1) + 314160*a0**3*a1**2*np.sin(9*ang1*k0 + phi0 + 2*phi1) + 209440*a0*a1**4*np.sin(9*ang1*k0 + phi0 + 2*phi1) + 428400*a0**3*a1**2*np.sin(11*ang1*k0 + 3*phi0 + 2*phi1) - 94248*a0**2*a1**3*np.sin(10*ang1*k0 - 2*phi0 + 3*phi1) + 336600*a0**2*a1**3*np.sin(14*ang1*k0 + 2*phi0 + 3*phi1) + 31416*a0*a1**4*np.sin(15*ang1*k0 - phi0 + 4*phi1) + 138600*a0*a1**4*np.sin(17*ang1*k0 + phi0 + 4*phi1))/(7.53984e6*k0))
                return res1 - res0
            else:
                res0 = (((30*a0*(a0**4 + 6*a0**2*a1**2 + 3*a1**4)*np.sin(ang0*k0 + phi0))/k0 - (5*(a0**5 + 4*a0**3*a1**2)*np.sin(3*(ang0*k0 + phi0)))/k0 - (3*a0**5*np.sin(5*(ang0*k0 + phi0)))/k0 - (15*a0*a1**4*np.sin(ang0*k0 - 4*ang0*k1 + phi0 - 4*phi1))/(k0 - 4*k1) + (30*a0**2*a1**3*np.sin(2*ang0*k0 - 3*ang0*k1 + 2*phi0 - 3*phi1))/(2*k0 - 3*k1) + (90*a0**3*a1**2*np.sin(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1))/(k0 - 2*k1) + (60*a0*a1**4*np.sin(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1))/(k0 - 2*k1) + (30*a0**3*a1**2*np.sin(3*ang0*k0 - 2*ang0*k1 + 3*phi0 - 2*phi1))/(3*k0 - 2*k1) + (60*a0**4*a1*np.sin(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1))/(2*k0 - k1) + (90*a0**2*a1**3*np.sin(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1))/(2*k0 - k1) - (15*a0**4*a1*np.sin(4*ang0*k0 - ang0*k1 + 4*phi0 - phi1))/(4*k0 - k1) + (90*a0**4*a1*np.sin(ang0*k1 + phi1))/k1 + (180*a0**2*a1**3*np.sin(ang0*k1 + phi1))/k1 + (30*a1**5*np.sin(ang0*k1 + phi1))/k1 - (20*a0**2*a1**3*np.sin(3*(ang0*k1 + phi1)))/k1 - (5*a1**5*np.sin(3*(ang0*k1 + phi1)))/k1 - (3*a1**5*np.sin(5*(ang0*k1 + phi1)))/k1 - (60*a0**4*a1*np.sin(2*ang0*k0 + ang0*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (90*a0**2*a1**3*np.sin(2*ang0*k0 + ang0*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (75*a0**4*a1*np.sin(4*ang0*k0 + ang0*k1 + 4*phi0 + phi1))/(4*k0 + k1) - (90*a0**3*a1**2*np.sin(ang0*k0 + 2*ang0*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (60*a0*a1**4*np.sin(ang0*k0 + 2*ang0*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (150*a0**3*a1**2*np.sin(3*ang0*k0 + 2*ang0*k1 + 3*phi0 + 2*phi1))/(3*k0 + 2*k1) - (150*a0**2*a1**3*np.sin(2*ang0*k0 + 3*ang0*k1 + 2*phi0 + 3*phi1))/(2*k0 + 3*k1) - (75*a0*a1**4*np.sin(ang0*k0 + 4*ang0*k1 + phi0 + 4*phi1))/(k0 + 4*k1))/240.)
                res1 = (((30*a0*(a0**4 + 6*a0**2*a1**2 + 3*a1**4)*np.sin(ang1*k0 + phi0))/k0 - (5*(a0**5 + 4*a0**3*a1**2)*np.sin(3*(ang1*k0 + phi0)))/k0 - (3*a0**5*np.sin(5*(ang1*k0 + phi0)))/k0 - (15*a0*a1**4*np.sin(ang1*k0 - 4*ang1*k1 + phi0 - 4*phi1))/(k0 - 4*k1) + (30*a0**2*a1**3*np.sin(2*ang1*k0 - 3*ang1*k1 + 2*phi0 - 3*phi1))/(2*k0 - 3*k1) + (90*a0**3*a1**2*np.sin(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1))/(k0 - 2*k1) + (60*a0*a1**4*np.sin(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1))/(k0 - 2*k1) + (30*a0**3*a1**2*np.sin(3*ang1*k0 - 2*ang1*k1 + 3*phi0 - 2*phi1))/(3*k0 - 2*k1) + (60*a0**4*a1*np.sin(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1))/(2*k0 - k1) + (90*a0**2*a1**3*np.sin(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1))/(2*k0 - k1) - (15*a0**4*a1*np.sin(4*ang1*k0 - ang1*k1 + 4*phi0 - phi1))/(4*k0 - k1) + (90*a0**4*a1*np.sin(ang1*k1 + phi1))/k1 + (180*a0**2*a1**3*np.sin(ang1*k1 + phi1))/k1 + (30*a1**5*np.sin(ang1*k1 + phi1))/k1 - (20*a0**2*a1**3*np.sin(3*(ang1*k1 + phi1)))/k1 - (5*a1**5*np.sin(3*(ang1*k1 + phi1)))/k1 - (3*a1**5*np.sin(5*(ang1*k1 + phi1)))/k1 - (60*a0**4*a1*np.sin(2*ang1*k0 + ang1*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (90*a0**2*a1**3*np.sin(2*ang1*k0 + ang1*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (75*a0**4*a1*np.sin(4*ang1*k0 + ang1*k1 + 4*phi0 + phi1))/(4*k0 + k1) - (90*a0**3*a1**2*np.sin(ang1*k0 + 2*ang1*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (60*a0*a1**4*np.sin(ang1*k0 + 2*ang1*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (150*a0**3*a1**2*np.sin(3*ang1*k0 + 2*ang1*k1 + 3*phi0 + 2*phi1))/(3*k0 + 2*k1) - (150*a0**2*a1**3*np.sin(2*ang1*k0 + 3*ang1*k1 + 2*phi0 + 3*phi1))/(2*k0 + 3*k1) - (75*a0*a1**4*np.sin(ang1*k0 + 4*ang1*k1 + phi0 + 4*phi1))/(k0 + 4*k1))/240.)
                return res1 - res0
        elif j == 3:
            if k0 == 3*k1:
                print('case of k0 == 3*k1 has not implemented yet')
                return 0.0
            elif k0 == 2*k1:
                print('case of k0 == 2*k1 has not implemented yet')
                return 0.0
            elif k0 == -k1:
                print('case of k0 == -k1 has not implemented yet')
                return 0.0
            elif k0 == -2*k1:
                print('case of k0 == -2*k1 has not implemented yet')
                return 0.0
            elif 2*k0 == -k1:
                print('case of 2*k0 == -k1 has not implemented yet')
                return 0.0
            elif 5*k0 == -k1:
                print('case of 5*k0 == -k1 has not implemented yet')
                return 0.0
            elif k0 == -5*k1:
                print('case of k0 == -4*k1 has not implemented yet')
                return 0.0
            elif 2*k0 == k1:
                res0 = (-(41580*(a0**6 + 8*a0**4*a1**2 + 6*a0**2*a1**4)*np.cos(2*(ang0*k0 + phi0)) - 4620*a0**6*np.cos(6*(ang0*k0 + phi0)) + a1*(166320*a0**3*(a0**2 + 2*a1**2)*np.cos(ang0*k0 + 3*phi0 - phi1) + 20790*a1*(6*a0**4 + 8*a0**2*a1**2 + a1**4)*np.cos(2*(2*ang0*k0 + phi1)) - 2310*a1**5*np.cos(6*(2*ang0*k0 + phi1)) + 110880*a0**5*np.cos(3*ang0*k0 + phi0 + phi1) + 332640*a0**3*a1**2*np.cos(3*ang0*k0 + phi0 + phi1) + 110880*a0*a1**4*np.cos(3*ang0*k0 + phi0 + phi1) - 61600*a0**3*a1**2*np.cos(3*(3*ang0*k0 + phi0 + phi1)) - 51975*a0**4*a1*np.cos(2*(4*ang0*k0 + 2*phi0 + phi1)) - 23760*a0**5*np.cos(7*ang0*k0 + 5*phi0 + phi1) - 41580*a0**2*a1**3*np.cos(2*(5*ang0*k0 + phi0 + 2*phi1)) + 66528*a0**3*a1**2*np.cos(5*ang0*k0 - phi0 + 3*phi1) + 33264*a0*a1**4*np.cos(5*ang0*k0 - phi0 + 3*phi1) + 13860*a0**2*a1**3*np.cos(6*ang0*k0 - 2*phi0 + 4*phi1) - 15120*a0*a1**4*np.cos(11*ang0*k0 + phi0 + 5*phi1) - 83160*a0**4*a1*ang0*k0*np.sin(4*phi0 - 2*phi1)))/(887040.*k0))
                res1 = (-(41580*(a0**6 + 8*a0**4*a1**2 + 6*a0**2*a1**4)*np.cos(2*(ang1*k0 + phi0)) - 4620*a0**6*np.cos(6*(ang1*k0 + phi0)) + a1*(166320*a0**3*(a0**2 + 2*a1**2)*np.cos(ang1*k0 + 3*phi0 - phi1) + 20790*a1*(6*a0**4 + 8*a0**2*a1**2 + a1**4)*np.cos(2*(2*ang1*k0 + phi1)) - 2310*a1**5*np.cos(6*(2*ang1*k0 + phi1)) + 110880*a0**5*np.cos(3*ang1*k0 + phi0 + phi1) + 332640*a0**3*a1**2*np.cos(3*ang1*k0 + phi0 + phi1) + 110880*a0*a1**4*np.cos(3*ang1*k0 + phi0 + phi1) - 61600*a0**3*a1**2*np.cos(3*(3*ang1*k0 + phi0 + phi1)) - 51975*a0**4*a1*np.cos(2*(4*ang1*k0 + 2*phi0 + phi1)) - 23760*a0**5*np.cos(7*ang1*k0 + 5*phi0 + phi1) - 41580*a0**2*a1**3*np.cos(2*(5*ang1*k0 + phi0 + 2*phi1)) + 66528*a0**3*a1**2*np.cos(5*ang1*k0 - phi0 + 3*phi1) + 33264*a0*a1**4*np.cos(5*ang1*k0 - phi0 + 3*phi1) + 13860*a0**2*a1**3*np.cos(6*ang1*k0 - 2*phi0 + 4*phi1) - 15120*a0*a1**4*np.cos(11*ang1*k0 + phi0 + 5*phi1) - 83160*a0**4*a1*ang1*k0*np.sin(4*phi0 - 2*phi1)))/(887040.*k0))
                return res1 - res0
            elif 3*k0 == k1:
                res0 = ((-3780*(a0**6 + 8*a0**4*a1**2 + 6*a0**2*a1**4)*np.cos(2*(ang0*k0 + phi0)) + 420*a0**6*np.cos(6*(ang0*k0 + phi0)) + a1*(-1260*a1*(6*a0**4 + 8*a0**2*a1**2 + a1**4)*np.cos(2*(3*ang0*k0 + phi1)) + 140*a1**5*np.cos(6*(3*ang0*k0 + phi1)) + 3*a0*(1260*a0**3*a1*np.cos(2*(ang0*k0 - 2*phi0 + phi1)) - 2520*(a0**4 + 3*a0**2*a1**2 + a1**4)*np.cos(4*ang0*k0 + phi0 + phi1) + 1400*a0**2*a1**2*np.cos(3*(4*ang0*k0 + phi0 + phi1)) + 1260*a0**3*a1*np.cos(2*(5*ang0*k0 + 2*phi0 + phi1)) + 630*a0**4*np.cos(8*ang0*k0 + 5*phi0 + phi1) + 900*a0*a1**3*np.cos(2*(7*ang0*k0 + phi0 + 2*phi1)) - 1260*a0**2*a1**2*np.cos(8*ang0*k0 - phi0 + 3*phi1) - 630*a1**4*np.cos(8*ang0*k0 - phi0 + 3*phi1) - 252*a0*a1**3*np.cos(10*ang0*k0 - 2*phi0 + 4*phi1) + 315*a1**4*np.cos(16*ang0*k0 + phi0 + 5*phi1) + 5040*a0**4*ang0*k0*np.sin(3*phi0 - phi1) + 10080*a0**2*a1**2*ang0*k0*np.sin(3*phi0 - phi1))))/(80640.*k0))
                res1 = ((-3780*(a0**6 + 8*a0**4*a1**2 + 6*a0**2*a1**4)*np.cos(2*(ang1*k0 + phi0)) + 420*a0**6*np.cos(6*(ang1*k0 + phi0)) + a1*(-1260*a1*(6*a0**4 + 8*a0**2*a1**2 + a1**4)*np.cos(2*(3*ang1*k0 + phi1)) + 140*a1**5*np.cos(6*(3*ang1*k0 + phi1)) + 3*a0*(1260*a0**3*a1*np.cos(2*(ang1*k0 - 2*phi0 + phi1)) - 2520*(a0**4 + 3*a0**2*a1**2 + a1**4)*np.cos(4*ang1*k0 + phi0 + phi1) + 1400*a0**2*a1**2*np.cos(3*(4*ang1*k0 + phi0 + phi1)) + 1260*a0**3*a1*np.cos(2*(5*ang1*k0 + 2*phi0 + phi1)) + 630*a0**4*np.cos(8*ang1*k0 + 5*phi0 + phi1) + 900*a0*a1**3*np.cos(2*(7*ang1*k0 + phi0 + 2*phi1)) - 1260*a0**2*a1**2*np.cos(8*ang1*k0 - phi0 + 3*phi1) - 630*a1**4*np.cos(8*ang1*k0 - phi0 + 3*phi1) - 252*a0*a1**3*np.cos(10*ang1*k0 - 2*phi0 + 4*phi1) + 315*a1**4*np.cos(16*ang1*k0 + phi0 + 5*phi1) + 5040*a0**4*ang1*k0*np.sin(3*phi0 - phi1) + 10080*a0**2*a1**2*ang1*k0*np.sin(3*phi0 - phi1))))/(80640.*k0))
                return res1 - res0
            else:
                res0 = ((-9*a0**2*(a0**4 + 8*a0**2*a1**2 + 6*a1**4)*k1*(60*k0**9 + 172*k0**8*k1 - 1063*k0**7*k1**2 - 1539*k0**6*k1**3 + 3666*k0**5*k1**4 + 3666*k0**4*k1**5 - 1539*k0**3*k1**6 - 1063*k0**2*k1**7 + 172*k0*k1**8 + 60*k1**9)*np.cos(2*(ang0*k0 + phi0)) + a0**6*k1*(60*k0**9 + 172*k0**8*k1 - 1063*k0**7*k1**2 - 1539*k0**6*k1**3 + 3666*k0**5*k1**4 + 3666*k0**4*k1**5 - 1539*k0**3*k1**6 - 1063*k0**2*k1**7 + 172*k0*k1**8 + 60*k1**9)*np.cos(6*(ang0*k0 + phi0)) - a1*k0*(36*a0*a1**2*(2*a0**2 + a1**2)*k1*(-60*k0**8 - 352*k0**7*k1 + 7*k0**6*k1**2 + 1560*k0**5*k1**3 + 1014*k0**4*k1**4 - 624*k0**3*k1**5 - 333*k0**2*k1**6 + 64*k0*k1**7 + 20*k1**8)*np.cos(ang0*(k0 - 3*k1) + phi0 - 3*phi1) + (k0 - 3*k1)*(-9*a0**2*a1**3*k1*(60*k0**7 + 472*k0**6*k1 + 937*k0**5*k1**2 + 314*k0**4*k1**3 - 386*k0**3*k1**4 - 148*k0**2*k1**5 + 37*k0*k1**6 + 10*k1**7)*np.cos(2*(ang0*(k0 - 2*k1) + phi0 - 2*phi1)) + (k0 - 2*k1)*(9*a0**4*a1*k1*(30*k0**6 + 251*k0**5*k1 + 594*k0**4*k1**2 + 454*k0**3*k1**3 + 34*k0**2*k1**4 - 57*k0*k1**5 - 10*k1**6)*np.cos(4*ang0*k0 - 2*ang0*k1 + 4*phi0 - 2*phi1) + (2*k0 - k1)*(36*a0**3*(a0**2 + 2*a1**2)*k1*(10*k0**5 + 87*k0**4*k1 + 227*k0**3*k1**2 + 227*k0**2*k1**3 + 87*k0*k1**4 + 10*k1**5)*np.cos(3*ang0*k0 - ang0*k1 + 3*phi0 - phi1) + (3*k0 - k1)*(9*a1*(6*a0**4 + 8*a0**2*a1**2 + a1**4)*(10*k0**5 + 87*k0**4*k1 + 227*k0**3*k1**2 + 227*k0**2*k1**3 + 87*k0*k1**4 + 10*k1**5)*np.cos(2*(ang0*k1 + phi1)) - a1**5*(10*k0**5 + 87*k0**4*k1 + 227*k0**3*k1**2 + 227*k0**2*k1**3 + 87*k0*k1**4 + 10*k1**5)*np.cos(6*(ang0*k1 + phi1)) + a0*k1*(72*(a0**4 + 3*a0**2*a1**2 + a1**4)*(10*k0**4 + 77*k0**3*k1 + 150*k0**2*k1**2 + 77*k0*k1**3 + 10*k1**4)*np.cos(ang0*(k0 + k1) + phi0 + phi1) - 40*a0**2*a1**2*(10*k0**4 + 77*k0**3*k1 + 150*k0**2*k1**2 + 77*k0*k1**3 + 10*k1**4)*np.cos(3*(ang0*(k0 + k1) + phi0 + phi1)) - 9*(k0 + k1)*(5*a0**3*a1*(5*k0**3 + 36*k0**2*k1 + 57*k0*k1**2 + 10*k1**3)*np.cos(2*(ang0*(2*k0 + k1) + 2*phi0 + phi1)) + (2*k0 + k1)*(4*a0**4*(k0**2 + 7*k0*k1 + 10*k1**2)*np.cos(ang0*(5*k0 + k1) + 5*phi0 + phi1) + a1**3*(5*k0 + k1)*(5*a0*(k0 + 5*k1)*np.cos(2*(ang0*(k0 + 2*k1) + phi0 + 2*phi1)) + 4*a1*(k0 + 2*k1)*np.cos(ang0*(k0 + 5*k1) + phi0 + 5*phi1)))))))))))/(192.*k0*(k0 - 3*k1)*(k0 - 2*k1)*(2*k0 - k1)*(3*k0 - k1)*k1*(k0 + k1)*(2*k0 + k1)*(5*k0 + k1)*(k0 + 2*k1)*(k0 + 5*k1)))
                res1 = ((-9*a0**2*(a0**4 + 8*a0**2*a1**2 + 6*a1**4)*k1*(60*k0**9 + 172*k0**8*k1 - 1063*k0**7*k1**2 - 1539*k0**6*k1**3 + 3666*k0**5*k1**4 + 3666*k0**4*k1**5 - 1539*k0**3*k1**6 - 1063*k0**2*k1**7 + 172*k0*k1**8 + 60*k1**9)*np.cos(2*(ang1*k0 + phi0)) + a0**6*k1*(60*k0**9 + 172*k0**8*k1 - 1063*k0**7*k1**2 - 1539*k0**6*k1**3 + 3666*k0**5*k1**4 + 3666*k0**4*k1**5 - 1539*k0**3*k1**6 - 1063*k0**2*k1**7 + 172*k0*k1**8 + 60*k1**9)*np.cos(6*(ang1*k0 + phi0)) - a1*k0*(36*a0*a1**2*(2*a0**2 + a1**2)*k1*(-60*k0**8 - 352*k0**7*k1 + 7*k0**6*k1**2 + 1560*k0**5*k1**3 + 1014*k0**4*k1**4 - 624*k0**3*k1**5 - 333*k0**2*k1**6 + 64*k0*k1**7 + 20*k1**8)*np.cos(ang1*(k0 - 3*k1) + phi0 - 3*phi1) + (k0 - 3*k1)*(-9*a0**2*a1**3*k1*(60*k0**7 + 472*k0**6*k1 + 937*k0**5*k1**2 + 314*k0**4*k1**3 - 386*k0**3*k1**4 - 148*k0**2*k1**5 + 37*k0*k1**6 + 10*k1**7)*np.cos(2*(ang1*(k0 - 2*k1) + phi0 - 2*phi1)) + (k0 - 2*k1)*(9*a0**4*a1*k1*(30*k0**6 + 251*k0**5*k1 + 594*k0**4*k1**2 + 454*k0**3*k1**3 + 34*k0**2*k1**4 - 57*k0*k1**5 - 10*k1**6)*np.cos(4*ang1*k0 - 2*ang1*k1 + 4*phi0 - 2*phi1) + (2*k0 - k1)*(36*a0**3*(a0**2 + 2*a1**2)*k1*(10*k0**5 + 87*k0**4*k1 + 227*k0**3*k1**2 + 227*k0**2*k1**3 + 87*k0*k1**4 + 10*k1**5)*np.cos(3*ang1*k0 - ang1*k1 + 3*phi0 - phi1) + (3*k0 - k1)*(9*a1*(6*a0**4 + 8*a0**2*a1**2 + a1**4)*(10*k0**5 + 87*k0**4*k1 + 227*k0**3*k1**2 + 227*k0**2*k1**3 + 87*k0*k1**4 + 10*k1**5)*np.cos(2*(ang1*k1 + phi1)) - a1**5*(10*k0**5 + 87*k0**4*k1 + 227*k0**3*k1**2 + 227*k0**2*k1**3 + 87*k0*k1**4 + 10*k1**5)*np.cos(6*(ang1*k1 + phi1)) + a0*k1*(72*(a0**4 + 3*a0**2*a1**2 + a1**4)*(10*k0**4 + 77*k0**3*k1 + 150*k0**2*k1**2 + 77*k0*k1**3 + 10*k1**4)*np.cos(ang1*(k0 + k1) + phi0 + phi1) - 40*a0**2*a1**2*(10*k0**4 + 77*k0**3*k1 + 150*k0**2*k1**2 + 77*k0*k1**3 + 10*k1**4)*np.cos(3*(ang1*(k0 + k1) + phi0 + phi1)) - 9*(k0 + k1)*(5*a0**3*a1*(5*k0**3 + 36*k0**2*k1 + 57*k0*k1**2 + 10*k1**3)*np.cos(2*(ang1*(2*k0 + k1) + 2*phi0 + phi1)) + (2*k0 + k1)*(4*a0**4*(k0**2 + 7*k0*k1 + 10*k1**2)*np.cos(ang1*(5*k0 + k1) + 5*phi0 + phi1) + a1**3*(5*k0 + k1)*(5*a0*(k0 + 5*k1)*np.cos(2*(ang1*(k0 + 2*k1) + phi0 + 2*phi1)) + 4*a1*(k0 + 2*k1)*np.cos(ang1*(k0 + 5*k1) + phi0 + 5*phi1)))))))))))/(192.*k0*(k0 - 3*k1)*(k0 - 2*k1)*(2*k0 - k1)*(3*k0 - k1)*k1*(k0 + k1)*(2*k0 + k1)*(5*k0 + k1)*(k0 + 2*k1)*(k0 + 5*k1)))
                return res1 - res0
        elif j == 4:
            res0 = (((105*a0*(a0**6 + 12*a0**4*a1**2 + 18*a0**2*a1**4 + 4*a1**6)*np.sin(ang0*k0 + phi0))/k0 - (35*(a0**7 + 10*a0**5*a1**2 + 10*a0**3*a1**4)*np.sin(3*(ang0*k0 + phi0)))/k0 - (7*a0**7*np.sin(5*(ang0*k0 + phi0)))/k0 - (42*a0**5*a1**2*np.sin(5*(ang0*k0 + phi0)))/k0 + (5*a0**7*np.sin(7*(ang0*k0 + phi0)))/k0 - (35*a0*a1**6*np.sin(ang0*k0 - 6*ang0*k1 + phi0 - 6*phi1))/(k0 - 6*k1) - (105*a0**2*a1**5*np.sin(2*ang0*k0 - 5*ang0*k1 + 2*phi0 - 5*phi1))/(2*k0 - 5*k1) - (525*a0**3*a1**4*np.sin(ang0*k0 - 4*ang0*k1 + phi0 - 4*phi1))/(k0 - 4*k1) - (210*a0*a1**6*np.sin(ang0*k0 - 4*ang0*k1 + phi0 - 4*phi1))/(k0 - 4*k1) + (105*a0**3*a1**4*np.sin(3*ang0*k0 - 4*ang0*k1 + 3*phi0 - 4*phi1))/(3*k0 - 4*k1) + (420*a0**4*a1**3*np.sin(2*ang0*k0 - 3*ang0*k1 + 2*phi0 - 3*phi1))/(2*k0 - 3*k1) + (315*a0**2*a1**5*np.sin(2*ang0*k0 - 3*ang0*k1 + 2*phi0 - 3*phi1))/(2*k0 - 3*k1) + (105*a0**4*a1**3*np.sin(4*ang0*k0 - 3*ang0*k1 + 4*phi0 - 3*phi1))/(4*k0 - 3*k1) + (630*a0**5*a1**2*np.sin(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1))/(k0 - 2*k1) + (1260*a0**3*a1**4*np.sin(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1))/(k0 - 2*k1) + (315*a0*a1**6*np.sin(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1))/(k0 - 2*k1) + (315*a0**5*a1**2*np.sin(3*ang0*k0 - 2*ang0*k1 + 3*phi0 - 2*phi1))/(3*k0 - 2*k1) + (420*a0**3*a1**4*np.sin(3*ang0*k0 - 2*ang0*k1 + 3*phi0 - 2*phi1))/(3*k0 - 2*k1) - (105*a0**5*a1**2*np.sin(5*ang0*k0 - 2*ang0*k1 + 5*phi0 - 2*phi1))/(5*k0 - 2*k1) + (315*a0**6*a1*np.sin(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1))/(2*k0 - k1) + (1260*a0**4*a1**3*np.sin(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1))/(2*k0 - k1) + (630*a0**2*a1**5*np.sin(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1))/(2*k0 - k1) - (210*a0**6*a1*np.sin(4*ang0*k0 - ang0*k1 + 4*phi0 - phi1))/(4*k0 - k1) - (525*a0**4*a1**3*np.sin(4*ang0*k0 - ang0*k1 + 4*phi0 - phi1))/(4*k0 - k1) - (35*a0**6*a1*np.sin(6*ang0*k0 - ang0*k1 + 6*phi0 - phi1))/(6*k0 - k1) + (420*a0**6*a1*np.sin(ang0*k1 + phi1))/k1 + (1890*a0**4*a1**3*np.sin(ang0*k1 + phi1))/k1 + (1260*a0**2*a1**5*np.sin(ang0*k1 + phi1))/k1 + (105*a1**7*np.sin(ang0*k1 + phi1))/k1 - (350*a0**4*a1**3*np.sin(3*(ang0*k1 + phi1)))/k1 - (350*a0**2*a1**5*np.sin(3*(ang0*k1 + phi1)))/k1 - (35*a1**7*np.sin(3*(ang0*k1 + phi1)))/k1 - (42*a0**2*a1**5*np.sin(5*(ang0*k1 + phi1)))/k1 - (7*a1**7*np.sin(5*(ang0*k1 + phi1)))/k1 + (5*a1**7*np.sin(7*(ang0*k1 + phi1)))/k1 - (525*a0**6*a1*np.sin(2*ang0*k0 + ang0*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (2100*a0**4*a1**3*np.sin(2*ang0*k0 + ang0*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (1050*a0**2*a1**5*np.sin(2*ang0*k0 + ang0*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (210*a0**6*a1*np.sin(4*ang0*k0 + ang0*k1 + 4*phi0 + phi1))/(4*k0 + k1) - (525*a0**4*a1**3*np.sin(4*ang0*k0 + ang0*k1 + 4*phi0 + phi1))/(4*k0 + k1) + (245*a0**6*a1*np.sin(6*ang0*k0 + ang0*k1 + 6*phi0 + phi1))/(6*k0 + k1) - (1050*a0**5*a1**2*np.sin(ang0*k0 + 2*ang0*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (2100*a0**3*a1**4*np.sin(ang0*k0 + 2*ang0*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (525*a0*a1**6*np.sin(ang0*k0 + 2*ang0*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (525*a0**5*a1**2*np.sin(3*ang0*k0 + 2*ang0*k1 + 3*phi0 + 2*phi1))/(3*k0 + 2*k1) - (700*a0**3*a1**4*np.sin(3*ang0*k0 + 2*ang0*k1 + 3*phi0 + 2*phi1))/(3*k0 + 2*k1) + (735*a0**5*a1**2*np.sin(5*ang0*k0 + 2*ang0*k1 + 5*phi0 + 2*phi1))/(5*k0 + 2*k1) - (700*a0**4*a1**3*np.sin(2*ang0*k0 + 3*ang0*k1 + 2*phi0 + 3*phi1))/(2*k0 + 3*k1) - (525*a0**2*a1**5*np.sin(2*ang0*k0 + 3*ang0*k1 + 2*phi0 + 3*phi1))/(2*k0 + 3*k1) + (1225*a0**4*a1**3*np.sin(4*ang0*k0 + 3*ang0*k1 + 4*phi0 + 3*phi1))/(4*k0 + 3*k1) - (525*a0**3*a1**4*np.sin(ang0*k0 + 4*ang0*k1 + phi0 + 4*phi1))/(k0 + 4*k1) - (210*a0*a1**6*np.sin(ang0*k0 + 4*ang0*k1 + phi0 + 4*phi1))/(k0 + 4*k1) + (1225*a0**3*a1**4*np.sin(3*ang0*k0 + 4*ang0*k1 + 3*phi0 + 4*phi1))/(3*k0 + 4*k1) + (735*a0**2*a1**5*np.sin(2*ang0*k0 + 5*ang0*k1 + 2*phi0 + 5*phi1))/(2*k0 + 5*k1) + (245*a0*a1**6*np.sin(ang0*k0 + 6*ang0*k1 + phi0 + 6*phi1))/(k0 + 6*k1))/2240.)
            res1 = (((105*a0*(a0**6 + 12*a0**4*a1**2 + 18*a0**2*a1**4 + 4*a1**6)*np.sin(ang1*k0 + phi0))/k0 - (35*(a0**7 + 10*a0**5*a1**2 + 10*a0**3*a1**4)*np.sin(3*(ang1*k0 + phi0)))/k0 - (7*a0**7*np.sin(5*(ang1*k0 + phi0)))/k0 - (42*a0**5*a1**2*np.sin(5*(ang1*k0 + phi0)))/k0 + (5*a0**7*np.sin(7*(ang1*k0 + phi0)))/k0 - (35*a0*a1**6*np.sin(ang1*k0 - 6*ang1*k1 + phi0 - 6*phi1))/(k0 - 6*k1) - (105*a0**2*a1**5*np.sin(2*ang1*k0 - 5*ang1*k1 + 2*phi0 - 5*phi1))/(2*k0 - 5*k1) - (525*a0**3*a1**4*np.sin(ang1*k0 - 4*ang1*k1 + phi0 - 4*phi1))/(k0 - 4*k1) - (210*a0*a1**6*np.sin(ang1*k0 - 4*ang1*k1 + phi0 - 4*phi1))/(k0 - 4*k1) + (105*a0**3*a1**4*np.sin(3*ang1*k0 - 4*ang1*k1 + 3*phi0 - 4*phi1))/(3*k0 - 4*k1) + (420*a0**4*a1**3*np.sin(2*ang1*k0 - 3*ang1*k1 + 2*phi0 - 3*phi1))/(2*k0 - 3*k1) + (315*a0**2*a1**5*np.sin(2*ang1*k0 - 3*ang1*k1 + 2*phi0 - 3*phi1))/(2*k0 - 3*k1) + (105*a0**4*a1**3*np.sin(4*ang1*k0 - 3*ang1*k1 + 4*phi0 - 3*phi1))/(4*k0 - 3*k1) + (630*a0**5*a1**2*np.sin(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1))/(k0 - 2*k1) + (1260*a0**3*a1**4*np.sin(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1))/(k0 - 2*k1) + (315*a0*a1**6*np.sin(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1))/(k0 - 2*k1) + (315*a0**5*a1**2*np.sin(3*ang1*k0 - 2*ang1*k1 + 3*phi0 - 2*phi1))/(3*k0 - 2*k1) + (420*a0**3*a1**4*np.sin(3*ang1*k0 - 2*ang1*k1 + 3*phi0 - 2*phi1))/(3*k0 - 2*k1) - (105*a0**5*a1**2*np.sin(5*ang1*k0 - 2*ang1*k1 + 5*phi0 - 2*phi1))/(5*k0 - 2*k1) + (315*a0**6*a1*np.sin(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1))/(2*k0 - k1) + (1260*a0**4*a1**3*np.sin(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1))/(2*k0 - k1) + (630*a0**2*a1**5*np.sin(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1))/(2*k0 - k1) - (210*a0**6*a1*np.sin(4*ang1*k0 - ang1*k1 + 4*phi0 - phi1))/(4*k0 - k1) - (525*a0**4*a1**3*np.sin(4*ang1*k0 - ang1*k1 + 4*phi0 - phi1))/(4*k0 - k1) - (35*a0**6*a1*np.sin(6*ang1*k0 - ang1*k1 + 6*phi0 - phi1))/(6*k0 - k1) + (420*a0**6*a1*np.sin(ang1*k1 + phi1))/k1 + (1890*a0**4*a1**3*np.sin(ang1*k1 + phi1))/k1 + (1260*a0**2*a1**5*np.sin(ang1*k1 + phi1))/k1 + (105*a1**7*np.sin(ang1*k1 + phi1))/k1 - (350*a0**4*a1**3*np.sin(3*(ang1*k1 + phi1)))/k1 - (350*a0**2*a1**5*np.sin(3*(ang1*k1 + phi1)))/k1 - (35*a1**7*np.sin(3*(ang1*k1 + phi1)))/k1 - (42*a0**2*a1**5*np.sin(5*(ang1*k1 + phi1)))/k1 - (7*a1**7*np.sin(5*(ang1*k1 + phi1)))/k1 + (5*a1**7*np.sin(7*(ang1*k1 + phi1)))/k1 - (525*a0**6*a1*np.sin(2*ang1*k0 + ang1*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (2100*a0**4*a1**3*np.sin(2*ang1*k0 + ang1*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (1050*a0**2*a1**5*np.sin(2*ang1*k0 + ang1*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (210*a0**6*a1*np.sin(4*ang1*k0 + ang1*k1 + 4*phi0 + phi1))/(4*k0 + k1) - (525*a0**4*a1**3*np.sin(4*ang1*k0 + ang1*k1 + 4*phi0 + phi1))/(4*k0 + k1) + (245*a0**6*a1*np.sin(6*ang1*k0 + ang1*k1 + 6*phi0 + phi1))/(6*k0 + k1) - (1050*a0**5*a1**2*np.sin(ang1*k0 + 2*ang1*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (2100*a0**3*a1**4*np.sin(ang1*k0 + 2*ang1*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (525*a0*a1**6*np.sin(ang1*k0 + 2*ang1*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (525*a0**5*a1**2*np.sin(3*ang1*k0 + 2*ang1*k1 + 3*phi0 + 2*phi1))/(3*k0 + 2*k1) - (700*a0**3*a1**4*np.sin(3*ang1*k0 + 2*ang1*k1 + 3*phi0 + 2*phi1))/(3*k0 + 2*k1) + (735*a0**5*a1**2*np.sin(5*ang1*k0 + 2*ang1*k1 + 5*phi0 + 2*phi1))/(5*k0 + 2*k1) - (700*a0**4*a1**3*np.sin(2*ang1*k0 + 3*ang1*k1 + 2*phi0 + 3*phi1))/(2*k0 + 3*k1) - (525*a0**2*a1**5*np.sin(2*ang1*k0 + 3*ang1*k1 + 2*phi0 + 3*phi1))/(2*k0 + 3*k1) + (1225*a0**4*a1**3*np.sin(4*ang1*k0 + 3*ang1*k1 + 4*phi0 + 3*phi1))/(4*k0 + 3*k1) - (525*a0**3*a1**4*np.sin(ang1*k0 + 4*ang1*k1 + phi0 + 4*phi1))/(k0 + 4*k1) - (210*a0*a1**6*np.sin(ang1*k0 + 4*ang1*k1 + phi0 + 4*phi1))/(k0 + 4*k1) + (1225*a0**3*a1**4*np.sin(3*ang1*k0 + 4*ang1*k1 + 3*phi0 + 4*phi1))/(3*k0 + 4*k1) + (735*a0**2*a1**5*np.sin(2*ang1*k0 + 5*ang1*k1 + 2*phi0 + 5*phi1))/(2*k0 + 5*k1) + (245*a0*a1**6*np.sin(ang1*k0 + 6*ang1*k1 + phi0 + 6*phi1))/(k0 + 6*k1))/2240.)
            return res1 - res0
        elif j == 5:
            res0 = (((-72*(a0**8 + 15*a0**6*a1**2 + 30*a0**4*a1**4 + 10*a0**2*a1**6)*np.cos(2*(ang0*k0 + phi0)))/k0 + (12*(a0**8 + 12*a0**6*a1**2 + 15*a0**4*a1**4)*np.cos(4*(ang0*k0 + phi0)))/k0 + (8*a0**8*np.cos(6*(ang0*k0 + phi0)))/k0 + (56*a0**6*a1**2*np.cos(6*(ang0*k0 + phi0)))/k0 - (3*a0**8*np.cos(8*(ang0*k0 + phi0)))/k0 - (48*a0*a1**7*np.cos(ang0*k0 - 7*ang0*k1 + phi0 - 7*phi1))/(k0 - 7*k1) - (288*a0**3*a1**5*np.cos(ang0*k0 - 5*ang0*k1 + phi0 - 5*phi1))/(k0 - 5*k1) - (96*a0*a1**7*np.cos(ang0*k0 - 5*ang0*k1 + phi0 - 5*phi1))/(k0 - 5*k1) + (144*a0**3*a1**5*np.cos(3*ang0*k0 - 5*ang0*k1 + 3*phi0 - 5*phi1))/(3*k0 - 5*k1) + (1440*a0**5*a1**3*np.cos(ang0*k0 - 3*ang0*k1 + phi0 - 3*phi1))/(k0 - 3*k1) + (2160*a0**3*a1**5*np.cos(ang0*k0 - 3*ang0*k1 + phi0 - 3*phi1))/(k0 - 3*k1) + (432*a0*a1**7*np.cos(ang0*k0 - 3*ang0*k1 + phi0 - 3*phi1))/(k0 - 3*k1) - (24*a0**2*a1**6*np.cos(2*(ang0*k0 - 3*ang0*k1 + phi0 - 3*phi1)))/(k0 - 3*k1) - (144*a0**5*a1**3*np.cos(5*ang0*k0 - 3*ang0*k1 + 5*phi0 - 3*phi1))/(5*k0 - 3*k1) + (360*a0**4*a1**4*np.cos(2*(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1)))/(k0 - 2*k1) + (216*a0**2*a1**6*np.cos(2*(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1)))/(k0 - 2*k1) - (216*a0**6*a1**2*np.cos(4*ang0*k0 - 2*ang0*k1 + 4*phi0 - 2*phi1))/(2*k0 - k1) - (360*a0**4*a1**4*np.cos(4*ang0*k0 - 2*ang0*k1 + 4*phi0 - 2*phi1))/(2*k0 - k1) + (24*a0**6*a1**2*np.cos(6*ang0*k0 - 2*ang0*k1 + 6*phi0 - 2*phi1))/(3*k0 - k1) - (432*a0**7*a1*np.cos(3*ang0*k0 - ang0*k1 + 3*phi0 - phi1))/(3*k0 - k1) - (2160*a0**5*a1**3*np.cos(3*ang0*k0 - ang0*k1 + 3*phi0 - phi1))/(3*k0 - k1) - (1440*a0**3*a1**5*np.cos(3*ang0*k0 - ang0*k1 + 3*phi0 - phi1))/(3*k0 - k1) + (96*a0**7*a1*np.cos(5*ang0*k0 - ang0*k1 + 5*phi0 - phi1))/(5*k0 - k1) + (288*a0**5*a1**3*np.cos(5*ang0*k0 - ang0*k1 + 5*phi0 - phi1))/(5*k0 - k1) + (48*a0**7*a1*np.cos(7*ang0*k0 - ang0*k1 + 7*phi0 - phi1))/(7*k0 - k1) - (720*a0**6*a1**2*np.cos(2*(ang0*k1 + phi1)))/k1 - (2160*a0**4*a1**4*np.cos(2*(ang0*k1 + phi1)))/k1 - (1080*a0**2*a1**6*np.cos(2*(ang0*k1 + phi1)))/k1 - (72*a1**8*np.cos(2*(ang0*k1 + phi1)))/k1 + (180*a0**4*a1**4*np.cos(4*(ang0*k1 + phi1)))/k1 + (144*a0**2*a1**6*np.cos(4*(ang0*k1 + phi1)))/k1 + (12*a1**8*np.cos(4*(ang0*k1 + phi1)))/k1 + (56*a0**2*a1**6*np.cos(6*(ang0*k1 + phi1)))/k1 + (8*a1**8*np.cos(6*(ang0*k1 + phi1)))/k1 - (3*a1**8*np.cos(8*(ang0*k1 + phi1)))/k1 - (720*a0**7*a1*np.cos(ang0*(k0 + k1) + phi0 + phi1))/(k0 + k1) - (4320*a0**5*a1**3*np.cos(ang0*(k0 + k1) + phi0 + phi1))/(k0 + k1) - (4320*a0**3*a1**5*np.cos(ang0*(k0 + k1) + phi0 + phi1))/(k0 + k1) - (720*a0*a1**7*np.cos(ang0*(k0 + k1) + phi0 + phi1))/(k0 + k1) + (360*a0**6*a1**2*np.cos(2*(ang0*(k0 + k1) + phi0 + phi1)))/(k0 + k1) + (960*a0**4*a1**4*np.cos(2*(ang0*(k0 + k1) + phi0 + phi1)))/(k0 + k1) + (360*a0**2*a1**6*np.cos(2*(ang0*(k0 + k1) + phi0 + phi1)))/(k0 + k1) + (560*a0**5*a1**3*np.cos(3*(ang0*(k0 + k1) + phi0 + phi1)))/(k0 + k1) + (560*a0**3*a1**5*np.cos(3*(ang0*(k0 + k1) + phi0 + phi1)))/(k0 + k1) - (420*a0**4*a1**4*np.cos(4*(ang0*(k0 + k1) + phi0 + phi1)))/(k0 + k1) + (504*a0**6*a1**2*np.cos(2*(2*ang0*k0 + ang0*k1 + 2*phi0 + phi1)))/(2*k0 + k1) + (840*a0**4*a1**4*np.cos(2*(2*ang0*k0 + ang0*k1 + 2*phi0 + phi1)))/(2*k0 + k1) + (288*a0**7*a1*np.cos(3*ang0*k0 + ang0*k1 + 3*phi0 + phi1))/(3*k0 + k1) + (1440*a0**5*a1**3*np.cos(3*ang0*k0 + ang0*k1 + 3*phi0 + phi1))/(3*k0 + k1) + (960*a0**3*a1**5*np.cos(3*ang0*k0 + ang0*k1 + 3*phi0 + phi1))/(3*k0 + k1) - (336*a0**6*a1**2*np.cos(2*(3*ang0*k0 + ang0*k1 + 3*phi0 + phi1)))/(3*k0 + k1) + (336*a0**7*a1*np.cos(5*ang0*k0 + ang0*k1 + 5*phi0 + phi1))/(5*k0 + k1) + (1008*a0**5*a1**3*np.cos(5*ang0*k0 + ang0*k1 + 5*phi0 + phi1))/(5*k0 + k1) - (192*a0**7*a1*np.cos(7*ang0*k0 + ang0*k1 + 7*phi0 + phi1))/(7*k0 + k1) + (840*a0**4*a1**4*np.cos(2*(ang0*k0 + 2*ang0*k1 + phi0 + 2*phi1)))/(k0 + 2*k1) + (504*a0**2*a1**6*np.cos(2*(ang0*k0 + 2*ang0*k1 + phi0 + 2*phi1)))/(k0 + 2*k1) + (960*a0**5*a1**3*np.cos(ang0*k0 + 3*ang0*k1 + phi0 + 3*phi1))/(k0 + 3*k1) + (1440*a0**3*a1**5*np.cos(ang0*k0 + 3*ang0*k1 + phi0 + 3*phi1))/(k0 + 3*k1) + (288*a0*a1**7*np.cos(ang0*k0 + 3*ang0*k1 + phi0 + 3*phi1))/(k0 + 3*k1) - (336*a0**2*a1**6*np.cos(2*(ang0*k0 + 3*ang0*k1 + phi0 + 3*phi1)))/(k0 + 3*k1) - (1344*a0**5*a1**3*np.cos(5*ang0*k0 + 3*ang0*k1 + 5*phi0 + 3*phi1))/(5*k0 + 3*k1) + (1008*a0**3*a1**5*np.cos(ang0*k0 + 5*ang0*k1 + phi0 + 5*phi1))/(k0 + 5*k1) + (336*a0*a1**7*np.cos(ang0*k0 + 5*ang0*k1 + phi0 + 5*phi1))/(k0 + 5*k1) - (1344*a0**3*a1**5*np.cos(3*ang0*k0 + 5*ang0*k1 + 3*phi0 + 5*phi1))/(3*k0 + 5*k1) - (192*a0*a1**7*np.cos(ang0*k0 + 7*ang0*k1 + phi0 + 7*phi1))/(k0 + 7*k1))/3072.)
            res1 = (((-72*(a0**8 + 15*a0**6*a1**2 + 30*a0**4*a1**4 + 10*a0**2*a1**6)*np.cos(2*(ang1*k0 + phi0)))/k0 + (12*(a0**8 + 12*a0**6*a1**2 + 15*a0**4*a1**4)*np.cos(4*(ang1*k0 + phi0)))/k0 + (8*a0**8*np.cos(6*(ang1*k0 + phi0)))/k0 + (56*a0**6*a1**2*np.cos(6*(ang1*k0 + phi0)))/k0 - (3*a0**8*np.cos(8*(ang1*k0 + phi0)))/k0 - (48*a0*a1**7*np.cos(ang1*k0 - 7*ang1*k1 + phi0 - 7*phi1))/(k0 - 7*k1) - (288*a0**3*a1**5*np.cos(ang1*k0 - 5*ang1*k1 + phi0 - 5*phi1))/(k0 - 5*k1) - (96*a0*a1**7*np.cos(ang1*k0 - 5*ang1*k1 + phi0 - 5*phi1))/(k0 - 5*k1) + (144*a0**3*a1**5*np.cos(3*ang1*k0 - 5*ang1*k1 + 3*phi0 - 5*phi1))/(3*k0 - 5*k1) + (1440*a0**5*a1**3*np.cos(ang1*k0 - 3*ang1*k1 + phi0 - 3*phi1))/(k0 - 3*k1) + (2160*a0**3*a1**5*np.cos(ang1*k0 - 3*ang1*k1 + phi0 - 3*phi1))/(k0 - 3*k1) + (432*a0*a1**7*np.cos(ang1*k0 - 3*ang1*k1 + phi0 - 3*phi1))/(k0 - 3*k1) - (24*a0**2*a1**6*np.cos(2*(ang1*k0 - 3*ang1*k1 + phi0 - 3*phi1)))/(k0 - 3*k1) - (144*a0**5*a1**3*np.cos(5*ang1*k0 - 3*ang1*k1 + 5*phi0 - 3*phi1))/(5*k0 - 3*k1) + (360*a0**4*a1**4*np.cos(2*(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1)))/(k0 - 2*k1) + (216*a0**2*a1**6*np.cos(2*(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1)))/(k0 - 2*k1) - (216*a0**6*a1**2*np.cos(4*ang1*k0 - 2*ang1*k1 + 4*phi0 - 2*phi1))/(2*k0 - k1) - (360*a0**4*a1**4*np.cos(4*ang1*k0 - 2*ang1*k1 + 4*phi0 - 2*phi1))/(2*k0 - k1) + (24*a0**6*a1**2*np.cos(6*ang1*k0 - 2*ang1*k1 + 6*phi0 - 2*phi1))/(3*k0 - k1) - (432*a0**7*a1*np.cos(3*ang1*k0 - ang1*k1 + 3*phi0 - phi1))/(3*k0 - k1) - (2160*a0**5*a1**3*np.cos(3*ang1*k0 - ang1*k1 + 3*phi0 - phi1))/(3*k0 - k1) - (1440*a0**3*a1**5*np.cos(3*ang1*k0 - ang1*k1 + 3*phi0 - phi1))/(3*k0 - k1) + (96*a0**7*a1*np.cos(5*ang1*k0 - ang1*k1 + 5*phi0 - phi1))/(5*k0 - k1) + (288*a0**5*a1**3*np.cos(5*ang1*k0 - ang1*k1 + 5*phi0 - phi1))/(5*k0 - k1) + (48*a0**7*a1*np.cos(7*ang1*k0 - ang1*k1 + 7*phi0 - phi1))/(7*k0 - k1) - (720*a0**6*a1**2*np.cos(2*(ang1*k1 + phi1)))/k1 - (2160*a0**4*a1**4*np.cos(2*(ang1*k1 + phi1)))/k1 - (1080*a0**2*a1**6*np.cos(2*(ang1*k1 + phi1)))/k1 - (72*a1**8*np.cos(2*(ang1*k1 + phi1)))/k1 + (180*a0**4*a1**4*np.cos(4*(ang1*k1 + phi1)))/k1 + (144*a0**2*a1**6*np.cos(4*(ang1*k1 + phi1)))/k1 + (12*a1**8*np.cos(4*(ang1*k1 + phi1)))/k1 + (56*a0**2*a1**6*np.cos(6*(ang1*k1 + phi1)))/k1 + (8*a1**8*np.cos(6*(ang1*k1 + phi1)))/k1 - (3*a1**8*np.cos(8*(ang1*k1 + phi1)))/k1 - (720*a0**7*a1*np.cos(ang1*(k0 + k1) + phi0 + phi1))/(k0 + k1) - (4320*a0**5*a1**3*np.cos(ang1*(k0 + k1) + phi0 + phi1))/(k0 + k1) - (4320*a0**3*a1**5*np.cos(ang1*(k0 + k1) + phi0 + phi1))/(k0 + k1) - (720*a0*a1**7*np.cos(ang1*(k0 + k1) + phi0 + phi1))/(k0 + k1) + (360*a0**6*a1**2*np.cos(2*(ang1*(k0 + k1) + phi0 + phi1)))/(k0 + k1) + (960*a0**4*a1**4*np.cos(2*(ang1*(k0 + k1) + phi0 + phi1)))/(k0 + k1) + (360*a0**2*a1**6*np.cos(2*(ang1*(k0 + k1) + phi0 + phi1)))/(k0 + k1) + (560*a0**5*a1**3*np.cos(3*(ang1*(k0 + k1) + phi0 + phi1)))/(k0 + k1) + (560*a0**3*a1**5*np.cos(3*(ang1*(k0 + k1) + phi0 + phi1)))/(k0 + k1) - (420*a0**4*a1**4*np.cos(4*(ang1*(k0 + k1) + phi0 + phi1)))/(k0 + k1) + (504*a0**6*a1**2*np.cos(2*(2*ang1*k0 + ang1*k1 + 2*phi0 + phi1)))/(2*k0 + k1) + (840*a0**4*a1**4*np.cos(2*(2*ang1*k0 + ang1*k1 + 2*phi0 + phi1)))/(2*k0 + k1) + (288*a0**7*a1*np.cos(3*ang1*k0 + ang1*k1 + 3*phi0 + phi1))/(3*k0 + k1) + (1440*a0**5*a1**3*np.cos(3*ang1*k0 + ang1*k1 + 3*phi0 + phi1))/(3*k0 + k1) + (960*a0**3*a1**5*np.cos(3*ang1*k0 + ang1*k1 + 3*phi0 + phi1))/(3*k0 + k1) - (336*a0**6*a1**2*np.cos(2*(3*ang1*k0 + ang1*k1 + 3*phi0 + phi1)))/(3*k0 + k1) + (336*a0**7*a1*np.cos(5*ang1*k0 + ang1*k1 + 5*phi0 + phi1))/(5*k0 + k1) + (1008*a0**5*a1**3*np.cos(5*ang1*k0 + ang1*k1 + 5*phi0 + phi1))/(5*k0 + k1) - (192*a0**7*a1*np.cos(7*ang1*k0 + ang1*k1 + 7*phi0 + phi1))/(7*k0 + k1) + (840*a0**4*a1**4*np.cos(2*(ang1*k0 + 2*ang1*k1 + phi0 + 2*phi1)))/(k0 + 2*k1) + (504*a0**2*a1**6*np.cos(2*(ang1*k0 + 2*ang1*k1 + phi0 + 2*phi1)))/(k0 + 2*k1) + (960*a0**5*a1**3*np.cos(ang1*k0 + 3*ang1*k1 + phi0 + 3*phi1))/(k0 + 3*k1) + (1440*a0**3*a1**5*np.cos(ang1*k0 + 3*ang1*k1 + phi0 + 3*phi1))/(k0 + 3*k1) + (288*a0*a1**7*np.cos(ang1*k0 + 3*ang1*k1 + phi0 + 3*phi1))/(k0 + 3*k1) - (336*a0**2*a1**6*np.cos(2*(ang1*k0 + 3*ang1*k1 + phi0 + 3*phi1)))/(k0 + 3*k1) - (1344*a0**5*a1**3*np.cos(5*ang1*k0 + 3*ang1*k1 + 5*phi0 + 3*phi1))/(5*k0 + 3*k1) + (1008*a0**3*a1**5*np.cos(ang1*k0 + 5*ang1*k1 + phi0 + 5*phi1))/(k0 + 5*k1) + (336*a0*a1**7*np.cos(ang1*k0 + 5*ang1*k1 + phi0 + 5*phi1))/(k0 + 5*k1) - (1344*a0**3*a1**5*np.cos(3*ang1*k0 + 5*ang1*k1 + 3*phi0 + 5*phi1))/(3*k0 + 5*k1) - (192*a0*a1**7*np.cos(ang1*k0 + 7*ang1*k1 + phi0 + 7*phi1))/(k0 + 7*k1))/3072.)
            return res1 - res0
        else:
            print('invalid index j = ' + str(j) + ' in Iij')
            return 0.0
    elif i == 4:
        if j == 0:
            res0 = ((8*a0**2*(a0**2 + 3*a1**2)*k1*(9*k0**6 - 91*k0**4*k1**2 + 91*k0**2*k1**4 - 9*k1**6)*np.sin(2*(ang0*k0 + phi0)) + a0**4*k1*(9*k0**6 - 91*k0**4*k1**2 + 91*k0**2*k1**4 - 9*k1**6)*np.sin(4*(ang0*k0 + phi0)) + k0*(16*a0*a1**3*k1*(9*k0**5 + 27*k0**4*k1 - 10*k0**3*k1**2 - 30*k0**2*k1**3 + k0*k1**4 + 3*k1**5)*np.sin(ang0*(k0 - 3*k1) + phi0 - 3*phi1) + (k0 - 3*k1)*(48*a0*a1*(a0**2 + a1**2)*k1*(9*k0**4 + 36*k0**3*k1 + 26*k0**2*k1**2 - 4*k0*k1**3 - 3*k1**4)*np.sin(ang0*(k0 - k1) + phi0 - phi1) + 12*a0**2*a1**2*k1*(9*k0**4 + 36*k0**3*k1 + 26*k0**2*k1**2 - 4*k0*k1**3 - 3*k1**4)*np.sin(2*(ang0*(k0 - k1) + phi0 - phi1)) + (k0 - k1)*(16*a0**3*a1*k1*(3*k0**3 + 13*k0**2*k1 + 13*k0*k1**2 + 3*k1**3)*np.sin(3*ang0*k0 - ang0*k1 + 3*phi0 - phi1) + (3*k0 - k1)*(8*a1**2*(3*a0**2 + a1**2)*(3*k0**3 + 13*k0**2*k1 + 13*k0*k1**2 + 3*k1**3)*np.sin(2*(ang0*k1 + phi1)) + a1**4*(3*k0**3 + 13*k0**2*k1 + 13*k0*k1**2 + 3*k1**3)*np.sin(4*(ang0*k1 + phi1)) + 4*k1*(12*a0*a1*(a0**2 + a1**2)*(3*k0**2 + 10*k0*k1 + 3*k1**2)*np.sin(ang0*(k0 + k1) + phi0 + phi1) + 3*a0**2*a1**2*(3*k0**2 + 10*k0*k1 + 3*k1**2)*np.sin(2*(ang0*(k0 + k1) + phi0 + phi1)) + (k0 + k1)*(4*a0**3*a1*(k0 + 3*k1)*np.sin(ang0*(3*k0 + k1) + 3*phi0 + phi1) + (3*k0 + k1)*(3*(a0**4 + 4*a0**2*a1**2 + a1**4)*ang0*(k0 + 3*k1) + 4*a0*a1**3*np.sin(ang0*(k0 + 3*k1) + phi0 + 3*phi1)))))))))/(32.*(9*k0**7*k1 - 91*k0**5*k1**3 + 91*k0**3*k1**5 - 9*k0*k1**7)))
            res1 = ((8*a0**2*(a0**2 + 3*a1**2)*k1*(9*k0**6 - 91*k0**4*k1**2 + 91*k0**2*k1**4 - 9*k1**6)*np.sin(2*(ang1*k0 + phi0)) + a0**4*k1*(9*k0**6 - 91*k0**4*k1**2 + 91*k0**2*k1**4 - 9*k1**6)*np.sin(4*(ang1*k0 + phi0)) + k0*(16*a0*a1**3*k1*(9*k0**5 + 27*k0**4*k1 - 10*k0**3*k1**2 - 30*k0**2*k1**3 + k0*k1**4 + 3*k1**5)*np.sin(ang1*(k0 - 3*k1) + phi0 - 3*phi1) + (k0 - 3*k1)*(48*a0*a1*(a0**2 + a1**2)*k1*(9*k0**4 + 36*k0**3*k1 + 26*k0**2*k1**2 - 4*k0*k1**3 - 3*k1**4)*np.sin(ang1*(k0 - k1) + phi0 - phi1) + 12*a0**2*a1**2*k1*(9*k0**4 + 36*k0**3*k1 + 26*k0**2*k1**2 - 4*k0*k1**3 - 3*k1**4)*np.sin(2*(ang1*(k0 - k1) + phi0 - phi1)) + (k0 - k1)*(16*a0**3*a1*k1*(3*k0**3 + 13*k0**2*k1 + 13*k0*k1**2 + 3*k1**3)*np.sin(3*ang1*k0 - ang1*k1 + 3*phi0 - phi1) + (3*k0 - k1)*(8*a1**2*(3*a0**2 + a1**2)*(3*k0**3 + 13*k0**2*k1 + 13*k0*k1**2 + 3*k1**3)*np.sin(2*(ang1*k1 + phi1)) + a1**4*(3*k0**3 + 13*k0**2*k1 + 13*k0*k1**2 + 3*k1**3)*np.sin(4*(ang1*k1 + phi1)) + 4*k1*(12*a0*a1*(a0**2 + a1**2)*(3*k0**2 + 10*k0*k1 + 3*k1**2)*np.sin(ang1*(k0 + k1) + phi0 + phi1) + 3*a0**2*a1**2*(3*k0**2 + 10*k0*k1 + 3*k1**2)*np.sin(2*(ang1*(k0 + k1) + phi0 + phi1)) + (k0 + k1)*(4*a0**3*a1*(k0 + 3*k1)*np.sin(ang1*(3*k0 + k1) + 3*phi0 + phi1) + (3*k0 + k1)*(3*(a0**4 + 4*a0**2*a1**2 + a1**4)*ang1*(k0 + 3*k1) + 4*a0*a1**3*np.sin(ang1*(k0 + 3*k1) + phi0 + 3*phi1)))))))))/(32.*(9*k0**7*k1 - 91*k0**5*k1**3 + 91*k0**3*k1**5 - 9*k0*k1**7)))
            return res1 - res0
        elif j == 1:
            res0 = (((-10*a0*(a0**4 + 6*a0**2*a1**2 + 3*a1**4)*np.cos(ang0*k0 + phi0))/k0 - (5*(a0**5 + 4*a0**3*a1**2)*np.cos(3*(ang0*k0 + phi0)))/k0 - (a0**5*np.cos(5*(ang0*k0 + phi0)))/k0 + (15*a0*a1**4*np.cos(ang0*k0 - 4*ang0*k1 + phi0 - 4*phi1))/(k0 - 4*k1) + (10*a0**2*a1**3*np.cos(2*ang0*k0 - 3*ang0*k1 + 2*phi0 - 3*phi1))/(2*k0 - 3*k1) + (30*a0**3*a1**2*np.cos(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1))/(k0 - 2*k1) + (20*a0*a1**4*np.cos(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1))/(k0 - 2*k1) - (10*a0**3*a1**2*np.cos(3*ang0*k0 - 2*ang0*k1 + 3*phi0 - 2*phi1))/(3*k0 - 2*k1) - (20*a0**4*a1*np.cos(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1))/(2*k0 - k1) - (30*a0**2*a1**3*np.cos(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1))/(2*k0 - k1) - (15*a0**4*a1*np.cos(4*ang0*k0 - ang0*k1 + 4*phi0 - phi1))/(4*k0 - k1) - (30*a0**4*a1*np.cos(ang0*k1 + phi1))/k1 - (60*a0**2*a1**3*np.cos(ang0*k1 + phi1))/k1 - (10*a1**5*np.cos(ang0*k1 + phi1))/k1 - (20*a0**2*a1**3*np.cos(3*(ang0*k1 + phi1)))/k1 - (5*a1**5*np.cos(3*(ang0*k1 + phi1)))/k1 - (a1**5*np.cos(5*(ang0*k1 + phi1)))/k1 - (60*a0**4*a1*np.cos(2*ang0*k0 + ang0*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (90*a0**2*a1**3*np.cos(2*ang0*k0 + ang0*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (25*a0**4*a1*np.cos(4*ang0*k0 + ang0*k1 + 4*phi0 + phi1))/(4*k0 + k1) - (90*a0**3*a1**2*np.cos(ang0*k0 + 2*ang0*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (60*a0*a1**4*np.cos(ang0*k0 + 2*ang0*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (50*a0**3*a1**2*np.cos(3*ang0*k0 + 2*ang0*k1 + 3*phi0 + 2*phi1))/(3*k0 + 2*k1) - (50*a0**2*a1**3*np.cos(2*ang0*k0 + 3*ang0*k1 + 2*phi0 + 3*phi1))/(2*k0 + 3*k1) - (25*a0*a1**4*np.cos(ang0*k0 + 4*ang0*k1 + phi0 + 4*phi1))/(k0 + 4*k1))/80.)
            res1 = (((-10*a0*(a0**4 + 6*a0**2*a1**2 + 3*a1**4)*np.cos(ang1*k0 + phi0))/k0 - (5*(a0**5 + 4*a0**3*a1**2)*np.cos(3*(ang1*k0 + phi0)))/k0 - (a0**5*np.cos(5*(ang1*k0 + phi0)))/k0 + (15*a0*a1**4*np.cos(ang1*k0 - 4*ang1*k1 + phi0 - 4*phi1))/(k0 - 4*k1) + (10*a0**2*a1**3*np.cos(2*ang1*k0 - 3*ang1*k1 + 2*phi0 - 3*phi1))/(2*k0 - 3*k1) + (30*a0**3*a1**2*np.cos(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1))/(k0 - 2*k1) + (20*a0*a1**4*np.cos(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1))/(k0 - 2*k1) - (10*a0**3*a1**2*np.cos(3*ang1*k0 - 2*ang1*k1 + 3*phi0 - 2*phi1))/(3*k0 - 2*k1) - (20*a0**4*a1*np.cos(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1))/(2*k0 - k1) - (30*a0**2*a1**3*np.cos(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1))/(2*k0 - k1) - (15*a0**4*a1*np.cos(4*ang1*k0 - ang1*k1 + 4*phi0 - phi1))/(4*k0 - k1) - (30*a0**4*a1*np.cos(ang1*k1 + phi1))/k1 - (60*a0**2*a1**3*np.cos(ang1*k1 + phi1))/k1 - (10*a1**5*np.cos(ang1*k1 + phi1))/k1 - (20*a0**2*a1**3*np.cos(3*(ang1*k1 + phi1)))/k1 - (5*a1**5*np.cos(3*(ang1*k1 + phi1)))/k1 - (a1**5*np.cos(5*(ang1*k1 + phi1)))/k1 - (60*a0**4*a1*np.cos(2*ang1*k0 + ang1*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (90*a0**2*a1**3*np.cos(2*ang1*k0 + ang1*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (25*a0**4*a1*np.cos(4*ang1*k0 + ang1*k1 + 4*phi0 + phi1))/(4*k0 + k1) - (90*a0**3*a1**2*np.cos(ang1*k0 + 2*ang1*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (60*a0*a1**4*np.cos(ang1*k0 + 2*ang1*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (50*a0**3*a1**2*np.cos(3*ang1*k0 + 2*ang1*k1 + 3*phi0 + 2*phi1))/(3*k0 + 2*k1) - (50*a0**2*a1**3*np.cos(2*ang1*k0 + 3*ang1*k1 + 2*phi0 + 3*phi1))/(2*k0 + 3*k1) - (25*a0*a1**4*np.cos(ang1*k0 + 4*ang1*k1 + phi0 + 4*phi1))/(k0 + 4*k1))/80.)
            return res1 - res0
        elif j == 2:
            res0 = ((12*a0**6*ang0 + 108*a0**4*a1**2*ang0 + 108*a0**2*a1**4*ang0 + 12*a1**6*ang0 + (3*(a0**6 + 8*a0**4*a1**2 + 6*a0**2*a1**4)*np.sin(2*(ang0*k0 + phi0)))/k0 - (3*(a0**6 + 5*a0**4*a1**2)*np.sin(4*(ang0*k0 + phi0)))/k0 - (a0**6*np.sin(6*(ang0*k0 + phi0)))/k0 - (12*a0*a1**5*np.sin(ang0*k0 - 5*ang0*k1 + phi0 - 5*phi1))/(k0 - 5*k1) + (24*a0**3*a1**3*np.sin(ang0*k0 - 3*ang0*k1 + phi0 - 3*phi1))/(k0 - 3*k1) + (12*a0*a1**5*np.sin(ang0*k0 - 3*ang0*k1 + phi0 - 3*phi1))/(k0 - 3*k1) + (3*a0**2*a1**4*np.sin(2*(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1)))/(k0 - 2*k1) + (3*a0**4*a1**2*np.sin(4*ang0*k0 - 2*ang0*k1 + 4*phi0 - 2*phi1))/(2*k0 - k1) + (72*a0**5*a1*np.sin(ang0*k0 - ang0*k1 + phi0 - phi1))/(k0 - k1) + (216*a0**3*a1**3*np.sin(ang0*k0 - ang0*k1 + phi0 - phi1))/(k0 - k1) + (72*a0*a1**5*np.sin(ang0*k0 - ang0*k1 + phi0 - phi1))/(k0 - k1) + (36*a0**4*a1**2*np.sin(2*(ang0*k0 - ang0*k1 + phi0 - phi1)))/(k0 - k1) + (36*a0**2*a1**4*np.sin(2*(ang0*k0 - ang0*k1 + phi0 - phi1)))/(k0 - k1) + (8*a0**3*a1**3*np.sin(3*(ang0*k0 - ang0*k1 + phi0 - phi1)))/(k0 - k1) + (12*a0**5*a1*np.sin(3*ang0*k0 - ang0*k1 + 3*phi0 - phi1))/(3*k0 - k1) + (24*a0**3*a1**3*np.sin(3*ang0*k0 - ang0*k1 + 3*phi0 - phi1))/(3*k0 - k1) - (12*a0**5*a1*np.sin(5*ang0*k0 - ang0*k1 + 5*phi0 - phi1))/(5*k0 - k1) + (18*a0**4*a1**2*np.sin(2*(ang0*k1 + phi1)))/k1 + (24*a0**2*a1**4*np.sin(2*(ang0*k1 + phi1)))/k1 + (3*a1**6*np.sin(2*(ang0*k1 + phi1)))/k1 - (15*a0**2*a1**4*np.sin(4*(ang0*k1 + phi1)))/k1 - (3*a1**6*np.sin(4*(ang0*k1 + phi1)))/k1 - (a1**6*np.sin(6*(ang0*k1 + phi1)))/k1 + (24*a0**5*a1*np.sin(ang0*(k0 + k1) + phi0 + phi1))/(k0 + k1) + (72*a0**3*a1**3*np.sin(ang0*(k0 + k1) + phi0 + phi1))/(k0 + k1) + (24*a0*a1**5*np.sin(ang0*(k0 + k1) + phi0 + phi1))/(k0 + k1) - (60*a0**4*a1**2*np.sin(2*(ang0*(k0 + k1) + phi0 + phi1)))/(k0 + k1) - (60*a0**2*a1**4*np.sin(2*(ang0*(k0 + k1) + phi0 + phi1)))/(k0 + k1) - (40*a0**3*a1**3*np.sin(3*(ang0*(k0 + k1) + phi0 + phi1)))/(k0 + k1) - (45*a0**4*a1**2*np.sin(2*(2*ang0*k0 + ang0*k1 + 2*phi0 + phi1)))/(2*k0 + k1) - (60*a0**5*a1*np.sin(3*ang0*k0 + ang0*k1 + 3*phi0 + phi1))/(3*k0 + k1) - (120*a0**3*a1**3*np.sin(3*ang0*k0 + ang0*k1 + 3*phi0 + phi1))/(3*k0 + k1) - (36*a0**5*a1*np.sin(5*ang0*k0 + ang0*k1 + 5*phi0 + phi1))/(5*k0 + k1) - (45*a0**2*a1**4*np.sin(2*(ang0*k0 + 2*ang0*k1 + phi0 + 2*phi1)))/(k0 + 2*k1) - (120*a0**3*a1**3*np.sin(ang0*k0 + 3*ang0*k1 + phi0 + 3*phi1))/(k0 + 3*k1) - (60*a0*a1**5*np.sin(ang0*k0 + 3*ang0*k1 + phi0 + 3*phi1))/(k0 + 3*k1) - (36*a0*a1**5*np.sin(ang0*k0 + 5*ang0*k1 + phi0 + 5*phi1))/(k0 + 5*k1))/192.)
            res1 = ((12*a0**6*ang1 + 108*a0**4*a1**2*ang1 + 108*a0**2*a1**4*ang1 + 12*a1**6*ang1 + (3*(a0**6 + 8*a0**4*a1**2 + 6*a0**2*a1**4)*np.sin(2*(ang1*k0 + phi0)))/k0 - (3*(a0**6 + 5*a0**4*a1**2)*np.sin(4*(ang1*k0 + phi0)))/k0 - (a0**6*np.sin(6*(ang1*k0 + phi0)))/k0 - (12*a0*a1**5*np.sin(ang1*k0 - 5*ang1*k1 + phi0 - 5*phi1))/(k0 - 5*k1) + (24*a0**3*a1**3*np.sin(ang1*k0 - 3*ang1*k1 + phi0 - 3*phi1))/(k0 - 3*k1) + (12*a0*a1**5*np.sin(ang1*k0 - 3*ang1*k1 + phi0 - 3*phi1))/(k0 - 3*k1) + (3*a0**2*a1**4*np.sin(2*(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1)))/(k0 - 2*k1) + (3*a0**4*a1**2*np.sin(4*ang1*k0 - 2*ang1*k1 + 4*phi0 - 2*phi1))/(2*k0 - k1) + (72*a0**5*a1*np.sin(ang1*k0 - ang1*k1 + phi0 - phi1))/(k0 - k1) + (216*a0**3*a1**3*np.sin(ang1*k0 - ang1*k1 + phi0 - phi1))/(k0 - k1) + (72*a0*a1**5*np.sin(ang1*k0 - ang1*k1 + phi0 - phi1))/(k0 - k1) + (36*a0**4*a1**2*np.sin(2*(ang1*k0 - ang1*k1 + phi0 - phi1)))/(k0 - k1) + (36*a0**2*a1**4*np.sin(2*(ang1*k0 - ang1*k1 + phi0 - phi1)))/(k0 - k1) + (8*a0**3*a1**3*np.sin(3*(ang1*k0 - ang1*k1 + phi0 - phi1)))/(k0 - k1) + (12*a0**5*a1*np.sin(3*ang1*k0 - ang1*k1 + 3*phi0 - phi1))/(3*k0 - k1) + (24*a0**3*a1**3*np.sin(3*ang1*k0 - ang1*k1 + 3*phi0 - phi1))/(3*k0 - k1) - (12*a0**5*a1*np.sin(5*ang1*k0 - ang1*k1 + 5*phi0 - phi1))/(5*k0 - k1) + (18*a0**4*a1**2*np.sin(2*(ang1*k1 + phi1)))/k1 + (24*a0**2*a1**4*np.sin(2*(ang1*k1 + phi1)))/k1 + (3*a1**6*np.sin(2*(ang1*k1 + phi1)))/k1 - (15*a0**2*a1**4*np.sin(4*(ang1*k1 + phi1)))/k1 - (3*a1**6*np.sin(4*(ang1*k1 + phi1)))/k1 - (a1**6*np.sin(6*(ang1*k1 + phi1)))/k1 + (24*a0**5*a1*np.sin(ang1*(k0 + k1) + phi0 + phi1))/(k0 + k1) + (72*a0**3*a1**3*np.sin(ang1*(k0 + k1) + phi0 + phi1))/(k0 + k1) + (24*a0*a1**5*np.sin(ang1*(k0 + k1) + phi0 + phi1))/(k0 + k1) - (60*a0**4*a1**2*np.sin(2*(ang1*(k0 + k1) + phi0 + phi1)))/(k0 + k1) - (60*a0**2*a1**4*np.sin(2*(ang1*(k0 + k1) + phi0 + phi1)))/(k0 + k1) - (40*a0**3*a1**3*np.sin(3*(ang1*(k0 + k1) + phi0 + phi1)))/(k0 + k1) - (45*a0**4*a1**2*np.sin(2*(2*ang1*k0 + ang1*k1 + 2*phi0 + phi1)))/(2*k0 + k1) - (60*a0**5*a1*np.sin(3*ang1*k0 + ang1*k1 + 3*phi0 + phi1))/(3*k0 + k1) - (120*a0**3*a1**3*np.sin(3*ang1*k0 + ang1*k1 + 3*phi0 + phi1))/(3*k0 + k1) - (36*a0**5*a1*np.sin(5*ang1*k0 + ang1*k1 + 5*phi0 + phi1))/(5*k0 + k1) - (45*a0**2*a1**4*np.sin(2*(ang1*k0 + 2*ang1*k1 + phi0 + 2*phi1)))/(k0 + 2*k1) - (120*a0**3*a1**3*np.sin(ang1*k0 + 3*ang1*k1 + phi0 + 3*phi1))/(k0 + 3*k1) - (60*a0*a1**5*np.sin(ang1*k0 + 3*ang1*k1 + phi0 + 3*phi1))/(k0 + 3*k1) - (36*a0*a1**5*np.sin(ang1*k0 + 5*ang1*k1 + phi0 + 5*phi1))/(k0 + 5*k1))/192.)
            return res1 - res0
        elif j == 3:
            res0 = (((-105*a0*(a0**6 + 12*a0**4*a1**2 + 18*a0**2*a1**4 + 4*a1**6)*np.cos(ang0*k0 + phi0))/k0 - (35*(a0**7 + 10*a0**5*a1**2 + 10*a0**3*a1**4)*np.cos(3*(ang0*k0 + phi0)))/k0 + (7*a0**7*np.cos(5*(ang0*k0 + phi0)))/k0 + (42*a0**5*a1**2*np.cos(5*(ang0*k0 + phi0)))/k0 + (5*a0**7*np.cos(7*(ang0*k0 + phi0)))/k0 - (35*a0*a1**6*np.cos(ang0*k0 - 6*ang0*k1 + phi0 - 6*phi1))/(k0 - 6*k1) + (105*a0**2*a1**5*np.cos(2*ang0*k0 - 5*ang0*k1 + 2*phi0 - 5*phi1))/(2*k0 - 5*k1) + (525*a0**3*a1**4*np.cos(ang0*k0 - 4*ang0*k1 + phi0 - 4*phi1))/(k0 - 4*k1) + (210*a0*a1**6*np.cos(ang0*k0 - 4*ang0*k1 + phi0 - 4*phi1))/(k0 - 4*k1) + (105*a0**3*a1**4*np.cos(3*ang0*k0 - 4*ang0*k1 + 3*phi0 - 4*phi1))/(3*k0 - 4*k1) + (420*a0**4*a1**3*np.cos(2*ang0*k0 - 3*ang0*k1 + 2*phi0 - 3*phi1))/(2*k0 - 3*k1) + (315*a0**2*a1**5*np.cos(2*ang0*k0 - 3*ang0*k1 + 2*phi0 - 3*phi1))/(2*k0 - 3*k1) - (105*a0**4*a1**3*np.cos(4*ang0*k0 - 3*ang0*k1 + 4*phi0 - 3*phi1))/(4*k0 - 3*k1) + (630*a0**5*a1**2*np.cos(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1))/(k0 - 2*k1) + (1260*a0**3*a1**4*np.cos(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1))/(k0 - 2*k1) + (315*a0*a1**6*np.cos(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1))/(k0 - 2*k1) - (315*a0**5*a1**2*np.cos(3*ang0*k0 - 2*ang0*k1 + 3*phi0 - 2*phi1))/(3*k0 - 2*k1) - (420*a0**3*a1**4*np.cos(3*ang0*k0 - 2*ang0*k1 + 3*phi0 - 2*phi1))/(3*k0 - 2*k1) - (105*a0**5*a1**2*np.cos(5*ang0*k0 - 2*ang0*k1 + 5*phi0 - 2*phi1))/(5*k0 - 2*k1) - (315*a0**6*a1*np.cos(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1))/(2*k0 - k1) - (1260*a0**4*a1**3*np.cos(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1))/(2*k0 - k1) - (630*a0**2*a1**5*np.cos(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1))/(2*k0 - k1) - (210*a0**6*a1*np.cos(4*ang0*k0 - ang0*k1 + 4*phi0 - phi1))/(4*k0 - k1) - (525*a0**4*a1**3*np.cos(4*ang0*k0 - ang0*k1 + 4*phi0 - phi1))/(4*k0 - k1) + (35*a0**6*a1*np.cos(6*ang0*k0 - ang0*k1 + 6*phi0 - phi1))/(6*k0 - k1) - (420*a0**6*a1*np.cos(ang0*k1 + phi1))/k1 - (1890*a0**4*a1**3*np.cos(ang0*k1 + phi1))/k1 - (1260*a0**2*a1**5*np.cos(ang0*k1 + phi1))/k1 - (105*a1**7*np.cos(ang0*k1 + phi1))/k1 - (350*a0**4*a1**3*np.cos(3*(ang0*k1 + phi1)))/k1 - (350*a0**2*a1**5*np.cos(3*(ang0*k1 + phi1)))/k1 - (35*a1**7*np.cos(3*(ang0*k1 + phi1)))/k1 + (42*a0**2*a1**5*np.cos(5*(ang0*k1 + phi1)))/k1 + (7*a1**7*np.cos(5*(ang0*k1 + phi1)))/k1 + (5*a1**7*np.cos(7*(ang0*k1 + phi1)))/k1 - (525*a0**6*a1*np.cos(2*ang0*k0 + ang0*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (2100*a0**4*a1**3*np.cos(2*ang0*k0 + ang0*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (1050*a0**2*a1**5*np.cos(2*ang0*k0 + ang0*k1 + 2*phi0 + phi1))/(2*k0 + k1) + (210*a0**6*a1*np.cos(4*ang0*k0 + ang0*k1 + 4*phi0 + phi1))/(4*k0 + k1) + (525*a0**4*a1**3*np.cos(4*ang0*k0 + ang0*k1 + 4*phi0 + phi1))/(4*k0 + k1) + (245*a0**6*a1*np.cos(6*ang0*k0 + ang0*k1 + 6*phi0 + phi1))/(6*k0 + k1) - (1050*a0**5*a1**2*np.cos(ang0*k0 + 2*ang0*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (2100*a0**3*a1**4*np.cos(ang0*k0 + 2*ang0*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (525*a0*a1**6*np.cos(ang0*k0 + 2*ang0*k1 + phi0 + 2*phi1))/(k0 + 2*k1) + (525*a0**5*a1**2*np.cos(3*ang0*k0 + 2*ang0*k1 + 3*phi0 + 2*phi1))/(3*k0 + 2*k1) + (700*a0**3*a1**4*np.cos(3*ang0*k0 + 2*ang0*k1 + 3*phi0 + 2*phi1))/(3*k0 + 2*k1) + (735*a0**5*a1**2*np.cos(5*ang0*k0 + 2*ang0*k1 + 5*phi0 + 2*phi1))/(5*k0 + 2*k1) + (700*a0**4*a1**3*np.cos(2*ang0*k0 + 3*ang0*k1 + 2*phi0 + 3*phi1))/(2*k0 + 3*k1) + (525*a0**2*a1**5*np.cos(2*ang0*k0 + 3*ang0*k1 + 2*phi0 + 3*phi1))/(2*k0 + 3*k1) + (1225*a0**4*a1**3*np.cos(4*ang0*k0 + 3*ang0*k1 + 4*phi0 + 3*phi1))/(4*k0 + 3*k1) + (525*a0**3*a1**4*np.cos(ang0*k0 + 4*ang0*k1 + phi0 + 4*phi1))/(k0 + 4*k1) + (210*a0*a1**6*np.cos(ang0*k0 + 4*ang0*k1 + phi0 + 4*phi1))/(k0 + 4*k1) + (1225*a0**3*a1**4*np.cos(3*ang0*k0 + 4*ang0*k1 + 3*phi0 + 4*phi1))/(3*k0 + 4*k1) + (735*a0**2*a1**5*np.cos(2*ang0*k0 + 5*ang0*k1 + 2*phi0 + 5*phi1))/(2*k0 + 5*k1) + (245*a0*a1**6*np.cos(ang0*k0 + 6*ang0*k1 + phi0 + 6*phi1))/(k0 + 6*k1))/2240.)
            res1 = (((-105*a0*(a0**6 + 12*a0**4*a1**2 + 18*a0**2*a1**4 + 4*a1**6)*np.cos(ang1*k0 + phi0))/k0 - (35*(a0**7 + 10*a0**5*a1**2 + 10*a0**3*a1**4)*np.cos(3*(ang1*k0 + phi0)))/k0 + (7*a0**7*np.cos(5*(ang1*k0 + phi0)))/k0 + (42*a0**5*a1**2*np.cos(5*(ang1*k0 + phi0)))/k0 + (5*a0**7*np.cos(7*(ang1*k0 + phi0)))/k0 - (35*a0*a1**6*np.cos(ang1*k0 - 6*ang1*k1 + phi0 - 6*phi1))/(k0 - 6*k1) + (105*a0**2*a1**5*np.cos(2*ang1*k0 - 5*ang1*k1 + 2*phi0 - 5*phi1))/(2*k0 - 5*k1) + (525*a0**3*a1**4*np.cos(ang1*k0 - 4*ang1*k1 + phi0 - 4*phi1))/(k0 - 4*k1) + (210*a0*a1**6*np.cos(ang1*k0 - 4*ang1*k1 + phi0 - 4*phi1))/(k0 - 4*k1) + (105*a0**3*a1**4*np.cos(3*ang1*k0 - 4*ang1*k1 + 3*phi0 - 4*phi1))/(3*k0 - 4*k1) + (420*a0**4*a1**3*np.cos(2*ang1*k0 - 3*ang1*k1 + 2*phi0 - 3*phi1))/(2*k0 - 3*k1) + (315*a0**2*a1**5*np.cos(2*ang1*k0 - 3*ang1*k1 + 2*phi0 - 3*phi1))/(2*k0 - 3*k1) - (105*a0**4*a1**3*np.cos(4*ang1*k0 - 3*ang1*k1 + 4*phi0 - 3*phi1))/(4*k0 - 3*k1) + (630*a0**5*a1**2*np.cos(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1))/(k0 - 2*k1) + (1260*a0**3*a1**4*np.cos(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1))/(k0 - 2*k1) + (315*a0*a1**6*np.cos(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1))/(k0 - 2*k1) - (315*a0**5*a1**2*np.cos(3*ang1*k0 - 2*ang1*k1 + 3*phi0 - 2*phi1))/(3*k0 - 2*k1) - (420*a0**3*a1**4*np.cos(3*ang1*k0 - 2*ang1*k1 + 3*phi0 - 2*phi1))/(3*k0 - 2*k1) - (105*a0**5*a1**2*np.cos(5*ang1*k0 - 2*ang1*k1 + 5*phi0 - 2*phi1))/(5*k0 - 2*k1) - (315*a0**6*a1*np.cos(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1))/(2*k0 - k1) - (1260*a0**4*a1**3*np.cos(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1))/(2*k0 - k1) - (630*a0**2*a1**5*np.cos(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1))/(2*k0 - k1) - (210*a0**6*a1*np.cos(4*ang1*k0 - ang1*k1 + 4*phi0 - phi1))/(4*k0 - k1) - (525*a0**4*a1**3*np.cos(4*ang1*k0 - ang1*k1 + 4*phi0 - phi1))/(4*k0 - k1) + (35*a0**6*a1*np.cos(6*ang1*k0 - ang1*k1 + 6*phi0 - phi1))/(6*k0 - k1) - (420*a0**6*a1*np.cos(ang1*k1 + phi1))/k1 - (1890*a0**4*a1**3*np.cos(ang1*k1 + phi1))/k1 - (1260*a0**2*a1**5*np.cos(ang1*k1 + phi1))/k1 - (105*a1**7*np.cos(ang1*k1 + phi1))/k1 - (350*a0**4*a1**3*np.cos(3*(ang1*k1 + phi1)))/k1 - (350*a0**2*a1**5*np.cos(3*(ang1*k1 + phi1)))/k1 - (35*a1**7*np.cos(3*(ang1*k1 + phi1)))/k1 + (42*a0**2*a1**5*np.cos(5*(ang1*k1 + phi1)))/k1 + (7*a1**7*np.cos(5*(ang1*k1 + phi1)))/k1 + (5*a1**7*np.cos(7*(ang1*k1 + phi1)))/k1 - (525*a0**6*a1*np.cos(2*ang1*k0 + ang1*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (2100*a0**4*a1**3*np.cos(2*ang1*k0 + ang1*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (1050*a0**2*a1**5*np.cos(2*ang1*k0 + ang1*k1 + 2*phi0 + phi1))/(2*k0 + k1) + (210*a0**6*a1*np.cos(4*ang1*k0 + ang1*k1 + 4*phi0 + phi1))/(4*k0 + k1) + (525*a0**4*a1**3*np.cos(4*ang1*k0 + ang1*k1 + 4*phi0 + phi1))/(4*k0 + k1) + (245*a0**6*a1*np.cos(6*ang1*k0 + ang1*k1 + 6*phi0 + phi1))/(6*k0 + k1) - (1050*a0**5*a1**2*np.cos(ang1*k0 + 2*ang1*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (2100*a0**3*a1**4*np.cos(ang1*k0 + 2*ang1*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (525*a0*a1**6*np.cos(ang1*k0 + 2*ang1*k1 + phi0 + 2*phi1))/(k0 + 2*k1) + (525*a0**5*a1**2*np.cos(3*ang1*k0 + 2*ang1*k1 + 3*phi0 + 2*phi1))/(3*k0 + 2*k1) + (700*a0**3*a1**4*np.cos(3*ang1*k0 + 2*ang1*k1 + 3*phi0 + 2*phi1))/(3*k0 + 2*k1) + (735*a0**5*a1**2*np.cos(5*ang1*k0 + 2*ang1*k1 + 5*phi0 + 2*phi1))/(5*k0 + 2*k1) + (700*a0**4*a1**3*np.cos(2*ang1*k0 + 3*ang1*k1 + 2*phi0 + 3*phi1))/(2*k0 + 3*k1) + (525*a0**2*a1**5*np.cos(2*ang1*k0 + 3*ang1*k1 + 2*phi0 + 3*phi1))/(2*k0 + 3*k1) + (1225*a0**4*a1**3*np.cos(4*ang1*k0 + 3*ang1*k1 + 4*phi0 + 3*phi1))/(4*k0 + 3*k1) + (525*a0**3*a1**4*np.cos(ang1*k0 + 4*ang1*k1 + phi0 + 4*phi1))/(k0 + 4*k1) + (210*a0*a1**6*np.cos(ang1*k0 + 4*ang1*k1 + phi0 + 4*phi1))/(k0 + 4*k1) + (1225*a0**3*a1**4*np.cos(3*ang1*k0 + 4*ang1*k1 + 3*phi0 + 4*phi1))/(3*k0 + 4*k1) + (735*a0**2*a1**5*np.cos(2*ang1*k0 + 5*ang1*k1 + 2*phi0 + 5*phi1))/(2*k0 + 5*k1) + (245*a0*a1**6*np.cos(ang1*k0 + 6*ang1*k1 + phi0 + 6*phi1))/(k0 + 6*k1))/2240.)
            return res1 - res0
        elif j == 4:
            res0 = ((24*a0**8*ang0 + 384*a0**6*a1**2*ang0 + 864*a0**4*a1**4*ang0 + 384*a0**2*a1**6*ang0 + 24*a1**8*ang0 - (8*(a0**8 + 12*a0**6*a1**2 + 15*a0**4*a1**4)*np.sin(4*(ang0*k0 + phi0)))/k0 + (a0**8*np.sin(8*(ang0*k0 + phi0)))/k0 - (192*a0**3*a1**5*np.sin(ang0*k0 - 5*ang0*k1 + phi0 - 5*phi1))/(k0 - 5*k1) - (64*a0*a1**7*np.sin(ang0*k0 - 5*ang0*k1 + phi0 - 5*phi1))/(k0 - 5*k1) - (16*a0**2*a1**6*np.sin(2*(ang0*k0 - 3*ang0*k1 + phi0 - 3*phi1)))/(k0 - 3*k1) - (16*a0**6*a1**2*np.sin(6*ang0*k0 - 2*ang0*k1 + 6*phi0 - 2*phi1))/(3*k0 - k1) + (192*a0**7*a1*np.sin(ang0*k0 - ang0*k1 + phi0 - phi1))/(k0 - k1) + (1152*a0**5*a1**3*np.sin(ang0*k0 - ang0*k1 + phi0 - phi1))/(k0 - k1) + (1152*a0**3*a1**5*np.sin(ang0*k0 - ang0*k1 + phi0 - phi1))/(k0 - k1) + (192*a0*a1**7*np.sin(ang0*k0 - ang0*k1 + phi0 - phi1))/(k0 - k1) + (144*a0**6*a1**2*np.sin(2*(ang0*k0 - ang0*k1 + phi0 - phi1)))/(k0 - k1) + (384*a0**4*a1**4*np.sin(2*(ang0*k0 - ang0*k1 + phi0 - phi1)))/(k0 - k1) + (144*a0**2*a1**6*np.sin(2*(ang0*k0 - ang0*k1 + phi0 - phi1)))/(k0 - k1) + (64*a0**5*a1**3*np.sin(3*(ang0*k0 - ang0*k1 + phi0 - phi1)))/(k0 - k1) + (64*a0**3*a1**5*np.sin(3*(ang0*k0 - ang0*k1 + phi0 - phi1)))/(k0 - k1) + (12*a0**4*a1**4*np.sin(4*(ang0*k0 - ang0*k1 + phi0 - phi1)))/(k0 - k1) - (64*a0**7*a1*np.sin(5*ang0*k0 - ang0*k1 + 5*phi0 - phi1))/(5*k0 - k1) - (192*a0**5*a1**3*np.sin(5*ang0*k0 - ang0*k1 + 5*phi0 - phi1))/(5*k0 - k1) - (120*a0**4*a1**4*np.sin(4*(ang0*k1 + phi1)))/k1 - (96*a0**2*a1**6*np.sin(4*(ang0*k1 + phi1)))/k1 - (8*a1**8*np.sin(4*(ang0*k1 + phi1)))/k1 + (a1**8*np.sin(8*(ang0*k1 + phi1)))/k1 - (240*a0**6*a1**2*np.sin(2*(ang0*(k0 + k1) + phi0 + phi1)))/(k0 + k1) - (640*a0**4*a1**4*np.sin(2*(ang0*(k0 + k1) + phi0 + phi1)))/(k0 + k1) - (240*a0**2*a1**6*np.sin(2*(ang0*(k0 + k1) + phi0 + phi1)))/(k0 + k1) + (140*a0**4*a1**4*np.sin(4*(ang0*(k0 + k1) + phi0 + phi1)))/(k0 + k1) - (192*a0**7*a1*np.sin(3*ang0*k0 + ang0*k1 + 3*phi0 + phi1))/(3*k0 + k1) - (960*a0**5*a1**3*np.sin(3*ang0*k0 + ang0*k1 + 3*phi0 + phi1))/(3*k0 + k1) - (640*a0**3*a1**5*np.sin(3*ang0*k0 + ang0*k1 + 3*phi0 + phi1))/(3*k0 + k1) + (112*a0**6*a1**2*np.sin(2*(3*ang0*k0 + ang0*k1 + 3*phi0 + phi1)))/(3*k0 + k1) + (64*a0**7*a1*np.sin(7*ang0*k0 + ang0*k1 + 7*phi0 + phi1))/(7*k0 + k1) - (640*a0**5*a1**3*np.sin(ang0*k0 + 3*ang0*k1 + phi0 + 3*phi1))/(k0 + 3*k1) - (960*a0**3*a1**5*np.sin(ang0*k0 + 3*ang0*k1 + phi0 + 3*phi1))/(k0 + 3*k1) - (192*a0*a1**7*np.sin(ang0*k0 + 3*ang0*k1 + phi0 + 3*phi1))/(k0 + 3*k1) + (112*a0**2*a1**6*np.sin(2*(ang0*k0 + 3*ang0*k1 + phi0 + 3*phi1)))/(k0 + 3*k1) + (448*a0**5*a1**3*np.sin(5*ang0*k0 + 3*ang0*k1 + 5*phi0 + 3*phi1))/(5*k0 + 3*k1) + (448*a0**3*a1**5*np.sin(3*ang0*k0 + 5*ang0*k1 + 3*phi0 + 5*phi1))/(3*k0 + 5*k1) + (64*a0*a1**7*np.sin(ang0*k0 + 7*ang0*k1 + phi0 + 7*phi1))/(k0 + 7*k1))/1024.)
            res1 = ((24*a0**8*ang1 + 384*a0**6*a1**2*ang1 + 864*a0**4*a1**4*ang1 + 384*a0**2*a1**6*ang1 + 24*a1**8*ang1 - (8*(a0**8 + 12*a0**6*a1**2 + 15*a0**4*a1**4)*np.sin(4*(ang1*k0 + phi0)))/k0 + (a0**8*np.sin(8*(ang1*k0 + phi0)))/k0 - (192*a0**3*a1**5*np.sin(ang1*k0 - 5*ang1*k1 + phi0 - 5*phi1))/(k0 - 5*k1) - (64*a0*a1**7*np.sin(ang1*k0 - 5*ang1*k1 + phi0 - 5*phi1))/(k0 - 5*k1) - (16*a0**2*a1**6*np.sin(2*(ang1*k0 - 3*ang1*k1 + phi0 - 3*phi1)))/(k0 - 3*k1) - (16*a0**6*a1**2*np.sin(6*ang1*k0 - 2*ang1*k1 + 6*phi0 - 2*phi1))/(3*k0 - k1) + (192*a0**7*a1*np.sin(ang1*k0 - ang1*k1 + phi0 - phi1))/(k0 - k1) + (1152*a0**5*a1**3*np.sin(ang1*k0 - ang1*k1 + phi0 - phi1))/(k0 - k1) + (1152*a0**3*a1**5*np.sin(ang1*k0 - ang1*k1 + phi0 - phi1))/(k0 - k1) + (192*a0*a1**7*np.sin(ang1*k0 - ang1*k1 + phi0 - phi1))/(k0 - k1) + (144*a0**6*a1**2*np.sin(2*(ang1*k0 - ang1*k1 + phi0 - phi1)))/(k0 - k1) + (384*a0**4*a1**4*np.sin(2*(ang1*k0 - ang1*k1 + phi0 - phi1)))/(k0 - k1) + (144*a0**2*a1**6*np.sin(2*(ang1*k0 - ang1*k1 + phi0 - phi1)))/(k0 - k1) + (64*a0**5*a1**3*np.sin(3*(ang1*k0 - ang1*k1 + phi0 - phi1)))/(k0 - k1) + (64*a0**3*a1**5*np.sin(3*(ang1*k0 - ang1*k1 + phi0 - phi1)))/(k0 - k1) + (12*a0**4*a1**4*np.sin(4*(ang1*k0 - ang1*k1 + phi0 - phi1)))/(k0 - k1) - (64*a0**7*a1*np.sin(5*ang1*k0 - ang1*k1 + 5*phi0 - phi1))/(5*k0 - k1) - (192*a0**5*a1**3*np.sin(5*ang1*k0 - ang1*k1 + 5*phi0 - phi1))/(5*k0 - k1) - (120*a0**4*a1**4*np.sin(4*(ang1*k1 + phi1)))/k1 - (96*a0**2*a1**6*np.sin(4*(ang1*k1 + phi1)))/k1 - (8*a1**8*np.sin(4*(ang1*k1 + phi1)))/k1 + (a1**8*np.sin(8*(ang1*k1 + phi1)))/k1 - (240*a0**6*a1**2*np.sin(2*(ang1*(k0 + k1) + phi0 + phi1)))/(k0 + k1) - (640*a0**4*a1**4*np.sin(2*(ang1*(k0 + k1) + phi0 + phi1)))/(k0 + k1) - (240*a0**2*a1**6*np.sin(2*(ang1*(k0 + k1) + phi0 + phi1)))/(k0 + k1) + (140*a0**4*a1**4*np.sin(4*(ang1*(k0 + k1) + phi0 + phi1)))/(k0 + k1) - (192*a0**7*a1*np.sin(3*ang1*k0 + ang1*k1 + 3*phi0 + phi1))/(3*k0 + k1) - (960*a0**5*a1**3*np.sin(3*ang1*k0 + ang1*k1 + 3*phi0 + phi1))/(3*k0 + k1) - (640*a0**3*a1**5*np.sin(3*ang1*k0 + ang1*k1 + 3*phi0 + phi1))/(3*k0 + k1) + (112*a0**6*a1**2*np.sin(2*(3*ang1*k0 + ang1*k1 + 3*phi0 + phi1)))/(3*k0 + k1) + (64*a0**7*a1*np.sin(7*ang1*k0 + ang1*k1 + 7*phi0 + phi1))/(7*k0 + k1) - (640*a0**5*a1**3*np.sin(ang1*k0 + 3*ang1*k1 + phi0 + 3*phi1))/(k0 + 3*k1) - (960*a0**3*a1**5*np.sin(ang1*k0 + 3*ang1*k1 + phi0 + 3*phi1))/(k0 + 3*k1) - (192*a0*a1**7*np.sin(ang1*k0 + 3*ang1*k1 + phi0 + 3*phi1))/(k0 + 3*k1) + (112*a0**2*a1**6*np.sin(2*(ang1*k0 + 3*ang1*k1 + phi0 + 3*phi1)))/(k0 + 3*k1) + (448*a0**5*a1**3*np.sin(5*ang1*k0 + 3*ang1*k1 + 5*phi0 + 3*phi1))/(5*k0 + 3*k1) + (448*a0**3*a1**5*np.sin(3*ang1*k0 + 5*ang1*k1 + 3*phi0 + 5*phi1))/(3*k0 + 5*k1) + (64*a0*a1**7*np.sin(ang1*k0 + 7*ang1*k1 + phi0 + 7*phi1))/(k0 + 7*k1))/1024.)
            return res1 - res0
        elif j == 5:
            res0 = (((-1890*a0*(a0**8 + 20*a0**6*a1**2 + 60*a0**4*a1**4 + 40*a0**2*a1**6 + 5*a1**8)*np.cos(ang0*k0 + phi0))/k0 - (420*(a0**9 + 18*a0**7*a1**2 + 45*a0**5*a1**4 + 20*a0**3*a1**6)*np.cos(3*(ang0*k0 + phi0)))/k0 + (252*a0**9*np.cos(5*(ang0*k0 + phi0)))/k0 + (3528*a0**7*a1**2*np.cos(5*(ang0*k0 + phi0)))/k0 + (5292*a0**5*a1**4*np.cos(5*(ang0*k0 + phi0)))/k0 + (45*a0**9*np.cos(7*(ang0*k0 + phi0)))/k0 + (360*a0**7*a1**2*np.cos(7*(ang0*k0 + phi0)))/k0 - (35*a0**9*np.cos(9*(ang0*k0 + phi0)))/k0 - (315*a0*a1**8*np.cos(ang0*k0 - 8*ang0*k1 + phi0 - 8*phi1))/(k0 - 8*k1) - (1260*a0**2*a1**7*np.cos(2*ang0*k0 - 7*ang0*k1 + 2*phi0 - 7*phi1))/(2*k0 - 7*k1) - (8820*a0**3*a1**6*np.cos(ang0*k0 - 6*ang0*k1 + phi0 - 6*phi1))/(k0 - 6*k1) - (2520*a0*a1**8*np.cos(ang0*k0 - 6*ang0*k1 + phi0 - 6*phi1))/(k0 - 6*k1) + (7560*a0**4*a1**5*np.cos(2*ang0*k0 - 5*ang0*k1 + 2*phi0 - 5*phi1))/(2*k0 - 5*k1) + (3780*a0**2*a1**7*np.cos(2*ang0*k0 - 5*ang0*k1 + 2*phi0 - 5*phi1))/(2*k0 - 5*k1) + (1890*a0**4*a1**5*np.cos(4*ang0*k0 - 5*ang0*k1 + 4*phi0 - 5*phi1))/(4*k0 - 5*k1) + (18900*a0**5*a1**4*np.cos(ang0*k0 - 4*ang0*k1 + phi0 - 4*phi1))/(k0 - 4*k1) + (22680*a0**3*a1**6*np.cos(ang0*k0 - 4*ang0*k1 + phi0 - 4*phi1))/(k0 - 4*k1) + (3780*a0*a1**8*np.cos(ang0*k0 - 4*ang0*k1 + phi0 - 4*phi1))/(k0 - 4*k1) + (9450*a0**5*a1**4*np.cos(3*ang0*k0 - 4*ang0*k1 + 3*phi0 - 4*phi1))/(3*k0 - 4*k1) + (7560*a0**3*a1**6*np.cos(3*ang0*k0 - 4*ang0*k1 + 3*phi0 - 4*phi1))/(3*k0 - 4*k1) - (1890*a0**5*a1**4*np.cos(5*ang0*k0 - 4*ang0*k1 + 5*phi0 - 4*phi1))/(5*k0 - 4*k1) + (18900*a0**6*a1**3*np.cos(2*ang0*k0 - 3*ang0*k1 + 2*phi0 - 3*phi1))/(2*k0 - 3*k1) + (37800*a0**4*a1**5*np.cos(2*ang0*k0 - 3*ang0*k1 + 2*phi0 - 3*phi1))/(2*k0 - 3*k1) + (11340*a0**2*a1**7*np.cos(2*ang0*k0 - 3*ang0*k1 + 2*phi0 - 3*phi1))/(2*k0 - 3*k1) - (7560*a0**6*a1**3*np.cos(4*ang0*k0 - 3*ang0*k1 + 4*phi0 - 3*phi1))/(4*k0 - 3*k1) - (9450*a0**4*a1**5*np.cos(4*ang0*k0 - 3*ang0*k1 + 4*phi0 - 3*phi1))/(4*k0 - 3*k1) - (420*a0**6*a1**3*np.cos(6*ang0*k0 - 3*ang0*k1 + 6*phi0 - 3*phi1))/(2*k0 - k1) + (18900*a0**7*a1**2*np.cos(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1))/(k0 - 2*k1) + (75600*a0**5*a1**4*np.cos(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1))/(k0 - 2*k1) + (56700*a0**3*a1**6*np.cos(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1))/(k0 - 2*k1) + (7560*a0*a1**8*np.cos(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1))/(k0 - 2*k1) + (420*a0**3*a1**6*np.cos(3*(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1)))/(k0 - 2*k1) - (11340*a0**7*a1**2*np.cos(3*ang0*k0 - 2*ang0*k1 + 3*phi0 - 2*phi1))/(3*k0 - 2*k1) - (37800*a0**5*a1**4*np.cos(3*ang0*k0 - 2*ang0*k1 + 3*phi0 - 2*phi1))/(3*k0 - 2*k1) - (18900*a0**3*a1**6*np.cos(3*ang0*k0 - 2*ang0*k1 + 3*phi0 - 2*phi1))/(3*k0 - 2*k1) - (3780*a0**7*a1**2*np.cos(5*ang0*k0 - 2*ang0*k1 + 5*phi0 - 2*phi1))/(5*k0 - 2*k1) - (7560*a0**5*a1**4*np.cos(5*ang0*k0 - 2*ang0*k1 + 5*phi0 - 2*phi1))/(5*k0 - 2*k1) + (1260*a0**7*a1**2*np.cos(7*ang0*k0 - 2*ang0*k1 + 7*phi0 - 2*phi1))/(7*k0 - 2*k1) - (7560*a0**8*a1*np.cos(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1))/(2*k0 - k1) - (56700*a0**6*a1**3*np.cos(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1))/(2*k0 - k1) - (75600*a0**4*a1**5*np.cos(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1))/(2*k0 - k1) - (18900*a0**2*a1**7*np.cos(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1))/(2*k0 - k1) - (3780*a0**8*a1*np.cos(4*ang0*k0 - ang0*k1 + 4*phi0 - phi1))/(4*k0 - k1) - (22680*a0**6*a1**3*np.cos(4*ang0*k0 - ang0*k1 + 4*phi0 - phi1))/(4*k0 - k1) - (18900*a0**4*a1**5*np.cos(4*ang0*k0 - ang0*k1 + 4*phi0 - phi1))/(4*k0 - k1) + (2520*a0**8*a1*np.cos(6*ang0*k0 - ang0*k1 + 6*phi0 - phi1))/(6*k0 - k1) + (8820*a0**6*a1**3*np.cos(6*ang0*k0 - ang0*k1 + 6*phi0 - phi1))/(6*k0 - k1) + (315*a0**8*a1*np.cos(8*ang0*k0 - ang0*k1 + 8*phi0 - phi1))/(8*k0 - k1) - (9450*a0**8*a1*np.cos(ang0*k1 + phi1))/k1 - (75600*a0**6*a1**3*np.cos(ang0*k1 + phi1))/k1 - (113400*a0**4*a1**5*np.cos(ang0*k1 + phi1))/k1 - (37800*a0**2*a1**7*np.cos(ang0*k1 + phi1))/k1 - (1890*a1**9*np.cos(ang0*k1 + phi1))/k1 - (8400*a0**6*a1**3*np.cos(3*(ang0*k1 + phi1)))/k1 - (18900*a0**4*a1**5*np.cos(3*(ang0*k1 + phi1)))/k1 - (7560*a0**2*a1**7*np.cos(3*(ang0*k1 + phi1)))/k1 - (420*a1**9*np.cos(3*(ang0*k1 + phi1)))/k1 + (5292*a0**4*a1**5*np.cos(5*(ang0*k1 + phi1)))/k1 + (3528*a0**2*a1**7*np.cos(5*(ang0*k1 + phi1)))/k1 + (252*a1**9*np.cos(5*(ang0*k1 + phi1)))/k1 + (360*a0**2*a1**7*np.cos(7*(ang0*k1 + phi1)))/k1 + (45*a1**9*np.cos(7*(ang0*k1 + phi1)))/k1 - (35*a1**9*np.cos(9*(ang0*k1 + phi1)))/k1 - (7560*a0**8*a1*np.cos(2*ang0*k0 + ang0*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (56700*a0**6*a1**3*np.cos(2*ang0*k0 + ang0*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (75600*a0**4*a1**5*np.cos(2*ang0*k0 + ang0*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (18900*a0**2*a1**7*np.cos(2*ang0*k0 + ang0*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (8820*a0**6*a1**3*np.cos(3*(2*ang0*k0 + ang0*k1 + 2*phi0 + phi1)))/(2*k0 + k1) + (8820*a0**8*a1*np.cos(4*ang0*k0 + ang0*k1 + 4*phi0 + phi1))/(4*k0 + k1) + (52920*a0**6*a1**3*np.cos(4*ang0*k0 + ang0*k1 + 4*phi0 + phi1))/(4*k0 + k1) + (44100*a0**4*a1**5*np.cos(4*ang0*k0 + ang0*k1 + 4*phi0 + phi1))/(4*k0 + k1) + (2520*a0**8*a1*np.cos(6*ang0*k0 + ang0*k1 + 6*phi0 + phi1))/(6*k0 + k1) + (8820*a0**6*a1**3*np.cos(6*ang0*k0 + ang0*k1 + 6*phi0 + phi1))/(6*k0 + k1) - (2835*a0**8*a1*np.cos(8*ang0*k0 + ang0*k1 + 8*phi0 + phi1))/(8*k0 + k1) - (18900*a0**7*a1**2*np.cos(ang0*k0 + 2*ang0*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (75600*a0**5*a1**4*np.cos(ang0*k0 + 2*ang0*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (56700*a0**3*a1**6*np.cos(ang0*k0 + 2*ang0*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (7560*a0*a1**8*np.cos(ang0*k0 + 2*ang0*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (8820*a0**3*a1**6*np.cos(3*(ang0*k0 + 2*ang0*k1 + phi0 + 2*phi1)))/(k0 + 2*k1) + (26460*a0**7*a1**2*np.cos(3*ang0*k0 + 2*ang0*k1 + 3*phi0 + 2*phi1))/(3*k0 + 2*k1) + (88200*a0**5*a1**4*np.cos(3*ang0*k0 + 2*ang0*k1 + 3*phi0 + 2*phi1))/(3*k0 + 2*k1) + (44100*a0**3*a1**6*np.cos(3*ang0*k0 + 2*ang0*k1 + 3*phi0 + 2*phi1))/(3*k0 + 2*k1) + (8820*a0**7*a1**2*np.cos(5*ang0*k0 + 2*ang0*k1 + 5*phi0 + 2*phi1))/(5*k0 + 2*k1) + (17640*a0**5*a1**4*np.cos(5*ang0*k0 + 2*ang0*k1 + 5*phi0 + 2*phi1))/(5*k0 + 2*k1) - (11340*a0**7*a1**2*np.cos(7*ang0*k0 + 2*ang0*k1 + 7*phi0 + 2*phi1))/(7*k0 + 2*k1) + (44100*a0**6*a1**3*np.cos(2*ang0*k0 + 3*ang0*k1 + 2*phi0 + 3*phi1))/(2*k0 + 3*k1) + (88200*a0**4*a1**5*np.cos(2*ang0*k0 + 3*ang0*k1 + 2*phi0 + 3*phi1))/(2*k0 + 3*k1) + (26460*a0**2*a1**7*np.cos(2*ang0*k0 + 3*ang0*k1 + 2*phi0 + 3*phi1))/(2*k0 + 3*k1) + (17640*a0**6*a1**3*np.cos(4*ang0*k0 + 3*ang0*k1 + 4*phi0 + 3*phi1))/(4*k0 + 3*k1) + (22050*a0**4*a1**5*np.cos(4*ang0*k0 + 3*ang0*k1 + 4*phi0 + 3*phi1))/(4*k0 + 3*k1) + (44100*a0**5*a1**4*np.cos(ang0*k0 + 4*ang0*k1 + phi0 + 4*phi1))/(k0 + 4*k1) + (52920*a0**3*a1**6*np.cos(ang0*k0 + 4*ang0*k1 + phi0 + 4*phi1))/(k0 + 4*k1) + (8820*a0*a1**8*np.cos(ang0*k0 + 4*ang0*k1 + phi0 + 4*phi1))/(k0 + 4*k1) + (22050*a0**5*a1**4*np.cos(3*ang0*k0 + 4*ang0*k1 + 3*phi0 + 4*phi1))/(3*k0 + 4*k1) + (17640*a0**3*a1**6*np.cos(3*ang0*k0 + 4*ang0*k1 + 3*phi0 + 4*phi1))/(3*k0 + 4*k1) - (39690*a0**5*a1**4*np.cos(5*ang0*k0 + 4*ang0*k1 + 5*phi0 + 4*phi1))/(5*k0 + 4*k1) + (17640*a0**4*a1**5*np.cos(2*ang0*k0 + 5*ang0*k1 + 2*phi0 + 5*phi1))/(2*k0 + 5*k1) + (8820*a0**2*a1**7*np.cos(2*ang0*k0 + 5*ang0*k1 + 2*phi0 + 5*phi1))/(2*k0 + 5*k1) - (39690*a0**4*a1**5*np.cos(4*ang0*k0 + 5*ang0*k1 + 4*phi0 + 5*phi1))/(4*k0 + 5*k1) + (8820*a0**3*a1**6*np.cos(ang0*k0 + 6*ang0*k1 + phi0 + 6*phi1))/(k0 + 6*k1) + (2520*a0*a1**8*np.cos(ang0*k0 + 6*ang0*k1 + phi0 + 6*phi1))/(k0 + 6*k1) - (11340*a0**2*a1**7*np.cos(2*ang0*k0 + 7*ang0*k1 + 2*phi0 + 7*phi1))/(2*k0 + 7*k1) - (2835*a0*a1**8*np.cos(ang0*k0 + 8*ang0*k1 + phi0 + 8*phi1))/(k0 + 8*k1))/80640.)
            res1 = (((-1890*a0*(a0**8 + 20*a0**6*a1**2 + 60*a0**4*a1**4 + 40*a0**2*a1**6 + 5*a1**8)*np.cos(ang1*k0 + phi0))/k0 - (420*(a0**9 + 18*a0**7*a1**2 + 45*a0**5*a1**4 + 20*a0**3*a1**6)*np.cos(3*(ang1*k0 + phi0)))/k0 + (252*a0**9*np.cos(5*(ang1*k0 + phi0)))/k0 + (3528*a0**7*a1**2*np.cos(5*(ang1*k0 + phi0)))/k0 + (5292*a0**5*a1**4*np.cos(5*(ang1*k0 + phi0)))/k0 + (45*a0**9*np.cos(7*(ang1*k0 + phi0)))/k0 + (360*a0**7*a1**2*np.cos(7*(ang1*k0 + phi0)))/k0 - (35*a0**9*np.cos(9*(ang1*k0 + phi0)))/k0 - (315*a0*a1**8*np.cos(ang1*k0 - 8*ang1*k1 + phi0 - 8*phi1))/(k0 - 8*k1) - (1260*a0**2*a1**7*np.cos(2*ang1*k0 - 7*ang1*k1 + 2*phi0 - 7*phi1))/(2*k0 - 7*k1) - (8820*a0**3*a1**6*np.cos(ang1*k0 - 6*ang1*k1 + phi0 - 6*phi1))/(k0 - 6*k1) - (2520*a0*a1**8*np.cos(ang1*k0 - 6*ang1*k1 + phi0 - 6*phi1))/(k0 - 6*k1) + (7560*a0**4*a1**5*np.cos(2*ang1*k0 - 5*ang1*k1 + 2*phi0 - 5*phi1))/(2*k0 - 5*k1) + (3780*a0**2*a1**7*np.cos(2*ang1*k0 - 5*ang1*k1 + 2*phi0 - 5*phi1))/(2*k0 - 5*k1) + (1890*a0**4*a1**5*np.cos(4*ang1*k0 - 5*ang1*k1 + 4*phi0 - 5*phi1))/(4*k0 - 5*k1) + (18900*a0**5*a1**4*np.cos(ang1*k0 - 4*ang1*k1 + phi0 - 4*phi1))/(k0 - 4*k1) + (22680*a0**3*a1**6*np.cos(ang1*k0 - 4*ang1*k1 + phi0 - 4*phi1))/(k0 - 4*k1) + (3780*a0*a1**8*np.cos(ang1*k0 - 4*ang1*k1 + phi0 - 4*phi1))/(k0 - 4*k1) + (9450*a0**5*a1**4*np.cos(3*ang1*k0 - 4*ang1*k1 + 3*phi0 - 4*phi1))/(3*k0 - 4*k1) + (7560*a0**3*a1**6*np.cos(3*ang1*k0 - 4*ang1*k1 + 3*phi0 - 4*phi1))/(3*k0 - 4*k1) - (1890*a0**5*a1**4*np.cos(5*ang1*k0 - 4*ang1*k1 + 5*phi0 - 4*phi1))/(5*k0 - 4*k1) + (18900*a0**6*a1**3*np.cos(2*ang1*k0 - 3*ang1*k1 + 2*phi0 - 3*phi1))/(2*k0 - 3*k1) + (37800*a0**4*a1**5*np.cos(2*ang1*k0 - 3*ang1*k1 + 2*phi0 - 3*phi1))/(2*k0 - 3*k1) + (11340*a0**2*a1**7*np.cos(2*ang1*k0 - 3*ang1*k1 + 2*phi0 - 3*phi1))/(2*k0 - 3*k1) - (7560*a0**6*a1**3*np.cos(4*ang1*k0 - 3*ang1*k1 + 4*phi0 - 3*phi1))/(4*k0 - 3*k1) - (9450*a0**4*a1**5*np.cos(4*ang1*k0 - 3*ang1*k1 + 4*phi0 - 3*phi1))/(4*k0 - 3*k1) - (420*a0**6*a1**3*np.cos(6*ang1*k0 - 3*ang1*k1 + 6*phi0 - 3*phi1))/(2*k0 - k1) + (18900*a0**7*a1**2*np.cos(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1))/(k0 - 2*k1) + (75600*a0**5*a1**4*np.cos(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1))/(k0 - 2*k1) + (56700*a0**3*a1**6*np.cos(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1))/(k0 - 2*k1) + (7560*a0*a1**8*np.cos(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1))/(k0 - 2*k1) + (420*a0**3*a1**6*np.cos(3*(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1)))/(k0 - 2*k1) - (11340*a0**7*a1**2*np.cos(3*ang1*k0 - 2*ang1*k1 + 3*phi0 - 2*phi1))/(3*k0 - 2*k1) - (37800*a0**5*a1**4*np.cos(3*ang1*k0 - 2*ang1*k1 + 3*phi0 - 2*phi1))/(3*k0 - 2*k1) - (18900*a0**3*a1**6*np.cos(3*ang1*k0 - 2*ang1*k1 + 3*phi0 - 2*phi1))/(3*k0 - 2*k1) - (3780*a0**7*a1**2*np.cos(5*ang1*k0 - 2*ang1*k1 + 5*phi0 - 2*phi1))/(5*k0 - 2*k1) - (7560*a0**5*a1**4*np.cos(5*ang1*k0 - 2*ang1*k1 + 5*phi0 - 2*phi1))/(5*k0 - 2*k1) + (1260*a0**7*a1**2*np.cos(7*ang1*k0 - 2*ang1*k1 + 7*phi0 - 2*phi1))/(7*k0 - 2*k1) - (7560*a0**8*a1*np.cos(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1))/(2*k0 - k1) - (56700*a0**6*a1**3*np.cos(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1))/(2*k0 - k1) - (75600*a0**4*a1**5*np.cos(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1))/(2*k0 - k1) - (18900*a0**2*a1**7*np.cos(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1))/(2*k0 - k1) - (3780*a0**8*a1*np.cos(4*ang1*k0 - ang1*k1 + 4*phi0 - phi1))/(4*k0 - k1) - (22680*a0**6*a1**3*np.cos(4*ang1*k0 - ang1*k1 + 4*phi0 - phi1))/(4*k0 - k1) - (18900*a0**4*a1**5*np.cos(4*ang1*k0 - ang1*k1 + 4*phi0 - phi1))/(4*k0 - k1) + (2520*a0**8*a1*np.cos(6*ang1*k0 - ang1*k1 + 6*phi0 - phi1))/(6*k0 - k1) + (8820*a0**6*a1**3*np.cos(6*ang1*k0 - ang1*k1 + 6*phi0 - phi1))/(6*k0 - k1) + (315*a0**8*a1*np.cos(8*ang1*k0 - ang1*k1 + 8*phi0 - phi1))/(8*k0 - k1) - (9450*a0**8*a1*np.cos(ang1*k1 + phi1))/k1 - (75600*a0**6*a1**3*np.cos(ang1*k1 + phi1))/k1 - (113400*a0**4*a1**5*np.cos(ang1*k1 + phi1))/k1 - (37800*a0**2*a1**7*np.cos(ang1*k1 + phi1))/k1 - (1890*a1**9*np.cos(ang1*k1 + phi1))/k1 - (8400*a0**6*a1**3*np.cos(3*(ang1*k1 + phi1)))/k1 - (18900*a0**4*a1**5*np.cos(3*(ang1*k1 + phi1)))/k1 - (7560*a0**2*a1**7*np.cos(3*(ang1*k1 + phi1)))/k1 - (420*a1**9*np.cos(3*(ang1*k1 + phi1)))/k1 + (5292*a0**4*a1**5*np.cos(5*(ang1*k1 + phi1)))/k1 + (3528*a0**2*a1**7*np.cos(5*(ang1*k1 + phi1)))/k1 + (252*a1**9*np.cos(5*(ang1*k1 + phi1)))/k1 + (360*a0**2*a1**7*np.cos(7*(ang1*k1 + phi1)))/k1 + (45*a1**9*np.cos(7*(ang1*k1 + phi1)))/k1 - (35*a1**9*np.cos(9*(ang1*k1 + phi1)))/k1 - (7560*a0**8*a1*np.cos(2*ang1*k0 + ang1*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (56700*a0**6*a1**3*np.cos(2*ang1*k0 + ang1*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (75600*a0**4*a1**5*np.cos(2*ang1*k0 + ang1*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (18900*a0**2*a1**7*np.cos(2*ang1*k0 + ang1*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (8820*a0**6*a1**3*np.cos(3*(2*ang1*k0 + ang1*k1 + 2*phi0 + phi1)))/(2*k0 + k1) + (8820*a0**8*a1*np.cos(4*ang1*k0 + ang1*k1 + 4*phi0 + phi1))/(4*k0 + k1) + (52920*a0**6*a1**3*np.cos(4*ang1*k0 + ang1*k1 + 4*phi0 + phi1))/(4*k0 + k1) + (44100*a0**4*a1**5*np.cos(4*ang1*k0 + ang1*k1 + 4*phi0 + phi1))/(4*k0 + k1) + (2520*a0**8*a1*np.cos(6*ang1*k0 + ang1*k1 + 6*phi0 + phi1))/(6*k0 + k1) + (8820*a0**6*a1**3*np.cos(6*ang1*k0 + ang1*k1 + 6*phi0 + phi1))/(6*k0 + k1) - (2835*a0**8*a1*np.cos(8*ang1*k0 + ang1*k1 + 8*phi0 + phi1))/(8*k0 + k1) - (18900*a0**7*a1**2*np.cos(ang1*k0 + 2*ang1*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (75600*a0**5*a1**4*np.cos(ang1*k0 + 2*ang1*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (56700*a0**3*a1**6*np.cos(ang1*k0 + 2*ang1*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (7560*a0*a1**8*np.cos(ang1*k0 + 2*ang1*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (8820*a0**3*a1**6*np.cos(3*(ang1*k0 + 2*ang1*k1 + phi0 + 2*phi1)))/(k0 + 2*k1) + (26460*a0**7*a1**2*np.cos(3*ang1*k0 + 2*ang1*k1 + 3*phi0 + 2*phi1))/(3*k0 + 2*k1) + (88200*a0**5*a1**4*np.cos(3*ang1*k0 + 2*ang1*k1 + 3*phi0 + 2*phi1))/(3*k0 + 2*k1) + (44100*a0**3*a1**6*np.cos(3*ang1*k0 + 2*ang1*k1 + 3*phi0 + 2*phi1))/(3*k0 + 2*k1) + (8820*a0**7*a1**2*np.cos(5*ang1*k0 + 2*ang1*k1 + 5*phi0 + 2*phi1))/(5*k0 + 2*k1) + (17640*a0**5*a1**4*np.cos(5*ang1*k0 + 2*ang1*k1 + 5*phi0 + 2*phi1))/(5*k0 + 2*k1) - (11340*a0**7*a1**2*np.cos(7*ang1*k0 + 2*ang1*k1 + 7*phi0 + 2*phi1))/(7*k0 + 2*k1) + (44100*a0**6*a1**3*np.cos(2*ang1*k0 + 3*ang1*k1 + 2*phi0 + 3*phi1))/(2*k0 + 3*k1) + (88200*a0**4*a1**5*np.cos(2*ang1*k0 + 3*ang1*k1 + 2*phi0 + 3*phi1))/(2*k0 + 3*k1) + (26460*a0**2*a1**7*np.cos(2*ang1*k0 + 3*ang1*k1 + 2*phi0 + 3*phi1))/(2*k0 + 3*k1) + (17640*a0**6*a1**3*np.cos(4*ang1*k0 + 3*ang1*k1 + 4*phi0 + 3*phi1))/(4*k0 + 3*k1) + (22050*a0**4*a1**5*np.cos(4*ang1*k0 + 3*ang1*k1 + 4*phi0 + 3*phi1))/(4*k0 + 3*k1) + (44100*a0**5*a1**4*np.cos(ang1*k0 + 4*ang1*k1 + phi0 + 4*phi1))/(k0 + 4*k1) + (52920*a0**3*a1**6*np.cos(ang1*k0 + 4*ang1*k1 + phi0 + 4*phi1))/(k0 + 4*k1) + (8820*a0*a1**8*np.cos(ang1*k0 + 4*ang1*k1 + phi0 + 4*phi1))/(k0 + 4*k1) + (22050*a0**5*a1**4*np.cos(3*ang1*k0 + 4*ang1*k1 + 3*phi0 + 4*phi1))/(3*k0 + 4*k1) + (17640*a0**3*a1**6*np.cos(3*ang1*k0 + 4*ang1*k1 + 3*phi0 + 4*phi1))/(3*k0 + 4*k1) - (39690*a0**5*a1**4*np.cos(5*ang1*k0 + 4*ang1*k1 + 5*phi0 + 4*phi1))/(5*k0 + 4*k1) + (17640*a0**4*a1**5*np.cos(2*ang1*k0 + 5*ang1*k1 + 2*phi0 + 5*phi1))/(2*k0 + 5*k1) + (8820*a0**2*a1**7*np.cos(2*ang1*k0 + 5*ang1*k1 + 2*phi0 + 5*phi1))/(2*k0 + 5*k1) - (39690*a0**4*a1**5*np.cos(4*ang1*k0 + 5*ang1*k1 + 4*phi0 + 5*phi1))/(4*k0 + 5*k1) + (8820*a0**3*a1**6*np.cos(ang1*k0 + 6*ang1*k1 + phi0 + 6*phi1))/(k0 + 6*k1) + (2520*a0*a1**8*np.cos(ang1*k0 + 6*ang1*k1 + phi0 + 6*phi1))/(k0 + 6*k1) - (11340*a0**2*a1**7*np.cos(2*ang1*k0 + 7*ang1*k1 + 2*phi0 + 7*phi1))/(2*k0 + 7*k1) - (2835*a0*a1**8*np.cos(ang1*k0 + 8*ang1*k1 + phi0 + 8*phi1))/(k0 + 8*k1))/80640.)
            return res1 - res0
        else:
            print('invalid index j = ' + str(j) + ' in Iij')
            return 0.0
    elif i == 5:
        if j == 0:
            res0 = (((150*a0*(a0**4 + 6*a0**2*a1**2 + 3*a1**4)*np.sin(ang0*k0 + phi0))/k0 + (25*(a0**5 + 4*a0**3*a1**2)*np.sin(3*(ang0*k0 + phi0)))/k0 + (3*a0**5*np.sin(5*(ang0*k0 + phi0)))/k0 + (75*a0*a1**4*np.sin(ang0*k0 - 4*ang0*k1 + phi0 - 4*phi1))/(k0 - 4*k1) + (150*a0**2*a1**3*np.sin(2*ang0*k0 - 3*ang0*k1 + 2*phi0 - 3*phi1))/(2*k0 - 3*k1) + (450*a0**3*a1**2*np.sin(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1))/(k0 - 2*k1) + (300*a0*a1**4*np.sin(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1))/(k0 - 2*k1) + (150*a0**3*a1**2*np.sin(3*ang0*k0 - 2*ang0*k1 + 3*phi0 - 2*phi1))/(3*k0 - 2*k1) + (300*a0**4*a1*np.sin(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1))/(2*k0 - k1) + (450*a0**2*a1**3*np.sin(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1))/(2*k0 - k1) + (75*a0**4*a1*np.sin(4*ang0*k0 - ang0*k1 + 4*phi0 - phi1))/(4*k0 - k1) + (450*a0**4*a1*np.sin(ang0*k1 + phi1))/k1 + (900*a0**2*a1**3*np.sin(ang0*k1 + phi1))/k1 + (150*a1**5*np.sin(ang0*k1 + phi1))/k1 + (100*a0**2*a1**3*np.sin(3*(ang0*k1 + phi1)))/k1 + (25*a1**5*np.sin(3*(ang0*k1 + phi1)))/k1 + (3*a1**5*np.sin(5*(ang0*k1 + phi1)))/k1 + (300*a0**4*a1*np.sin(2*ang0*k0 + ang0*k1 + 2*phi0 + phi1))/(2*k0 + k1) + (450*a0**2*a1**3*np.sin(2*ang0*k0 + ang0*k1 + 2*phi0 + phi1))/(2*k0 + k1) + (75*a0**4*a1*np.sin(4*ang0*k0 + ang0*k1 + 4*phi0 + phi1))/(4*k0 + k1) + (450*a0**3*a1**2*np.sin(ang0*k0 + 2*ang0*k1 + phi0 + 2*phi1))/(k0 + 2*k1) + (300*a0*a1**4*np.sin(ang0*k0 + 2*ang0*k1 + phi0 + 2*phi1))/(k0 + 2*k1) + (150*a0**3*a1**2*np.sin(3*ang0*k0 + 2*ang0*k1 + 3*phi0 + 2*phi1))/(3*k0 + 2*k1) + (150*a0**2*a1**3*np.sin(2*ang0*k0 + 3*ang0*k1 + 2*phi0 + 3*phi1))/(2*k0 + 3*k1) + (75*a0*a1**4*np.sin(ang0*k0 + 4*ang0*k1 + phi0 + 4*phi1))/(k0 + 4*k1))/240.)
            res1 = (((150*a0*(a0**4 + 6*a0**2*a1**2 + 3*a1**4)*np.sin(ang1*k0 + phi0))/k0 + (25*(a0**5 + 4*a0**3*a1**2)*np.sin(3*(ang1*k0 + phi0)))/k0 + (3*a0**5*np.sin(5*(ang1*k0 + phi0)))/k0 + (75*a0*a1**4*np.sin(ang1*k0 - 4*ang1*k1 + phi0 - 4*phi1))/(k0 - 4*k1) + (150*a0**2*a1**3*np.sin(2*ang1*k0 - 3*ang1*k1 + 2*phi0 - 3*phi1))/(2*k0 - 3*k1) + (450*a0**3*a1**2*np.sin(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1))/(k0 - 2*k1) + (300*a0*a1**4*np.sin(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1))/(k0 - 2*k1) + (150*a0**3*a1**2*np.sin(3*ang1*k0 - 2*ang1*k1 + 3*phi0 - 2*phi1))/(3*k0 - 2*k1) + (300*a0**4*a1*np.sin(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1))/(2*k0 - k1) + (450*a0**2*a1**3*np.sin(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1))/(2*k0 - k1) + (75*a0**4*a1*np.sin(4*ang1*k0 - ang1*k1 + 4*phi0 - phi1))/(4*k0 - k1) + (450*a0**4*a1*np.sin(ang1*k1 + phi1))/k1 + (900*a0**2*a1**3*np.sin(ang1*k1 + phi1))/k1 + (150*a1**5*np.sin(ang1*k1 + phi1))/k1 + (100*a0**2*a1**3*np.sin(3*(ang1*k1 + phi1)))/k1 + (25*a1**5*np.sin(3*(ang1*k1 + phi1)))/k1 + (3*a1**5*np.sin(5*(ang1*k1 + phi1)))/k1 + (300*a0**4*a1*np.sin(2*ang1*k0 + ang1*k1 + 2*phi0 + phi1))/(2*k0 + k1) + (450*a0**2*a1**3*np.sin(2*ang1*k0 + ang1*k1 + 2*phi0 + phi1))/(2*k0 + k1) + (75*a0**4*a1*np.sin(4*ang1*k0 + ang1*k1 + 4*phi0 + phi1))/(4*k0 + k1) + (450*a0**3*a1**2*np.sin(ang1*k0 + 2*ang1*k1 + phi0 + 2*phi1))/(k0 + 2*k1) + (300*a0*a1**4*np.sin(ang1*k0 + 2*ang1*k1 + phi0 + 2*phi1))/(k0 + 2*k1) + (150*a0**3*a1**2*np.sin(3*ang1*k0 + 2*ang1*k1 + 3*phi0 + 2*phi1))/(3*k0 + 2*k1) + (150*a0**2*a1**3*np.sin(2*ang1*k0 + 3*ang1*k1 + 2*phi0 + 3*phi1))/(2*k0 + 3*k1) + (75*a0*a1**4*np.sin(ang1*k0 + 4*ang1*k1 + phi0 + 4*phi1))/(k0 + 4*k1))/240.)
            return res1 - res0
        elif j == 1:
            res0 = (((-15*(a0**6 + 8*a0**4*a1**2 + 6*a0**2*a1**4)*np.cos(2*(ang0*k0 + phi0)))/k0 - (6*(a0**6 + 5*a0**4*a1**2)*np.cos(4*(ang0*k0 + phi0)))/k0 - (a0**6*np.cos(6*(ang0*k0 + phi0)))/k0 + (24*a0*a1**5*np.cos(ang0*k0 - 5*ang0*k1 + phi0 - 5*phi1))/(k0 - 5*k1) + (120*a0**3*a1**3*np.cos(ang0*k0 - 3*ang0*k1 + phi0 - 3*phi1))/(k0 - 3*k1) + (60*a0*a1**5*np.cos(ang0*k0 - 3*ang0*k1 + phi0 - 3*phi1))/(k0 - 3*k1) + (15*a0**2*a1**4*np.cos(2*(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1)))/(k0 - 2*k1) - (15*a0**4*a1**2*np.cos(4*ang0*k0 - 2*ang0*k1 + 4*phi0 - 2*phi1))/(2*k0 - k1) - (60*a0**5*a1*np.cos(3*ang0*k0 - ang0*k1 + 3*phi0 - phi1))/(3*k0 - k1) - (120*a0**3*a1**3*np.cos(3*ang0*k0 - ang0*k1 + 3*phi0 - phi1))/(3*k0 - k1) - (24*a0**5*a1*np.cos(5*ang0*k0 - ang0*k1 + 5*phi0 - phi1))/(5*k0 - k1) - (90*a0**4*a1**2*np.cos(2*(ang0*k1 + phi1)))/k1 - (120*a0**2*a1**4*np.cos(2*(ang0*k1 + phi1)))/k1 - (15*a1**6*np.cos(2*(ang0*k1 + phi1)))/k1 - (30*a0**2*a1**4*np.cos(4*(ang0*k1 + phi1)))/k1 - (6*a1**6*np.cos(4*(ang0*k1 + phi1)))/k1 - (a1**6*np.cos(6*(ang0*k1 + phi1)))/k1 - (120*a0**5*a1*np.cos(ang0*(k0 + k1) + phi0 + phi1))/(k0 + k1) - (360*a0**3*a1**3*np.cos(ang0*(k0 + k1) + phi0 + phi1))/(k0 + k1) - (120*a0*a1**5*np.cos(ang0*(k0 + k1) + phi0 + phi1))/(k0 + k1) - (120*a0**4*a1**2*np.cos(2*(ang0*(k0 + k1) + phi0 + phi1)))/(k0 + k1) - (120*a0**2*a1**4*np.cos(2*(ang0*(k0 + k1) + phi0 + phi1)))/(k0 + k1) - (40*a0**3*a1**3*np.cos(3*(ang0*(k0 + k1) + phi0 + phi1)))/(k0 + k1) - (45*a0**4*a1**2*np.cos(2*(2*ang0*k0 + ang0*k1 + 2*phi0 + phi1)))/(2*k0 + k1) - (120*a0**5*a1*np.cos(3*ang0*k0 + ang0*k1 + 3*phi0 + phi1))/(3*k0 + k1) - (240*a0**3*a1**3*np.cos(3*ang0*k0 + ang0*k1 + 3*phi0 + phi1))/(3*k0 + k1) - (36*a0**5*a1*np.cos(5*ang0*k0 + ang0*k1 + 5*phi0 + phi1))/(5*k0 + k1) - (45*a0**2*a1**4*np.cos(2*(ang0*k0 + 2*ang0*k1 + phi0 + 2*phi1)))/(k0 + 2*k1) - (240*a0**3*a1**3*np.cos(ang0*k0 + 3*ang0*k1 + phi0 + 3*phi1))/(k0 + 3*k1) - (120*a0*a1**5*np.cos(ang0*k0 + 3*ang0*k1 + phi0 + 3*phi1))/(k0 + 3*k1) - (36*a0*a1**5*np.cos(ang0*k0 + 5*ang0*k1 + phi0 + 5*phi1))/(k0 + 5*k1))/192.)
            res1 = (((-15*(a0**6 + 8*a0**4*a1**2 + 6*a0**2*a1**4)*np.cos(2*(ang1*k0 + phi0)))/k0 - (6*(a0**6 + 5*a0**4*a1**2)*np.cos(4*(ang1*k0 + phi0)))/k0 - (a0**6*np.cos(6*(ang1*k0 + phi0)))/k0 + (24*a0*a1**5*np.cos(ang1*k0 - 5*ang1*k1 + phi0 - 5*phi1))/(k0 - 5*k1) + (120*a0**3*a1**3*np.cos(ang1*k0 - 3*ang1*k1 + phi0 - 3*phi1))/(k0 - 3*k1) + (60*a0*a1**5*np.cos(ang1*k0 - 3*ang1*k1 + phi0 - 3*phi1))/(k0 - 3*k1) + (15*a0**2*a1**4*np.cos(2*(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1)))/(k0 - 2*k1) - (15*a0**4*a1**2*np.cos(4*ang1*k0 - 2*ang1*k1 + 4*phi0 - 2*phi1))/(2*k0 - k1) - (60*a0**5*a1*np.cos(3*ang1*k0 - ang1*k1 + 3*phi0 - phi1))/(3*k0 - k1) - (120*a0**3*a1**3*np.cos(3*ang1*k0 - ang1*k1 + 3*phi0 - phi1))/(3*k0 - k1) - (24*a0**5*a1*np.cos(5*ang1*k0 - ang1*k1 + 5*phi0 - phi1))/(5*k0 - k1) - (90*a0**4*a1**2*np.cos(2*(ang1*k1 + phi1)))/k1 - (120*a0**2*a1**4*np.cos(2*(ang1*k1 + phi1)))/k1 - (15*a1**6*np.cos(2*(ang1*k1 + phi1)))/k1 - (30*a0**2*a1**4*np.cos(4*(ang1*k1 + phi1)))/k1 - (6*a1**6*np.cos(4*(ang1*k1 + phi1)))/k1 - (a1**6*np.cos(6*(ang1*k1 + phi1)))/k1 - (120*a0**5*a1*np.cos(ang1*(k0 + k1) + phi0 + phi1))/(k0 + k1) - (360*a0**3*a1**3*np.cos(ang1*(k0 + k1) + phi0 + phi1))/(k0 + k1) - (120*a0*a1**5*np.cos(ang1*(k0 + k1) + phi0 + phi1))/(k0 + k1) - (120*a0**4*a1**2*np.cos(2*(ang1*(k0 + k1) + phi0 + phi1)))/(k0 + k1) - (120*a0**2*a1**4*np.cos(2*(ang1*(k0 + k1) + phi0 + phi1)))/(k0 + k1) - (40*a0**3*a1**3*np.cos(3*(ang1*(k0 + k1) + phi0 + phi1)))/(k0 + k1) - (45*a0**4*a1**2*np.cos(2*(2*ang1*k0 + ang1*k1 + 2*phi0 + phi1)))/(2*k0 + k1) - (120*a0**5*a1*np.cos(3*ang1*k0 + ang1*k1 + 3*phi0 + phi1))/(3*k0 + k1) - (240*a0**3*a1**3*np.cos(3*ang1*k0 + ang1*k1 + 3*phi0 + phi1))/(3*k0 + k1) - (36*a0**5*a1*np.cos(5*ang1*k0 + ang1*k1 + 5*phi0 + phi1))/(5*k0 + k1) - (45*a0**2*a1**4*np.cos(2*(ang1*k0 + 2*ang1*k1 + phi0 + 2*phi1)))/(k0 + 2*k1) - (240*a0**3*a1**3*np.cos(ang1*k0 + 3*ang1*k1 + phi0 + 3*phi1))/(k0 + 3*k1) - (120*a0*a1**5*np.cos(ang1*k0 + 3*ang1*k1 + phi0 + 3*phi1))/(k0 + 3*k1) - (36*a0*a1**5*np.cos(ang1*k0 + 5*ang1*k1 + phi0 + 5*phi1))/(k0 + 5*k1))/192.)
            return res1 - res0
        elif j == 2:
            res0 = (((525*a0*(a0**6 + 12*a0**4*a1**2 + 18*a0**2*a1**4 + 4*a1**6)*np.sin(ang0*k0 + phi0))/k0 - (35*(a0**7 + 10*a0**5*a1**2 + 10*a0**3*a1**4)*np.sin(3*(ang0*k0 + phi0)))/k0 - (63*a0**7*np.sin(5*(ang0*k0 + phi0)))/k0 - (378*a0**5*a1**2*np.sin(5*(ang0*k0 + phi0)))/k0 - (15*a0**7*np.sin(7*(ang0*k0 + phi0)))/k0 - (315*a0*a1**6*np.sin(ang0*k0 - 6*ang0*k1 + phi0 - 6*phi1))/(k0 - 6*k1) - (105*a0**2*a1**5*np.sin(2*ang0*k0 - 5*ang0*k1 + 2*phi0 - 5*phi1))/(2*k0 - 5*k1) - (525*a0**3*a1**4*np.sin(ang0*k0 - 4*ang0*k1 + phi0 - 4*phi1))/(k0 - 4*k1) - (210*a0*a1**6*np.sin(ang0*k0 - 4*ang0*k1 + phi0 - 4*phi1))/(k0 - 4*k1) + (525*a0**3*a1**4*np.sin(3*ang0*k0 - 4*ang0*k1 + 3*phi0 - 4*phi1))/(3*k0 - 4*k1) + (2100*a0**4*a1**3*np.sin(2*ang0*k0 - 3*ang0*k1 + 2*phi0 - 3*phi1))/(2*k0 - 3*k1) + (1575*a0**2*a1**5*np.sin(2*ang0*k0 - 3*ang0*k1 + 2*phi0 - 3*phi1))/(2*k0 - 3*k1) + (525*a0**4*a1**3*np.sin(4*ang0*k0 - 3*ang0*k1 + 4*phi0 - 3*phi1))/(4*k0 - 3*k1) + (3150*a0**5*a1**2*np.sin(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1))/(k0 - 2*k1) + (6300*a0**3*a1**4*np.sin(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1))/(k0 - 2*k1) + (1575*a0*a1**6*np.sin(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1))/(k0 - 2*k1) + (1575*a0**5*a1**2*np.sin(3*ang0*k0 - 2*ang0*k1 + 3*phi0 - 2*phi1))/(3*k0 - 2*k1) + (2100*a0**3*a1**4*np.sin(3*ang0*k0 - 2*ang0*k1 + 3*phi0 - 2*phi1))/(3*k0 - 2*k1) - (105*a0**5*a1**2*np.sin(5*ang0*k0 - 2*ang0*k1 + 5*phi0 - 2*phi1))/(5*k0 - 2*k1) + (1575*a0**6*a1*np.sin(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1))/(2*k0 - k1) + (6300*a0**4*a1**3*np.sin(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1))/(2*k0 - k1) + (3150*a0**2*a1**5*np.sin(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1))/(2*k0 - k1) - (210*a0**6*a1*np.sin(4*ang0*k0 - ang0*k1 + 4*phi0 - phi1))/(4*k0 - k1) - (525*a0**4*a1**3*np.sin(4*ang0*k0 - ang0*k1 + 4*phi0 - phi1))/(4*k0 - k1) - (315*a0**6*a1*np.sin(6*ang0*k0 - ang0*k1 + 6*phi0 - phi1))/(6*k0 - k1) + (2100*a0**6*a1*np.sin(ang0*k1 + phi1))/k1 + (9450*a0**4*a1**3*np.sin(ang0*k1 + phi1))/k1 + (6300*a0**2*a1**5*np.sin(ang0*k1 + phi1))/k1 + (525*a1**7*np.sin(ang0*k1 + phi1))/k1 - (350*a0**4*a1**3*np.sin(3*(ang0*k1 + phi1)))/k1 - (350*a0**2*a1**5*np.sin(3*(ang0*k1 + phi1)))/k1 - (35*a1**7*np.sin(3*(ang0*k1 + phi1)))/k1 - (378*a0**2*a1**5*np.sin(5*(ang0*k1 + phi1)))/k1 - (63*a1**7*np.sin(5*(ang0*k1 + phi1)))/k1 - (15*a1**7*np.sin(7*(ang0*k1 + phi1)))/k1 - (525*a0**6*a1*np.sin(2*ang0*k0 + ang0*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (2100*a0**4*a1**3*np.sin(2*ang0*k0 + ang0*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (1050*a0**2*a1**5*np.sin(2*ang0*k0 + ang0*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (1890*a0**6*a1*np.sin(4*ang0*k0 + ang0*k1 + 4*phi0 + phi1))/(4*k0 + k1) - (4725*a0**4*a1**3*np.sin(4*ang0*k0 + ang0*k1 + 4*phi0 + phi1))/(4*k0 + k1) - (735*a0**6*a1*np.sin(6*ang0*k0 + ang0*k1 + 6*phi0 + phi1))/(6*k0 + k1) - (1050*a0**5*a1**2*np.sin(ang0*k0 + 2*ang0*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (2100*a0**3*a1**4*np.sin(ang0*k0 + 2*ang0*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (525*a0*a1**6*np.sin(ang0*k0 + 2*ang0*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (4725*a0**5*a1**2*np.sin(3*ang0*k0 + 2*ang0*k1 + 3*phi0 + 2*phi1))/(3*k0 + 2*k1) - (6300*a0**3*a1**4*np.sin(3*ang0*k0 + 2*ang0*k1 + 3*phi0 + 2*phi1))/(3*k0 + 2*k1) - (2205*a0**5*a1**2*np.sin(5*ang0*k0 + 2*ang0*k1 + 5*phi0 + 2*phi1))/(5*k0 + 2*k1) - (6300*a0**4*a1**3*np.sin(2*ang0*k0 + 3*ang0*k1 + 2*phi0 + 3*phi1))/(2*k0 + 3*k1) - (4725*a0**2*a1**5*np.sin(2*ang0*k0 + 3*ang0*k1 + 2*phi0 + 3*phi1))/(2*k0 + 3*k1) - (3675*a0**4*a1**3*np.sin(4*ang0*k0 + 3*ang0*k1 + 4*phi0 + 3*phi1))/(4*k0 + 3*k1) - (4725*a0**3*a1**4*np.sin(ang0*k0 + 4*ang0*k1 + phi0 + 4*phi1))/(k0 + 4*k1) - (1890*a0*a1**6*np.sin(ang0*k0 + 4*ang0*k1 + phi0 + 4*phi1))/(k0 + 4*k1) - (3675*a0**3*a1**4*np.sin(3*ang0*k0 + 4*ang0*k1 + 3*phi0 + 4*phi1))/(3*k0 + 4*k1) - (2205*a0**2*a1**5*np.sin(2*ang0*k0 + 5*ang0*k1 + 2*phi0 + 5*phi1))/(2*k0 + 5*k1) - (735*a0*a1**6*np.sin(ang0*k0 + 6*ang0*k1 + phi0 + 6*phi1))/(k0 + 6*k1))/6720.)
            res1 = (((525*a0*(a0**6 + 12*a0**4*a1**2 + 18*a0**2*a1**4 + 4*a1**6)*np.sin(ang1*k0 + phi0))/k0 - (35*(a0**7 + 10*a0**5*a1**2 + 10*a0**3*a1**4)*np.sin(3*(ang1*k0 + phi0)))/k0 - (63*a0**7*np.sin(5*(ang1*k0 + phi0)))/k0 - (378*a0**5*a1**2*np.sin(5*(ang1*k0 + phi0)))/k0 - (15*a0**7*np.sin(7*(ang1*k0 + phi0)))/k0 - (315*a0*a1**6*np.sin(ang1*k0 - 6*ang1*k1 + phi0 - 6*phi1))/(k0 - 6*k1) - (105*a0**2*a1**5*np.sin(2*ang1*k0 - 5*ang1*k1 + 2*phi0 - 5*phi1))/(2*k0 - 5*k1) - (525*a0**3*a1**4*np.sin(ang1*k0 - 4*ang1*k1 + phi0 - 4*phi1))/(k0 - 4*k1) - (210*a0*a1**6*np.sin(ang1*k0 - 4*ang1*k1 + phi0 - 4*phi1))/(k0 - 4*k1) + (525*a0**3*a1**4*np.sin(3*ang1*k0 - 4*ang1*k1 + 3*phi0 - 4*phi1))/(3*k0 - 4*k1) + (2100*a0**4*a1**3*np.sin(2*ang1*k0 - 3*ang1*k1 + 2*phi0 - 3*phi1))/(2*k0 - 3*k1) + (1575*a0**2*a1**5*np.sin(2*ang1*k0 - 3*ang1*k1 + 2*phi0 - 3*phi1))/(2*k0 - 3*k1) + (525*a0**4*a1**3*np.sin(4*ang1*k0 - 3*ang1*k1 + 4*phi0 - 3*phi1))/(4*k0 - 3*k1) + (3150*a0**5*a1**2*np.sin(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1))/(k0 - 2*k1) + (6300*a0**3*a1**4*np.sin(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1))/(k0 - 2*k1) + (1575*a0*a1**6*np.sin(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1))/(k0 - 2*k1) + (1575*a0**5*a1**2*np.sin(3*ang1*k0 - 2*ang1*k1 + 3*phi0 - 2*phi1))/(3*k0 - 2*k1) + (2100*a0**3*a1**4*np.sin(3*ang1*k0 - 2*ang1*k1 + 3*phi0 - 2*phi1))/(3*k0 - 2*k1) - (105*a0**5*a1**2*np.sin(5*ang1*k0 - 2*ang1*k1 + 5*phi0 - 2*phi1))/(5*k0 - 2*k1) + (1575*a0**6*a1*np.sin(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1))/(2*k0 - k1) + (6300*a0**4*a1**3*np.sin(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1))/(2*k0 - k1) + (3150*a0**2*a1**5*np.sin(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1))/(2*k0 - k1) - (210*a0**6*a1*np.sin(4*ang1*k0 - ang1*k1 + 4*phi0 - phi1))/(4*k0 - k1) - (525*a0**4*a1**3*np.sin(4*ang1*k0 - ang1*k1 + 4*phi0 - phi1))/(4*k0 - k1) - (315*a0**6*a1*np.sin(6*ang1*k0 - ang1*k1 + 6*phi0 - phi1))/(6*k0 - k1) + (2100*a0**6*a1*np.sin(ang1*k1 + phi1))/k1 + (9450*a0**4*a1**3*np.sin(ang1*k1 + phi1))/k1 + (6300*a0**2*a1**5*np.sin(ang1*k1 + phi1))/k1 + (525*a1**7*np.sin(ang1*k1 + phi1))/k1 - (350*a0**4*a1**3*np.sin(3*(ang1*k1 + phi1)))/k1 - (350*a0**2*a1**5*np.sin(3*(ang1*k1 + phi1)))/k1 - (35*a1**7*np.sin(3*(ang1*k1 + phi1)))/k1 - (378*a0**2*a1**5*np.sin(5*(ang1*k1 + phi1)))/k1 - (63*a1**7*np.sin(5*(ang1*k1 + phi1)))/k1 - (15*a1**7*np.sin(7*(ang1*k1 + phi1)))/k1 - (525*a0**6*a1*np.sin(2*ang1*k0 + ang1*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (2100*a0**4*a1**3*np.sin(2*ang1*k0 + ang1*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (1050*a0**2*a1**5*np.sin(2*ang1*k0 + ang1*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (1890*a0**6*a1*np.sin(4*ang1*k0 + ang1*k1 + 4*phi0 + phi1))/(4*k0 + k1) - (4725*a0**4*a1**3*np.sin(4*ang1*k0 + ang1*k1 + 4*phi0 + phi1))/(4*k0 + k1) - (735*a0**6*a1*np.sin(6*ang1*k0 + ang1*k1 + 6*phi0 + phi1))/(6*k0 + k1) - (1050*a0**5*a1**2*np.sin(ang1*k0 + 2*ang1*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (2100*a0**3*a1**4*np.sin(ang1*k0 + 2*ang1*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (525*a0*a1**6*np.sin(ang1*k0 + 2*ang1*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (4725*a0**5*a1**2*np.sin(3*ang1*k0 + 2*ang1*k1 + 3*phi0 + 2*phi1))/(3*k0 + 2*k1) - (6300*a0**3*a1**4*np.sin(3*ang1*k0 + 2*ang1*k1 + 3*phi0 + 2*phi1))/(3*k0 + 2*k1) - (2205*a0**5*a1**2*np.sin(5*ang1*k0 + 2*ang1*k1 + 5*phi0 + 2*phi1))/(5*k0 + 2*k1) - (6300*a0**4*a1**3*np.sin(2*ang1*k0 + 3*ang1*k1 + 2*phi0 + 3*phi1))/(2*k0 + 3*k1) - (4725*a0**2*a1**5*np.sin(2*ang1*k0 + 3*ang1*k1 + 2*phi0 + 3*phi1))/(2*k0 + 3*k1) - (3675*a0**4*a1**3*np.sin(4*ang1*k0 + 3*ang1*k1 + 4*phi0 + 3*phi1))/(4*k0 + 3*k1) - (4725*a0**3*a1**4*np.sin(ang1*k0 + 4*ang1*k1 + phi0 + 4*phi1))/(k0 + 4*k1) - (1890*a0*a1**6*np.sin(ang1*k0 + 4*ang1*k1 + phi0 + 4*phi1))/(k0 + 4*k1) - (3675*a0**3*a1**4*np.sin(3*ang1*k0 + 4*ang1*k1 + 3*phi0 + 4*phi1))/(3*k0 + 4*k1) - (2205*a0**2*a1**5*np.sin(2*ang1*k0 + 5*ang1*k1 + 2*phi0 + 5*phi1))/(2*k0 + 5*k1) - (735*a0*a1**6*np.sin(ang1*k0 + 6*ang1*k1 + phi0 + 6*phi1))/(k0 + 6*k1))/6720.)
            return res1 - res0
        elif j == 3:
            res0 = (((-72*(a0**8 + 15*a0**6*a1**2 + 30*a0**4*a1**4 + 10*a0**2*a1**6)*np.cos(2*(ang0*k0 + phi0)))/k0 - (12*(a0**8 + 12*a0**6*a1**2 + 15*a0**4*a1**4)*np.cos(4*(ang0*k0 + phi0)))/k0 + (8*a0**8*np.cos(6*(ang0*k0 + phi0)))/k0 + (56*a0**6*a1**2*np.cos(6*(ang0*k0 + phi0)))/k0 + (3*a0**8*np.cos(8*(ang0*k0 + phi0)))/k0 - (48*a0*a1**7*np.cos(ang0*k0 - 7*ang0*k1 + phi0 - 7*phi1))/(k0 - 7*k1) + (288*a0**3*a1**5*np.cos(ang0*k0 - 5*ang0*k1 + phi0 - 5*phi1))/(k0 - 5*k1) + (96*a0*a1**7*np.cos(ang0*k0 - 5*ang0*k1 + phi0 - 5*phi1))/(k0 - 5*k1) + (144*a0**3*a1**5*np.cos(3*ang0*k0 - 5*ang0*k1 + 3*phi0 - 5*phi1))/(3*k0 - 5*k1) + (1440*a0**5*a1**3*np.cos(ang0*k0 - 3*ang0*k1 + phi0 - 3*phi1))/(k0 - 3*k1) + (2160*a0**3*a1**5*np.cos(ang0*k0 - 3*ang0*k1 + phi0 - 3*phi1))/(k0 - 3*k1) + (432*a0*a1**7*np.cos(ang0*k0 - 3*ang0*k1 + phi0 - 3*phi1))/(k0 - 3*k1) + (24*a0**2*a1**6*np.cos(2*(ang0*k0 - 3*ang0*k1 + phi0 - 3*phi1)))/(k0 - 3*k1) - (144*a0**5*a1**3*np.cos(5*ang0*k0 - 3*ang0*k1 + 5*phi0 - 3*phi1))/(5*k0 - 3*k1) + (360*a0**4*a1**4*np.cos(2*(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1)))/(k0 - 2*k1) + (216*a0**2*a1**6*np.cos(2*(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1)))/(k0 - 2*k1) - (216*a0**6*a1**2*np.cos(4*ang0*k0 - 2*ang0*k1 + 4*phi0 - 2*phi1))/(2*k0 - k1) - (360*a0**4*a1**4*np.cos(4*ang0*k0 - 2*ang0*k1 + 4*phi0 - 2*phi1))/(2*k0 - k1) - (24*a0**6*a1**2*np.cos(6*ang0*k0 - 2*ang0*k1 + 6*phi0 - 2*phi1))/(3*k0 - k1) - (432*a0**7*a1*np.cos(3*ang0*k0 - ang0*k1 + 3*phi0 - phi1))/(3*k0 - k1) - (2160*a0**5*a1**3*np.cos(3*ang0*k0 - ang0*k1 + 3*phi0 - phi1))/(3*k0 - k1) - (1440*a0**3*a1**5*np.cos(3*ang0*k0 - ang0*k1 + 3*phi0 - phi1))/(3*k0 - k1) - (96*a0**7*a1*np.cos(5*ang0*k0 - ang0*k1 + 5*phi0 - phi1))/(5*k0 - k1) - (288*a0**5*a1**3*np.cos(5*ang0*k0 - ang0*k1 + 5*phi0 - phi1))/(5*k0 - k1) + (48*a0**7*a1*np.cos(7*ang0*k0 - ang0*k1 + 7*phi0 - phi1))/(7*k0 - k1) - (720*a0**6*a1**2*np.cos(2*(ang0*k1 + phi1)))/k1 - (2160*a0**4*a1**4*np.cos(2*(ang0*k1 + phi1)))/k1 - (1080*a0**2*a1**6*np.cos(2*(ang0*k1 + phi1)))/k1 - (72*a1**8*np.cos(2*(ang0*k1 + phi1)))/k1 - (180*a0**4*a1**4*np.cos(4*(ang0*k1 + phi1)))/k1 - (144*a0**2*a1**6*np.cos(4*(ang0*k1 + phi1)))/k1 - (12*a1**8*np.cos(4*(ang0*k1 + phi1)))/k1 + (56*a0**2*a1**6*np.cos(6*(ang0*k1 + phi1)))/k1 + (8*a1**8*np.cos(6*(ang0*k1 + phi1)))/k1 + (3*a1**8*np.cos(8*(ang0*k1 + phi1)))/k1 - (720*a0**7*a1*np.cos(ang0*(k0 + k1) + phi0 + phi1))/(k0 + k1) - (4320*a0**5*a1**3*np.cos(ang0*(k0 + k1) + phi0 + phi1))/(k0 + k1) - (4320*a0**3*a1**5*np.cos(ang0*(k0 + k1) + phi0 + phi1))/(k0 + k1) - (720*a0*a1**7*np.cos(ang0*(k0 + k1) + phi0 + phi1))/(k0 + k1) - (360*a0**6*a1**2*np.cos(2*(ang0*(k0 + k1) + phi0 + phi1)))/(k0 + k1) - (960*a0**4*a1**4*np.cos(2*(ang0*(k0 + k1) + phi0 + phi1)))/(k0 + k1) - (360*a0**2*a1**6*np.cos(2*(ang0*(k0 + k1) + phi0 + phi1)))/(k0 + k1) + (560*a0**5*a1**3*np.cos(3*(ang0*(k0 + k1) + phi0 + phi1)))/(k0 + k1) + (560*a0**3*a1**5*np.cos(3*(ang0*(k0 + k1) + phi0 + phi1)))/(k0 + k1) + (420*a0**4*a1**4*np.cos(4*(ang0*(k0 + k1) + phi0 + phi1)))/(k0 + k1) + (504*a0**6*a1**2*np.cos(2*(2*ang0*k0 + ang0*k1 + 2*phi0 + phi1)))/(2*k0 + k1) + (840*a0**4*a1**4*np.cos(2*(2*ang0*k0 + ang0*k1 + 2*phi0 + phi1)))/(2*k0 + k1) - (288*a0**7*a1*np.cos(3*ang0*k0 + ang0*k1 + 3*phi0 + phi1))/(3*k0 + k1) - (1440*a0**5*a1**3*np.cos(3*ang0*k0 + ang0*k1 + 3*phi0 + phi1))/(3*k0 + k1) - (960*a0**3*a1**5*np.cos(3*ang0*k0 + ang0*k1 + 3*phi0 + phi1))/(3*k0 + k1) + (336*a0**6*a1**2*np.cos(2*(3*ang0*k0 + ang0*k1 + 3*phi0 + phi1)))/(3*k0 + k1) + (336*a0**7*a1*np.cos(5*ang0*k0 + ang0*k1 + 5*phi0 + phi1))/(5*k0 + k1) + (1008*a0**5*a1**3*np.cos(5*ang0*k0 + ang0*k1 + 5*phi0 + phi1))/(5*k0 + k1) + (192*a0**7*a1*np.cos(7*ang0*k0 + ang0*k1 + 7*phi0 + phi1))/(7*k0 + k1) + (840*a0**4*a1**4*np.cos(2*(ang0*k0 + 2*ang0*k1 + phi0 + 2*phi1)))/(k0 + 2*k1) + (504*a0**2*a1**6*np.cos(2*(ang0*k0 + 2*ang0*k1 + phi0 + 2*phi1)))/(k0 + 2*k1) - (960*a0**5*a1**3*np.cos(ang0*k0 + 3*ang0*k1 + phi0 + 3*phi1))/(k0 + 3*k1) - (1440*a0**3*a1**5*np.cos(ang0*k0 + 3*ang0*k1 + phi0 + 3*phi1))/(k0 + 3*k1) - (288*a0*a1**7*np.cos(ang0*k0 + 3*ang0*k1 + phi0 + 3*phi1))/(k0 + 3*k1) + (336*a0**2*a1**6*np.cos(2*(ang0*k0 + 3*ang0*k1 + phi0 + 3*phi1)))/(k0 + 3*k1) + (1344*a0**5*a1**3*np.cos(5*ang0*k0 + 3*ang0*k1 + 5*phi0 + 3*phi1))/(5*k0 + 3*k1) + (1008*a0**3*a1**5*np.cos(ang0*k0 + 5*ang0*k1 + phi0 + 5*phi1))/(k0 + 5*k1) + (336*a0*a1**7*np.cos(ang0*k0 + 5*ang0*k1 + phi0 + 5*phi1))/(k0 + 5*k1) + (1344*a0**3*a1**5*np.cos(3*ang0*k0 + 5*ang0*k1 + 3*phi0 + 5*phi1))/(3*k0 + 5*k1) + (192*a0*a1**7*np.cos(ang0*k0 + 7*ang0*k1 + phi0 + 7*phi1))/(k0 + 7*k1))/3072.)
            res1 = (((-72*(a0**8 + 15*a0**6*a1**2 + 30*a0**4*a1**4 + 10*a0**2*a1**6)*np.cos(2*(ang1*k0 + phi0)))/k0 - (12*(a0**8 + 12*a0**6*a1**2 + 15*a0**4*a1**4)*np.cos(4*(ang1*k0 + phi0)))/k0 + (8*a0**8*np.cos(6*(ang1*k0 + phi0)))/k0 + (56*a0**6*a1**2*np.cos(6*(ang1*k0 + phi0)))/k0 + (3*a0**8*np.cos(8*(ang1*k0 + phi0)))/k0 - (48*a0*a1**7*np.cos(ang1*k0 - 7*ang1*k1 + phi0 - 7*phi1))/(k0 - 7*k1) + (288*a0**3*a1**5*np.cos(ang1*k0 - 5*ang1*k1 + phi0 - 5*phi1))/(k0 - 5*k1) + (96*a0*a1**7*np.cos(ang1*k0 - 5*ang1*k1 + phi0 - 5*phi1))/(k0 - 5*k1) + (144*a0**3*a1**5*np.cos(3*ang1*k0 - 5*ang1*k1 + 3*phi0 - 5*phi1))/(3*k0 - 5*k1) + (1440*a0**5*a1**3*np.cos(ang1*k0 - 3*ang1*k1 + phi0 - 3*phi1))/(k0 - 3*k1) + (2160*a0**3*a1**5*np.cos(ang1*k0 - 3*ang1*k1 + phi0 - 3*phi1))/(k0 - 3*k1) + (432*a0*a1**7*np.cos(ang1*k0 - 3*ang1*k1 + phi0 - 3*phi1))/(k0 - 3*k1) + (24*a0**2*a1**6*np.cos(2*(ang1*k0 - 3*ang1*k1 + phi0 - 3*phi1)))/(k0 - 3*k1) - (144*a0**5*a1**3*np.cos(5*ang1*k0 - 3*ang1*k1 + 5*phi0 - 3*phi1))/(5*k0 - 3*k1) + (360*a0**4*a1**4*np.cos(2*(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1)))/(k0 - 2*k1) + (216*a0**2*a1**6*np.cos(2*(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1)))/(k0 - 2*k1) - (216*a0**6*a1**2*np.cos(4*ang1*k0 - 2*ang1*k1 + 4*phi0 - 2*phi1))/(2*k0 - k1) - (360*a0**4*a1**4*np.cos(4*ang1*k0 - 2*ang1*k1 + 4*phi0 - 2*phi1))/(2*k0 - k1) - (24*a0**6*a1**2*np.cos(6*ang1*k0 - 2*ang1*k1 + 6*phi0 - 2*phi1))/(3*k0 - k1) - (432*a0**7*a1*np.cos(3*ang1*k0 - ang1*k1 + 3*phi0 - phi1))/(3*k0 - k1) - (2160*a0**5*a1**3*np.cos(3*ang1*k0 - ang1*k1 + 3*phi0 - phi1))/(3*k0 - k1) - (1440*a0**3*a1**5*np.cos(3*ang1*k0 - ang1*k1 + 3*phi0 - phi1))/(3*k0 - k1) - (96*a0**7*a1*np.cos(5*ang1*k0 - ang1*k1 + 5*phi0 - phi1))/(5*k0 - k1) - (288*a0**5*a1**3*np.cos(5*ang1*k0 - ang1*k1 + 5*phi0 - phi1))/(5*k0 - k1) + (48*a0**7*a1*np.cos(7*ang1*k0 - ang1*k1 + 7*phi0 - phi1))/(7*k0 - k1) - (720*a0**6*a1**2*np.cos(2*(ang1*k1 + phi1)))/k1 - (2160*a0**4*a1**4*np.cos(2*(ang1*k1 + phi1)))/k1 - (1080*a0**2*a1**6*np.cos(2*(ang1*k1 + phi1)))/k1 - (72*a1**8*np.cos(2*(ang1*k1 + phi1)))/k1 - (180*a0**4*a1**4*np.cos(4*(ang1*k1 + phi1)))/k1 - (144*a0**2*a1**6*np.cos(4*(ang1*k1 + phi1)))/k1 - (12*a1**8*np.cos(4*(ang1*k1 + phi1)))/k1 + (56*a0**2*a1**6*np.cos(6*(ang1*k1 + phi1)))/k1 + (8*a1**8*np.cos(6*(ang1*k1 + phi1)))/k1 + (3*a1**8*np.cos(8*(ang1*k1 + phi1)))/k1 - (720*a0**7*a1*np.cos(ang1*(k0 + k1) + phi0 + phi1))/(k0 + k1) - (4320*a0**5*a1**3*np.cos(ang1*(k0 + k1) + phi0 + phi1))/(k0 + k1) - (4320*a0**3*a1**5*np.cos(ang1*(k0 + k1) + phi0 + phi1))/(k0 + k1) - (720*a0*a1**7*np.cos(ang1*(k0 + k1) + phi0 + phi1))/(k0 + k1) - (360*a0**6*a1**2*np.cos(2*(ang1*(k0 + k1) + phi0 + phi1)))/(k0 + k1) - (960*a0**4*a1**4*np.cos(2*(ang1*(k0 + k1) + phi0 + phi1)))/(k0 + k1) - (360*a0**2*a1**6*np.cos(2*(ang1*(k0 + k1) + phi0 + phi1)))/(k0 + k1) + (560*a0**5*a1**3*np.cos(3*(ang1*(k0 + k1) + phi0 + phi1)))/(k0 + k1) + (560*a0**3*a1**5*np.cos(3*(ang1*(k0 + k1) + phi0 + phi1)))/(k0 + k1) + (420*a0**4*a1**4*np.cos(4*(ang1*(k0 + k1) + phi0 + phi1)))/(k0 + k1) + (504*a0**6*a1**2*np.cos(2*(2*ang1*k0 + ang1*k1 + 2*phi0 + phi1)))/(2*k0 + k1) + (840*a0**4*a1**4*np.cos(2*(2*ang1*k0 + ang1*k1 + 2*phi0 + phi1)))/(2*k0 + k1) - (288*a0**7*a1*np.cos(3*ang1*k0 + ang1*k1 + 3*phi0 + phi1))/(3*k0 + k1) - (1440*a0**5*a1**3*np.cos(3*ang1*k0 + ang1*k1 + 3*phi0 + phi1))/(3*k0 + k1) - (960*a0**3*a1**5*np.cos(3*ang1*k0 + ang1*k1 + 3*phi0 + phi1))/(3*k0 + k1) + (336*a0**6*a1**2*np.cos(2*(3*ang1*k0 + ang1*k1 + 3*phi0 + phi1)))/(3*k0 + k1) + (336*a0**7*a1*np.cos(5*ang1*k0 + ang1*k1 + 5*phi0 + phi1))/(5*k0 + k1) + (1008*a0**5*a1**3*np.cos(5*ang1*k0 + ang1*k1 + 5*phi0 + phi1))/(5*k0 + k1) + (192*a0**7*a1*np.cos(7*ang1*k0 + ang1*k1 + 7*phi0 + phi1))/(7*k0 + k1) + (840*a0**4*a1**4*np.cos(2*(ang1*k0 + 2*ang1*k1 + phi0 + 2*phi1)))/(k0 + 2*k1) + (504*a0**2*a1**6*np.cos(2*(ang1*k0 + 2*ang1*k1 + phi0 + 2*phi1)))/(k0 + 2*k1) - (960*a0**5*a1**3*np.cos(ang1*k0 + 3*ang1*k1 + phi0 + 3*phi1))/(k0 + 3*k1) - (1440*a0**3*a1**5*np.cos(ang1*k0 + 3*ang1*k1 + phi0 + 3*phi1))/(k0 + 3*k1) - (288*a0*a1**7*np.cos(ang1*k0 + 3*ang1*k1 + phi0 + 3*phi1))/(k0 + 3*k1) + (336*a0**2*a1**6*np.cos(2*(ang1*k0 + 3*ang1*k1 + phi0 + 3*phi1)))/(k0 + 3*k1) + (1344*a0**5*a1**3*np.cos(5*ang1*k0 + 3*ang1*k1 + 5*phi0 + 3*phi1))/(5*k0 + 3*k1) + (1008*a0**3*a1**5*np.cos(ang1*k0 + 5*ang1*k1 + phi0 + 5*phi1))/(k0 + 5*k1) + (336*a0*a1**7*np.cos(ang1*k0 + 5*ang1*k1 + phi0 + 5*phi1))/(k0 + 5*k1) + (1344*a0**3*a1**5*np.cos(3*ang1*k0 + 5*ang1*k1 + 3*phi0 + 5*phi1))/(3*k0 + 5*k1) + (192*a0*a1**7*np.cos(ang1*k0 + 7*ang1*k1 + phi0 + 7*phi1))/(k0 + 7*k1))/3072.)
            return res1 - res0
        elif j == 4:
            res0 = (((1890*a0*(a0**8 + 20*a0**6*a1**2 + 60*a0**4*a1**4 + 40*a0**2*a1**6 + 5*a1**8)*np.sin(ang0*k0 + phi0))/k0 - (420*(a0**9 + 18*a0**7*a1**2 + 45*a0**5*a1**4 + 20*a0**3*a1**6)*np.sin(3*(ang0*k0 + phi0)))/k0 - (252*a0**9*np.sin(5*(ang0*k0 + phi0)))/k0 - (3528*a0**7*a1**2*np.sin(5*(ang0*k0 + phi0)))/k0 - (5292*a0**5*a1**4*np.sin(5*(ang0*k0 + phi0)))/k0 + (45*a0**9*np.sin(7*(ang0*k0 + phi0)))/k0 + (360*a0**7*a1**2*np.sin(7*(ang0*k0 + phi0)))/k0 + (35*a0**9*np.sin(9*(ang0*k0 + phi0)))/k0 + (315*a0*a1**8*np.sin(ang0*k0 - 8*ang0*k1 + phi0 - 8*phi1))/(k0 - 8*k1) - (1260*a0**2*a1**7*np.sin(2*ang0*k0 - 7*ang0*k1 + 2*phi0 - 7*phi1))/(2*k0 - 7*k1) - (8820*a0**3*a1**6*np.sin(ang0*k0 - 6*ang0*k1 + phi0 - 6*phi1))/(k0 - 6*k1) - (2520*a0*a1**8*np.sin(ang0*k0 - 6*ang0*k1 + phi0 - 6*phi1))/(k0 - 6*k1) - (7560*a0**4*a1**5*np.sin(2*ang0*k0 - 5*ang0*k1 + 2*phi0 - 5*phi1))/(2*k0 - 5*k1) - (3780*a0**2*a1**7*np.sin(2*ang0*k0 - 5*ang0*k1 + 2*phi0 - 5*phi1))/(2*k0 - 5*k1) + (1890*a0**4*a1**5*np.sin(4*ang0*k0 - 5*ang0*k1 + 4*phi0 - 5*phi1))/(4*k0 - 5*k1) - (18900*a0**5*a1**4*np.sin(ang0*k0 - 4*ang0*k1 + phi0 - 4*phi1))/(k0 - 4*k1) - (22680*a0**3*a1**6*np.sin(ang0*k0 - 4*ang0*k1 + phi0 - 4*phi1))/(k0 - 4*k1) - (3780*a0*a1**8*np.sin(ang0*k0 - 4*ang0*k1 + phi0 - 4*phi1))/(k0 - 4*k1) + (9450*a0**5*a1**4*np.sin(3*ang0*k0 - 4*ang0*k1 + 3*phi0 - 4*phi1))/(3*k0 - 4*k1) + (7560*a0**3*a1**6*np.sin(3*ang0*k0 - 4*ang0*k1 + 3*phi0 - 4*phi1))/(3*k0 - 4*k1) + (1890*a0**5*a1**4*np.sin(5*ang0*k0 - 4*ang0*k1 + 5*phi0 - 4*phi1))/(5*k0 - 4*k1) + (18900*a0**6*a1**3*np.sin(2*ang0*k0 - 3*ang0*k1 + 2*phi0 - 3*phi1))/(2*k0 - 3*k1) + (37800*a0**4*a1**5*np.sin(2*ang0*k0 - 3*ang0*k1 + 2*phi0 - 3*phi1))/(2*k0 - 3*k1) + (11340*a0**2*a1**7*np.sin(2*ang0*k0 - 3*ang0*k1 + 2*phi0 - 3*phi1))/(2*k0 - 3*k1) + (7560*a0**6*a1**3*np.sin(4*ang0*k0 - 3*ang0*k1 + 4*phi0 - 3*phi1))/(4*k0 - 3*k1) + (9450*a0**4*a1**5*np.sin(4*ang0*k0 - 3*ang0*k1 + 4*phi0 - 3*phi1))/(4*k0 - 3*k1) - (420*a0**6*a1**3*np.sin(6*ang0*k0 - 3*ang0*k1 + 6*phi0 - 3*phi1))/(2*k0 - k1) + (18900*a0**7*a1**2*np.sin(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1))/(k0 - 2*k1) + (75600*a0**5*a1**4*np.sin(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1))/(k0 - 2*k1) + (56700*a0**3*a1**6*np.sin(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1))/(k0 - 2*k1) + (7560*a0*a1**8*np.sin(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1))/(k0 - 2*k1) - (420*a0**3*a1**6*np.sin(3*(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1)))/(k0 - 2*k1) + (11340*a0**7*a1**2*np.sin(3*ang0*k0 - 2*ang0*k1 + 3*phi0 - 2*phi1))/(3*k0 - 2*k1) + (37800*a0**5*a1**4*np.sin(3*ang0*k0 - 2*ang0*k1 + 3*phi0 - 2*phi1))/(3*k0 - 2*k1) + (18900*a0**3*a1**6*np.sin(3*ang0*k0 - 2*ang0*k1 + 3*phi0 - 2*phi1))/(3*k0 - 2*k1) - (3780*a0**7*a1**2*np.sin(5*ang0*k0 - 2*ang0*k1 + 5*phi0 - 2*phi1))/(5*k0 - 2*k1) - (7560*a0**5*a1**4*np.sin(5*ang0*k0 - 2*ang0*k1 + 5*phi0 - 2*phi1))/(5*k0 - 2*k1) - (1260*a0**7*a1**2*np.sin(7*ang0*k0 - 2*ang0*k1 + 7*phi0 - 2*phi1))/(7*k0 - 2*k1) + (7560*a0**8*a1*np.sin(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1))/(2*k0 - k1) + (56700*a0**6*a1**3*np.sin(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1))/(2*k0 - k1) + (75600*a0**4*a1**5*np.sin(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1))/(2*k0 - k1) + (18900*a0**2*a1**7*np.sin(2*ang0*k0 - ang0*k1 + 2*phi0 - phi1))/(2*k0 - k1) - (3780*a0**8*a1*np.sin(4*ang0*k0 - ang0*k1 + 4*phi0 - phi1))/(4*k0 - k1) - (22680*a0**6*a1**3*np.sin(4*ang0*k0 - ang0*k1 + 4*phi0 - phi1))/(4*k0 - k1) - (18900*a0**4*a1**5*np.sin(4*ang0*k0 - ang0*k1 + 4*phi0 - phi1))/(4*k0 - k1) - (2520*a0**8*a1*np.sin(6*ang0*k0 - ang0*k1 + 6*phi0 - phi1))/(6*k0 - k1) - (8820*a0**6*a1**3*np.sin(6*ang0*k0 - ang0*k1 + 6*phi0 - phi1))/(6*k0 - k1) + (315*a0**8*a1*np.sin(8*ang0*k0 - ang0*k1 + 8*phi0 - phi1))/(8*k0 - k1) + (9450*a0**8*a1*np.sin(ang0*k1 + phi1))/k1 + (75600*a0**6*a1**3*np.sin(ang0*k1 + phi1))/k1 + (113400*a0**4*a1**5*np.sin(ang0*k1 + phi1))/k1 + (37800*a0**2*a1**7*np.sin(ang0*k1 + phi1))/k1 + (1890*a1**9*np.sin(ang0*k1 + phi1))/k1 - (8400*a0**6*a1**3*np.sin(3*(ang0*k1 + phi1)))/k1 - (18900*a0**4*a1**5*np.sin(3*(ang0*k1 + phi1)))/k1 - (7560*a0**2*a1**7*np.sin(3*(ang0*k1 + phi1)))/k1 - (420*a1**9*np.sin(3*(ang0*k1 + phi1)))/k1 - (5292*a0**4*a1**5*np.sin(5*(ang0*k1 + phi1)))/k1 - (3528*a0**2*a1**7*np.sin(5*(ang0*k1 + phi1)))/k1 - (252*a1**9*np.sin(5*(ang0*k1 + phi1)))/k1 + (360*a0**2*a1**7*np.sin(7*(ang0*k1 + phi1)))/k1 + (45*a1**9*np.sin(7*(ang0*k1 + phi1)))/k1 + (35*a1**9*np.sin(9*(ang0*k1 + phi1)))/k1 - (7560*a0**8*a1*np.sin(2*ang0*k0 + ang0*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (56700*a0**6*a1**3*np.sin(2*ang0*k0 + ang0*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (75600*a0**4*a1**5*np.sin(2*ang0*k0 + ang0*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (18900*a0**2*a1**7*np.sin(2*ang0*k0 + ang0*k1 + 2*phi0 + phi1))/(2*k0 + k1) + (8820*a0**6*a1**3*np.sin(3*(2*ang0*k0 + ang0*k1 + 2*phi0 + phi1)))/(2*k0 + k1) - (8820*a0**8*a1*np.sin(4*ang0*k0 + ang0*k1 + 4*phi0 + phi1))/(4*k0 + k1) - (52920*a0**6*a1**3*np.sin(4*ang0*k0 + ang0*k1 + 4*phi0 + phi1))/(4*k0 + k1) - (44100*a0**4*a1**5*np.sin(4*ang0*k0 + ang0*k1 + 4*phi0 + phi1))/(4*k0 + k1) + (2520*a0**8*a1*np.sin(6*ang0*k0 + ang0*k1 + 6*phi0 + phi1))/(6*k0 + k1) + (8820*a0**6*a1**3*np.sin(6*ang0*k0 + ang0*k1 + 6*phi0 + phi1))/(6*k0 + k1) + (2835*a0**8*a1*np.sin(8*ang0*k0 + ang0*k1 + 8*phi0 + phi1))/(8*k0 + k1) - (18900*a0**7*a1**2*np.sin(ang0*k0 + 2*ang0*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (75600*a0**5*a1**4*np.sin(ang0*k0 + 2*ang0*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (56700*a0**3*a1**6*np.sin(ang0*k0 + 2*ang0*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (7560*a0*a1**8*np.sin(ang0*k0 + 2*ang0*k1 + phi0 + 2*phi1))/(k0 + 2*k1) + (8820*a0**3*a1**6*np.sin(3*(ang0*k0 + 2*ang0*k1 + phi0 + 2*phi1)))/(k0 + 2*k1) - (26460*a0**7*a1**2*np.sin(3*ang0*k0 + 2*ang0*k1 + 3*phi0 + 2*phi1))/(3*k0 + 2*k1) - (88200*a0**5*a1**4*np.sin(3*ang0*k0 + 2*ang0*k1 + 3*phi0 + 2*phi1))/(3*k0 + 2*k1) - (44100*a0**3*a1**6*np.sin(3*ang0*k0 + 2*ang0*k1 + 3*phi0 + 2*phi1))/(3*k0 + 2*k1) + (8820*a0**7*a1**2*np.sin(5*ang0*k0 + 2*ang0*k1 + 5*phi0 + 2*phi1))/(5*k0 + 2*k1) + (17640*a0**5*a1**4*np.sin(5*ang0*k0 + 2*ang0*k1 + 5*phi0 + 2*phi1))/(5*k0 + 2*k1) + (11340*a0**7*a1**2*np.sin(7*ang0*k0 + 2*ang0*k1 + 7*phi0 + 2*phi1))/(7*k0 + 2*k1) - (44100*a0**6*a1**3*np.sin(2*ang0*k0 + 3*ang0*k1 + 2*phi0 + 3*phi1))/(2*k0 + 3*k1) - (88200*a0**4*a1**5*np.sin(2*ang0*k0 + 3*ang0*k1 + 2*phi0 + 3*phi1))/(2*k0 + 3*k1) - (26460*a0**2*a1**7*np.sin(2*ang0*k0 + 3*ang0*k1 + 2*phi0 + 3*phi1))/(2*k0 + 3*k1) + (17640*a0**6*a1**3*np.sin(4*ang0*k0 + 3*ang0*k1 + 4*phi0 + 3*phi1))/(4*k0 + 3*k1) + (22050*a0**4*a1**5*np.sin(4*ang0*k0 + 3*ang0*k1 + 4*phi0 + 3*phi1))/(4*k0 + 3*k1) - (44100*a0**5*a1**4*np.sin(ang0*k0 + 4*ang0*k1 + phi0 + 4*phi1))/(k0 + 4*k1) - (52920*a0**3*a1**6*np.sin(ang0*k0 + 4*ang0*k1 + phi0 + 4*phi1))/(k0 + 4*k1) - (8820*a0*a1**8*np.sin(ang0*k0 + 4*ang0*k1 + phi0 + 4*phi1))/(k0 + 4*k1) + (22050*a0**5*a1**4*np.sin(3*ang0*k0 + 4*ang0*k1 + 3*phi0 + 4*phi1))/(3*k0 + 4*k1) + (17640*a0**3*a1**6*np.sin(3*ang0*k0 + 4*ang0*k1 + 3*phi0 + 4*phi1))/(3*k0 + 4*k1) + (39690*a0**5*a1**4*np.sin(5*ang0*k0 + 4*ang0*k1 + 5*phi0 + 4*phi1))/(5*k0 + 4*k1) + (17640*a0**4*a1**5*np.sin(2*ang0*k0 + 5*ang0*k1 + 2*phi0 + 5*phi1))/(2*k0 + 5*k1) + (8820*a0**2*a1**7*np.sin(2*ang0*k0 + 5*ang0*k1 + 2*phi0 + 5*phi1))/(2*k0 + 5*k1) + (39690*a0**4*a1**5*np.sin(4*ang0*k0 + 5*ang0*k1 + 4*phi0 + 5*phi1))/(4*k0 + 5*k1) + (8820*a0**3*a1**6*np.sin(ang0*k0 + 6*ang0*k1 + phi0 + 6*phi1))/(k0 + 6*k1) + (2520*a0*a1**8*np.sin(ang0*k0 + 6*ang0*k1 + phi0 + 6*phi1))/(k0 + 6*k1) + (11340*a0**2*a1**7*np.sin(2*ang0*k0 + 7*ang0*k1 + 2*phi0 + 7*phi1))/(2*k0 + 7*k1) + (2835*a0*a1**8*np.sin(ang0*k0 + 8*ang0*k1 + phi0 + 8*phi1))/(k0 + 8*k1))/80640.)
            res1 = (((1890*a0*(a0**8 + 20*a0**6*a1**2 + 60*a0**4*a1**4 + 40*a0**2*a1**6 + 5*a1**8)*np.sin(ang1*k0 + phi0))/k0 - (420*(a0**9 + 18*a0**7*a1**2 + 45*a0**5*a1**4 + 20*a0**3*a1**6)*np.sin(3*(ang1*k0 + phi0)))/k0 - (252*a0**9*np.sin(5*(ang1*k0 + phi0)))/k0 - (3528*a0**7*a1**2*np.sin(5*(ang1*k0 + phi0)))/k0 - (5292*a0**5*a1**4*np.sin(5*(ang1*k0 + phi0)))/k0 + (45*a0**9*np.sin(7*(ang1*k0 + phi0)))/k0 + (360*a0**7*a1**2*np.sin(7*(ang1*k0 + phi0)))/k0 + (35*a0**9*np.sin(9*(ang1*k0 + phi0)))/k0 + (315*a0*a1**8*np.sin(ang1*k0 - 8*ang1*k1 + phi0 - 8*phi1))/(k0 - 8*k1) - (1260*a0**2*a1**7*np.sin(2*ang1*k0 - 7*ang1*k1 + 2*phi0 - 7*phi1))/(2*k0 - 7*k1) - (8820*a0**3*a1**6*np.sin(ang1*k0 - 6*ang1*k1 + phi0 - 6*phi1))/(k0 - 6*k1) - (2520*a0*a1**8*np.sin(ang1*k0 - 6*ang1*k1 + phi0 - 6*phi1))/(k0 - 6*k1) - (7560*a0**4*a1**5*np.sin(2*ang1*k0 - 5*ang1*k1 + 2*phi0 - 5*phi1))/(2*k0 - 5*k1) - (3780*a0**2*a1**7*np.sin(2*ang1*k0 - 5*ang1*k1 + 2*phi0 - 5*phi1))/(2*k0 - 5*k1) + (1890*a0**4*a1**5*np.sin(4*ang1*k0 - 5*ang1*k1 + 4*phi0 - 5*phi1))/(4*k0 - 5*k1) - (18900*a0**5*a1**4*np.sin(ang1*k0 - 4*ang1*k1 + phi0 - 4*phi1))/(k0 - 4*k1) - (22680*a0**3*a1**6*np.sin(ang1*k0 - 4*ang1*k1 + phi0 - 4*phi1))/(k0 - 4*k1) - (3780*a0*a1**8*np.sin(ang1*k0 - 4*ang1*k1 + phi0 - 4*phi1))/(k0 - 4*k1) + (9450*a0**5*a1**4*np.sin(3*ang1*k0 - 4*ang1*k1 + 3*phi0 - 4*phi1))/(3*k0 - 4*k1) + (7560*a0**3*a1**6*np.sin(3*ang1*k0 - 4*ang1*k1 + 3*phi0 - 4*phi1))/(3*k0 - 4*k1) + (1890*a0**5*a1**4*np.sin(5*ang1*k0 - 4*ang1*k1 + 5*phi0 - 4*phi1))/(5*k0 - 4*k1) + (18900*a0**6*a1**3*np.sin(2*ang1*k0 - 3*ang1*k1 + 2*phi0 - 3*phi1))/(2*k0 - 3*k1) + (37800*a0**4*a1**5*np.sin(2*ang1*k0 - 3*ang1*k1 + 2*phi0 - 3*phi1))/(2*k0 - 3*k1) + (11340*a0**2*a1**7*np.sin(2*ang1*k0 - 3*ang1*k1 + 2*phi0 - 3*phi1))/(2*k0 - 3*k1) + (7560*a0**6*a1**3*np.sin(4*ang1*k0 - 3*ang1*k1 + 4*phi0 - 3*phi1))/(4*k0 - 3*k1) + (9450*a0**4*a1**5*np.sin(4*ang1*k0 - 3*ang1*k1 + 4*phi0 - 3*phi1))/(4*k0 - 3*k1) - (420*a0**6*a1**3*np.sin(6*ang1*k0 - 3*ang1*k1 + 6*phi0 - 3*phi1))/(2*k0 - k1) + (18900*a0**7*a1**2*np.sin(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1))/(k0 - 2*k1) + (75600*a0**5*a1**4*np.sin(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1))/(k0 - 2*k1) + (56700*a0**3*a1**6*np.sin(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1))/(k0 - 2*k1) + (7560*a0*a1**8*np.sin(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1))/(k0 - 2*k1) - (420*a0**3*a1**6*np.sin(3*(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1)))/(k0 - 2*k1) + (11340*a0**7*a1**2*np.sin(3*ang1*k0 - 2*ang1*k1 + 3*phi0 - 2*phi1))/(3*k0 - 2*k1) + (37800*a0**5*a1**4*np.sin(3*ang1*k0 - 2*ang1*k1 + 3*phi0 - 2*phi1))/(3*k0 - 2*k1) + (18900*a0**3*a1**6*np.sin(3*ang1*k0 - 2*ang1*k1 + 3*phi0 - 2*phi1))/(3*k0 - 2*k1) - (3780*a0**7*a1**2*np.sin(5*ang1*k0 - 2*ang1*k1 + 5*phi0 - 2*phi1))/(5*k0 - 2*k1) - (7560*a0**5*a1**4*np.sin(5*ang1*k0 - 2*ang1*k1 + 5*phi0 - 2*phi1))/(5*k0 - 2*k1) - (1260*a0**7*a1**2*np.sin(7*ang1*k0 - 2*ang1*k1 + 7*phi0 - 2*phi1))/(7*k0 - 2*k1) + (7560*a0**8*a1*np.sin(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1))/(2*k0 - k1) + (56700*a0**6*a1**3*np.sin(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1))/(2*k0 - k1) + (75600*a0**4*a1**5*np.sin(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1))/(2*k0 - k1) + (18900*a0**2*a1**7*np.sin(2*ang1*k0 - ang1*k1 + 2*phi0 - phi1))/(2*k0 - k1) - (3780*a0**8*a1*np.sin(4*ang1*k0 - ang1*k1 + 4*phi0 - phi1))/(4*k0 - k1) - (22680*a0**6*a1**3*np.sin(4*ang1*k0 - ang1*k1 + 4*phi0 - phi1))/(4*k0 - k1) - (18900*a0**4*a1**5*np.sin(4*ang1*k0 - ang1*k1 + 4*phi0 - phi1))/(4*k0 - k1) - (2520*a0**8*a1*np.sin(6*ang1*k0 - ang1*k1 + 6*phi0 - phi1))/(6*k0 - k1) - (8820*a0**6*a1**3*np.sin(6*ang1*k0 - ang1*k1 + 6*phi0 - phi1))/(6*k0 - k1) + (315*a0**8*a1*np.sin(8*ang1*k0 - ang1*k1 + 8*phi0 - phi1))/(8*k0 - k1) + (9450*a0**8*a1*np.sin(ang1*k1 + phi1))/k1 + (75600*a0**6*a1**3*np.sin(ang1*k1 + phi1))/k1 + (113400*a0**4*a1**5*np.sin(ang1*k1 + phi1))/k1 + (37800*a0**2*a1**7*np.sin(ang1*k1 + phi1))/k1 + (1890*a1**9*np.sin(ang1*k1 + phi1))/k1 - (8400*a0**6*a1**3*np.sin(3*(ang1*k1 + phi1)))/k1 - (18900*a0**4*a1**5*np.sin(3*(ang1*k1 + phi1)))/k1 - (7560*a0**2*a1**7*np.sin(3*(ang1*k1 + phi1)))/k1 - (420*a1**9*np.sin(3*(ang1*k1 + phi1)))/k1 - (5292*a0**4*a1**5*np.sin(5*(ang1*k1 + phi1)))/k1 - (3528*a0**2*a1**7*np.sin(5*(ang1*k1 + phi1)))/k1 - (252*a1**9*np.sin(5*(ang1*k1 + phi1)))/k1 + (360*a0**2*a1**7*np.sin(7*(ang1*k1 + phi1)))/k1 + (45*a1**9*np.sin(7*(ang1*k1 + phi1)))/k1 + (35*a1**9*np.sin(9*(ang1*k1 + phi1)))/k1 - (7560*a0**8*a1*np.sin(2*ang1*k0 + ang1*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (56700*a0**6*a1**3*np.sin(2*ang1*k0 + ang1*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (75600*a0**4*a1**5*np.sin(2*ang1*k0 + ang1*k1 + 2*phi0 + phi1))/(2*k0 + k1) - (18900*a0**2*a1**7*np.sin(2*ang1*k0 + ang1*k1 + 2*phi0 + phi1))/(2*k0 + k1) + (8820*a0**6*a1**3*np.sin(3*(2*ang1*k0 + ang1*k1 + 2*phi0 + phi1)))/(2*k0 + k1) - (8820*a0**8*a1*np.sin(4*ang1*k0 + ang1*k1 + 4*phi0 + phi1))/(4*k0 + k1) - (52920*a0**6*a1**3*np.sin(4*ang1*k0 + ang1*k1 + 4*phi0 + phi1))/(4*k0 + k1) - (44100*a0**4*a1**5*np.sin(4*ang1*k0 + ang1*k1 + 4*phi0 + phi1))/(4*k0 + k1) + (2520*a0**8*a1*np.sin(6*ang1*k0 + ang1*k1 + 6*phi0 + phi1))/(6*k0 + k1) + (8820*a0**6*a1**3*np.sin(6*ang1*k0 + ang1*k1 + 6*phi0 + phi1))/(6*k0 + k1) + (2835*a0**8*a1*np.sin(8*ang1*k0 + ang1*k1 + 8*phi0 + phi1))/(8*k0 + k1) - (18900*a0**7*a1**2*np.sin(ang1*k0 + 2*ang1*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (75600*a0**5*a1**4*np.sin(ang1*k0 + 2*ang1*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (56700*a0**3*a1**6*np.sin(ang1*k0 + 2*ang1*k1 + phi0 + 2*phi1))/(k0 + 2*k1) - (7560*a0*a1**8*np.sin(ang1*k0 + 2*ang1*k1 + phi0 + 2*phi1))/(k0 + 2*k1) + (8820*a0**3*a1**6*np.sin(3*(ang1*k0 + 2*ang1*k1 + phi0 + 2*phi1)))/(k0 + 2*k1) - (26460*a0**7*a1**2*np.sin(3*ang1*k0 + 2*ang1*k1 + 3*phi0 + 2*phi1))/(3*k0 + 2*k1) - (88200*a0**5*a1**4*np.sin(3*ang1*k0 + 2*ang1*k1 + 3*phi0 + 2*phi1))/(3*k0 + 2*k1) - (44100*a0**3*a1**6*np.sin(3*ang1*k0 + 2*ang1*k1 + 3*phi0 + 2*phi1))/(3*k0 + 2*k1) + (8820*a0**7*a1**2*np.sin(5*ang1*k0 + 2*ang1*k1 + 5*phi0 + 2*phi1))/(5*k0 + 2*k1) + (17640*a0**5*a1**4*np.sin(5*ang1*k0 + 2*ang1*k1 + 5*phi0 + 2*phi1))/(5*k0 + 2*k1) + (11340*a0**7*a1**2*np.sin(7*ang1*k0 + 2*ang1*k1 + 7*phi0 + 2*phi1))/(7*k0 + 2*k1) - (44100*a0**6*a1**3*np.sin(2*ang1*k0 + 3*ang1*k1 + 2*phi0 + 3*phi1))/(2*k0 + 3*k1) - (88200*a0**4*a1**5*np.sin(2*ang1*k0 + 3*ang1*k1 + 2*phi0 + 3*phi1))/(2*k0 + 3*k1) - (26460*a0**2*a1**7*np.sin(2*ang1*k0 + 3*ang1*k1 + 2*phi0 + 3*phi1))/(2*k0 + 3*k1) + (17640*a0**6*a1**3*np.sin(4*ang1*k0 + 3*ang1*k1 + 4*phi0 + 3*phi1))/(4*k0 + 3*k1) + (22050*a0**4*a1**5*np.sin(4*ang1*k0 + 3*ang1*k1 + 4*phi0 + 3*phi1))/(4*k0 + 3*k1) - (44100*a0**5*a1**4*np.sin(ang1*k0 + 4*ang1*k1 + phi0 + 4*phi1))/(k0 + 4*k1) - (52920*a0**3*a1**6*np.sin(ang1*k0 + 4*ang1*k1 + phi0 + 4*phi1))/(k0 + 4*k1) - (8820*a0*a1**8*np.sin(ang1*k0 + 4*ang1*k1 + phi0 + 4*phi1))/(k0 + 4*k1) + (22050*a0**5*a1**4*np.sin(3*ang1*k0 + 4*ang1*k1 + 3*phi0 + 4*phi1))/(3*k0 + 4*k1) + (17640*a0**3*a1**6*np.sin(3*ang1*k0 + 4*ang1*k1 + 3*phi0 + 4*phi1))/(3*k0 + 4*k1) + (39690*a0**5*a1**4*np.sin(5*ang1*k0 + 4*ang1*k1 + 5*phi0 + 4*phi1))/(5*k0 + 4*k1) + (17640*a0**4*a1**5*np.sin(2*ang1*k0 + 5*ang1*k1 + 2*phi0 + 5*phi1))/(2*k0 + 5*k1) + (8820*a0**2*a1**7*np.sin(2*ang1*k0 + 5*ang1*k1 + 2*phi0 + 5*phi1))/(2*k0 + 5*k1) + (39690*a0**4*a1**5*np.sin(4*ang1*k0 + 5*ang1*k1 + 4*phi0 + 5*phi1))/(4*k0 + 5*k1) + (8820*a0**3*a1**6*np.sin(ang1*k0 + 6*ang1*k1 + phi0 + 6*phi1))/(k0 + 6*k1) + (2520*a0*a1**8*np.sin(ang1*k0 + 6*ang1*k1 + phi0 + 6*phi1))/(k0 + 6*k1) + (11340*a0**2*a1**7*np.sin(2*ang1*k0 + 7*ang1*k1 + 2*phi0 + 7*phi1))/(2*k0 + 7*k1) + (2835*a0*a1**8*np.sin(ang1*k0 + 8*ang1*k1 + phi0 + 8*phi1))/(k0 + 8*k1))/80640.)
            return res1 - res0
        elif j == 5:
            res0 = (((-150*a0**2*(a0**8 + 24*a0**6*a1**2 + 90*a0**4*a1**4 + 80*a0**2*a1**6 + 15*a1**8)*np.cos(2*(ang0*k0 + phi0)))/k0 + (25*a0**6*(a0**4 + 16*a0**2*a1**2 + 28*a1**4)*np.cos(6*(ang0*k0 + phi0)))/k0 - (3*a0**10*np.cos(10*(ang0*k0 + phi0)))/k0 - (1200*a0**3*a1**7*np.cos(ang0*k0 - 7*ang0*k1 + phi0 - 7*phi1))/(k0 - 7*k1) - (300*a0*a1**9*np.cos(ang0*k0 - 7*ang0*k1 + phi0 - 7*phi1))/(k0 - 7*k1) + (150*a0**4*a1**6*np.cos(4*ang0*k0 - 6*ang0*k1 + 4*phi0 - 6*phi1))/(2*k0 - 3*k1) + (1800*a0**5*a1**5*np.cos(3*ang0*k0 - 5*ang0*k1 + 3*phi0 - 5*phi1))/(3*k0 - 5*k1) + (1200*a0**3*a1**7*np.cos(3*ang0*k0 - 5*ang0*k1 + 3*phi0 - 5*phi1))/(3*k0 - 5*k1) - (75*a0**2*a1**8*np.cos(2*(ang0*k0 - 4*ang0*k1 + phi0 - 4*phi1)))/(k0 - 4*k1) - (150*a0**6*a1**4*np.cos(6*ang0*k0 - 4*ang0*k1 + 6*phi0 - 4*phi1))/(3*k0 - 2*k1) + (6000*a0**7*a1**3*np.cos(ang0*k0 - 3*ang0*k1 + phi0 - 3*phi1))/(k0 - 3*k1) + (18000*a0**5*a1**5*np.cos(ang0*k0 - 3*ang0*k1 + phi0 - 3*phi1))/(k0 - 3*k1) + (10800*a0**3*a1**7*np.cos(ang0*k0 - 3*ang0*k1 + phi0 - 3*phi1))/(k0 - 3*k1) + (1200*a0*a1**9*np.cos(ang0*k0 - 3*ang0*k1 + phi0 - 3*phi1))/(k0 - 3*k1) - (1200*a0**7*a1**3*np.cos(5*ang0*k0 - 3*ang0*k1 + 5*phi0 - 3*phi1))/(5*k0 - 3*k1) - (1800*a0**5*a1**5*np.cos(5*ang0*k0 - 3*ang0*k1 + 5*phi0 - 3*phi1))/(5*k0 - 3*k1) + (2250*a0**6*a1**4*np.cos(2*(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1)))/(k0 - 2*k1) + (3600*a0**4*a1**6*np.cos(2*(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1)))/(k0 - 2*k1) + (900*a0**2*a1**8*np.cos(2*(ang0*k0 - 2*ang0*k1 + phi0 - 2*phi1)))/(k0 - 2*k1) - (900*a0**8*a1**2*np.cos(4*ang0*k0 - 2*ang0*k1 + 4*phi0 - 2*phi1))/(2*k0 - k1) - (3600*a0**6*a1**4*np.cos(4*ang0*k0 - 2*ang0*k1 + 4*phi0 - 2*phi1))/(2*k0 - k1) - (2250*a0**4*a1**6*np.cos(4*ang0*k0 - 2*ang0*k1 + 4*phi0 - 2*phi1))/(2*k0 - k1) + (75*a0**8*a1**2*np.cos(8*ang0*k0 - 2*ang0*k1 + 8*phi0 - 2*phi1))/(4*k0 - k1) - (1200*a0**9*a1*np.cos(3*ang0*k0 - ang0*k1 + 3*phi0 - phi1))/(3*k0 - k1) - (10800*a0**7*a1**3*np.cos(3*ang0*k0 - ang0*k1 + 3*phi0 - phi1))/(3*k0 - k1) - (18000*a0**5*a1**5*np.cos(3*ang0*k0 - ang0*k1 + 3*phi0 - phi1))/(3*k0 - k1) - (6000*a0**3*a1**7*np.cos(3*ang0*k0 - ang0*k1 + 3*phi0 - phi1))/(3*k0 - k1) + (300*a0**9*a1*np.cos(7*ang0*k0 - ang0*k1 + 7*phi0 - phi1))/(7*k0 - k1) + (1200*a0**7*a1**3*np.cos(7*ang0*k0 - ang0*k1 + 7*phi0 - phi1))/(7*k0 - k1) - (2250*a0**8*a1**2*np.cos(2*(ang0*k1 + phi1)))/k1 - (12000*a0**6*a1**4*np.cos(2*(ang0*k1 + phi1)))/k1 - (13500*a0**4*a1**6*np.cos(2*(ang0*k1 + phi1)))/k1 - (3600*a0**2*a1**8*np.cos(2*(ang0*k1 + phi1)))/k1 - (150*a1**10*np.cos(2*(ang0*k1 + phi1)))/k1 + (700*a0**4*a1**6*np.cos(6*(ang0*k1 + phi1)))/k1 + (400*a0**2*a1**8*np.cos(6*(ang0*k1 + phi1)))/k1 + (25*a1**10*np.cos(6*(ang0*k1 + phi1)))/k1 - (3*a1**10*np.cos(10*(ang0*k1 + phi1)))/k1 - (1800*a0**9*a1*np.cos(ang0*(k0 + k1) + phi0 + phi1))/(k0 + k1) - (18000*a0**7*a1**3*np.cos(ang0*(k0 + k1) + phi0 + phi1))/(k0 + k1) - (36000*a0**5*a1**5*np.cos(ang0*(k0 + k1) + phi0 + phi1))/(k0 + k1) - (18000*a0**3*a1**7*np.cos(ang0*(k0 + k1) + phi0 + phi1))/(k0 + k1) - (1800*a0*a1**9*np.cos(ang0*(k0 + k1) + phi0 + phi1))/(k0 + k1) + (2800*a0**7*a1**3*np.cos(3*(ang0*(k0 + k1) + phi0 + phi1)))/(k0 + k1) + (7000*a0**5*a1**5*np.cos(3*(ang0*(k0 + k1) + phi0 + phi1)))/(k0 + k1) + (2800*a0**3*a1**7*np.cos(3*(ang0*(k0 + k1) + phi0 + phi1)))/(k0 + k1) - (1512*a0**5*a1**5*np.cos(5*(ang0*(k0 + k1) + phi0 + phi1)))/(k0 + k1) + (2100*a0**8*a1**2*np.cos(2*(2*ang0*k0 + ang0*k1 + 2*phi0 + phi1)))/(2*k0 + k1) + (8400*a0**6*a1**4*np.cos(2*(2*ang0*k0 + ang0*k1 + 2*phi0 + phi1)))/(2*k0 + k1) + (5250*a0**4*a1**6*np.cos(2*(2*ang0*k0 + ang0*k1 + 2*phi0 + phi1)))/(2*k0 + k1) - (675*a0**8*a1**2*np.cos(2*(4*ang0*k0 + ang0*k1 + 4*phi0 + phi1)))/(4*k0 + k1) + (1200*a0**9*a1*np.cos(5*ang0*k0 + ang0*k1 + 5*phi0 + phi1))/(5*k0 + k1) + (8400*a0**7*a1**3*np.cos(5*ang0*k0 + ang0*k1 + 5*phi0 + phi1))/(5*k0 + k1) + (8400*a0**5*a1**5*np.cos(5*ang0*k0 + ang0*k1 + 5*phi0 + phi1))/(5*k0 + k1) - (300*a0**9*a1*np.cos(9*ang0*k0 + ang0*k1 + 9*phi0 + phi1))/(9*k0 + k1) + (5250*a0**6*a1**4*np.cos(2*(ang0*k0 + 2*ang0*k1 + phi0 + 2*phi1)))/(k0 + 2*k1) + (8400*a0**4*a1**6*np.cos(2*(ang0*k0 + 2*ang0*k1 + phi0 + 2*phi1)))/(k0 + 2*k1) + (2100*a0**2*a1**8*np.cos(2*(ang0*k0 + 2*ang0*k1 + phi0 + 2*phi1)))/(k0 + 2*k1) - (3600*a0**7*a1**3*np.cos(7*ang0*k0 + 3*ang0*k1 + 7*phi0 + 3*phi1))/(7*k0 + 3*k1) - (675*a0**2*a1**8*np.cos(2*(ang0*k0 + 4*ang0*k1 + phi0 + 4*phi1)))/(k0 + 4*k1) - (3150*a0**6*a1**4*np.cos(6*ang0*k0 + 4*ang0*k1 + 6*phi0 + 4*phi1))/(3*k0 + 2*k1) + (8400*a0**5*a1**5*np.cos(ang0*k0 + 5*ang0*k1 + phi0 + 5*phi1))/(k0 + 5*k1) + (8400*a0**3*a1**7*np.cos(ang0*k0 + 5*ang0*k1 + phi0 + 5*phi1))/(k0 + 5*k1) + (1200*a0*a1**9*np.cos(ang0*k0 + 5*ang0*k1 + phi0 + 5*phi1))/(k0 + 5*k1) - (3150*a0**4*a1**6*np.cos(4*ang0*k0 + 6*ang0*k1 + 4*phi0 + 6*phi1))/(2*k0 + 3*k1) - (3600*a0**3*a1**7*np.cos(3*ang0*k0 + 7*ang0*k1 + 3*phi0 + 7*phi1))/(3*k0 + 7*k1) - (300*a0*a1**9*np.cos(ang0*k0 + 9*ang0*k1 + phi0 + 9*phi1))/(k0 + 9*k1))/15360.)
            res1 = (((-150*a0**2*(a0**8 + 24*a0**6*a1**2 + 90*a0**4*a1**4 + 80*a0**2*a1**6 + 15*a1**8)*np.cos(2*(ang1*k0 + phi0)))/k0 + (25*a0**6*(a0**4 + 16*a0**2*a1**2 + 28*a1**4)*np.cos(6*(ang1*k0 + phi0)))/k0 - (3*a0**10*np.cos(10*(ang1*k0 + phi0)))/k0 - (1200*a0**3*a1**7*np.cos(ang1*k0 - 7*ang1*k1 + phi0 - 7*phi1))/(k0 - 7*k1) - (300*a0*a1**9*np.cos(ang1*k0 - 7*ang1*k1 + phi0 - 7*phi1))/(k0 - 7*k1) + (150*a0**4*a1**6*np.cos(4*ang1*k0 - 6*ang1*k1 + 4*phi0 - 6*phi1))/(2*k0 - 3*k1) + (1800*a0**5*a1**5*np.cos(3*ang1*k0 - 5*ang1*k1 + 3*phi0 - 5*phi1))/(3*k0 - 5*k1) + (1200*a0**3*a1**7*np.cos(3*ang1*k0 - 5*ang1*k1 + 3*phi0 - 5*phi1))/(3*k0 - 5*k1) - (75*a0**2*a1**8*np.cos(2*(ang1*k0 - 4*ang1*k1 + phi0 - 4*phi1)))/(k0 - 4*k1) - (150*a0**6*a1**4*np.cos(6*ang1*k0 - 4*ang1*k1 + 6*phi0 - 4*phi1))/(3*k0 - 2*k1) + (6000*a0**7*a1**3*np.cos(ang1*k0 - 3*ang1*k1 + phi0 - 3*phi1))/(k0 - 3*k1) + (18000*a0**5*a1**5*np.cos(ang1*k0 - 3*ang1*k1 + phi0 - 3*phi1))/(k0 - 3*k1) + (10800*a0**3*a1**7*np.cos(ang1*k0 - 3*ang1*k1 + phi0 - 3*phi1))/(k0 - 3*k1) + (1200*a0*a1**9*np.cos(ang1*k0 - 3*ang1*k1 + phi0 - 3*phi1))/(k0 - 3*k1) - (1200*a0**7*a1**3*np.cos(5*ang1*k0 - 3*ang1*k1 + 5*phi0 - 3*phi1))/(5*k0 - 3*k1) - (1800*a0**5*a1**5*np.cos(5*ang1*k0 - 3*ang1*k1 + 5*phi0 - 3*phi1))/(5*k0 - 3*k1) + (2250*a0**6*a1**4*np.cos(2*(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1)))/(k0 - 2*k1) + (3600*a0**4*a1**6*np.cos(2*(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1)))/(k0 - 2*k1) + (900*a0**2*a1**8*np.cos(2*(ang1*k0 - 2*ang1*k1 + phi0 - 2*phi1)))/(k0 - 2*k1) - (900*a0**8*a1**2*np.cos(4*ang1*k0 - 2*ang1*k1 + 4*phi0 - 2*phi1))/(2*k0 - k1) - (3600*a0**6*a1**4*np.cos(4*ang1*k0 - 2*ang1*k1 + 4*phi0 - 2*phi1))/(2*k0 - k1) - (2250*a0**4*a1**6*np.cos(4*ang1*k0 - 2*ang1*k1 + 4*phi0 - 2*phi1))/(2*k0 - k1) + (75*a0**8*a1**2*np.cos(8*ang1*k0 - 2*ang1*k1 + 8*phi0 - 2*phi1))/(4*k0 - k1) - (1200*a0**9*a1*np.cos(3*ang1*k0 - ang1*k1 + 3*phi0 - phi1))/(3*k0 - k1) - (10800*a0**7*a1**3*np.cos(3*ang1*k0 - ang1*k1 + 3*phi0 - phi1))/(3*k0 - k1) - (18000*a0**5*a1**5*np.cos(3*ang1*k0 - ang1*k1 + 3*phi0 - phi1))/(3*k0 - k1) - (6000*a0**3*a1**7*np.cos(3*ang1*k0 - ang1*k1 + 3*phi0 - phi1))/(3*k0 - k1) + (300*a0**9*a1*np.cos(7*ang1*k0 - ang1*k1 + 7*phi0 - phi1))/(7*k0 - k1) + (1200*a0**7*a1**3*np.cos(7*ang1*k0 - ang1*k1 + 7*phi0 - phi1))/(7*k0 - k1) - (2250*a0**8*a1**2*np.cos(2*(ang1*k1 + phi1)))/k1 - (12000*a0**6*a1**4*np.cos(2*(ang1*k1 + phi1)))/k1 - (13500*a0**4*a1**6*np.cos(2*(ang1*k1 + phi1)))/k1 - (3600*a0**2*a1**8*np.cos(2*(ang1*k1 + phi1)))/k1 - (150*a1**10*np.cos(2*(ang1*k1 + phi1)))/k1 + (700*a0**4*a1**6*np.cos(6*(ang1*k1 + phi1)))/k1 + (400*a0**2*a1**8*np.cos(6*(ang1*k1 + phi1)))/k1 + (25*a1**10*np.cos(6*(ang1*k1 + phi1)))/k1 - (3*a1**10*np.cos(10*(ang1*k1 + phi1)))/k1 - (1800*a0**9*a1*np.cos(ang1*(k0 + k1) + phi0 + phi1))/(k0 + k1) - (18000*a0**7*a1**3*np.cos(ang1*(k0 + k1) + phi0 + phi1))/(k0 + k1) - (36000*a0**5*a1**5*np.cos(ang1*(k0 + k1) + phi0 + phi1))/(k0 + k1) - (18000*a0**3*a1**7*np.cos(ang1*(k0 + k1) + phi0 + phi1))/(k0 + k1) - (1800*a0*a1**9*np.cos(ang1*(k0 + k1) + phi0 + phi1))/(k0 + k1) + (2800*a0**7*a1**3*np.cos(3*(ang1*(k0 + k1) + phi0 + phi1)))/(k0 + k1) + (7000*a0**5*a1**5*np.cos(3*(ang1*(k0 + k1) + phi0 + phi1)))/(k0 + k1) + (2800*a0**3*a1**7*np.cos(3*(ang1*(k0 + k1) + phi0 + phi1)))/(k0 + k1) - (1512*a0**5*a1**5*np.cos(5*(ang1*(k0 + k1) + phi0 + phi1)))/(k0 + k1) + (2100*a0**8*a1**2*np.cos(2*(2*ang1*k0 + ang1*k1 + 2*phi0 + phi1)))/(2*k0 + k1) + (8400*a0**6*a1**4*np.cos(2*(2*ang1*k0 + ang1*k1 + 2*phi0 + phi1)))/(2*k0 + k1) + (5250*a0**4*a1**6*np.cos(2*(2*ang1*k0 + ang1*k1 + 2*phi0 + phi1)))/(2*k0 + k1) - (675*a0**8*a1**2*np.cos(2*(4*ang1*k0 + ang1*k1 + 4*phi0 + phi1)))/(4*k0 + k1) + (1200*a0**9*a1*np.cos(5*ang1*k0 + ang1*k1 + 5*phi0 + phi1))/(5*k0 + k1) + (8400*a0**7*a1**3*np.cos(5*ang1*k0 + ang1*k1 + 5*phi0 + phi1))/(5*k0 + k1) + (8400*a0**5*a1**5*np.cos(5*ang1*k0 + ang1*k1 + 5*phi0 + phi1))/(5*k0 + k1) - (300*a0**9*a1*np.cos(9*ang1*k0 + ang1*k1 + 9*phi0 + phi1))/(9*k0 + k1) + (5250*a0**6*a1**4*np.cos(2*(ang1*k0 + 2*ang1*k1 + phi0 + 2*phi1)))/(k0 + 2*k1) + (8400*a0**4*a1**6*np.cos(2*(ang1*k0 + 2*ang1*k1 + phi0 + 2*phi1)))/(k0 + 2*k1) + (2100*a0**2*a1**8*np.cos(2*(ang1*k0 + 2*ang1*k1 + phi0 + 2*phi1)))/(k0 + 2*k1) - (3600*a0**7*a1**3*np.cos(7*ang1*k0 + 3*ang1*k1 + 7*phi0 + 3*phi1))/(7*k0 + 3*k1) - (675*a0**2*a1**8*np.cos(2*(ang1*k0 + 4*ang1*k1 + phi0 + 4*phi1)))/(k0 + 4*k1) - (3150*a0**6*a1**4*np.cos(6*ang1*k0 + 4*ang1*k1 + 6*phi0 + 4*phi1))/(3*k0 + 2*k1) + (8400*a0**5*a1**5*np.cos(ang1*k0 + 5*ang1*k1 + phi0 + 5*phi1))/(k0 + 5*k1) + (8400*a0**3*a1**7*np.cos(ang1*k0 + 5*ang1*k1 + phi0 + 5*phi1))/(k0 + 5*k1) + (1200*a0*a1**9*np.cos(ang1*k0 + 5*ang1*k1 + phi0 + 5*phi1))/(k0 + 5*k1) - (3150*a0**4*a1**6*np.cos(4*ang1*k0 + 6*ang1*k1 + 4*phi0 + 6*phi1))/(2*k0 + 3*k1) - (3600*a0**3*a1**7*np.cos(3*ang1*k0 + 7*ang1*k1 + 3*phi0 + 7*phi1))/(3*k0 + 7*k1) - (300*a0*a1**9*np.cos(ang1*k0 + 9*ang1*k1 + phi0 + 9*phi1))/(k0 + 9*k1))/15360.)
            return res1 - res0
    else:
        print('invalid index i = ' + str(i) + ' in Iij')
        return 0.0
    
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
