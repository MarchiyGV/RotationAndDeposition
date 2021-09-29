from numpy import (
    convolve, ones, cos, sin, power, genfromtxt, arange, array, sqrt, pi,
    linspace, meshgrid, arctan2, rot90, transpose, loadtxt, reshape, mean, 
    log10, arcsin
    )
import numpy as np
#import numpy.matlib
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
    
    def __init__(self, 
                 fname_sim='depz.txt', #path to dep profile
                 fname_exp='depliney.txt',
                 rotation_type = 'Planet',
                 C=4.46,  #thickness [nm] per minute
                 source='Experiment',
                 magnetron_x = 0, #mm
                 magnetron_y = 0, #mm
                 substrate_shape = 'Circle',
                 substrate_radius = 50, #mm
                 substrate_x_len=100 , # Substrate width, mm
                 substrate_y_len=100, # Substrate length, mm
                 substrate_res=0.05, # Substrate x resolution, 1/mm
                 cores=1, # number of jobs for paralleling
                 verbose=True,  # True: print message each time when function of deposition called
                 delete_cache=True, # True: delete history of function evaluations in the beggining 
                            #of work. Warning: if = False, some changes in the code may be ignored
                 point_tolerance=5, # needed relative tolerance for thickness in each point
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
                 omega_s_max = 100,
                 omega_p_max = 100,
                 x0_1 = 35, #initial guess for optimisation [R0, k0]
                 x0_2 = 4.1,
                 x0_3 = 1,
                 minimizer = 'NM_custom',
                 R_mc_interval = 5, #step for MC <= R_mc_interval*(R_max_bound-R_min_bound)
                 k_mc_interval = 5, #step for MC <= k_mc_interval*(k_max_bound-k_min_bound)\
                 NR_mc_interval = 15,
                 R_min_step = 1, #step for MC >= R_min_step
                 k_min_step = 0.01, #step for MC >= k_min_step
                 NR_min_step = 1,
                 mc_iter = 2, # number of Monte-Carlo algoritm's iterations (number of visited local minima) 
                 T = 2 #"temperature" for MC algoritm
                 ):
        QObject.__init__(self)
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
        self.deposition_res_x = deposition_res_x # 1/mm
        self.deposition_res_y = deposition_res_y # 1/mm
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
            if R_min < R_min_self:
                error(f'При такой конструкции подложкодержателя и размере подложки минимальный радиус {round(R_min_self, R_decimals+1)} мм. Это значение установленно автоматически.')
            if R_max > R_max_self:
                error(f'При такой конструкции подложкодержателя и размере подложки максимальный радиус {round(R_max_self, R_decimals+1)} мм. Это значение установленно автоматически.')
        if R_min>R_max:
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
        self.R_bounds = (max(R_min, R_min_self), min(R_max, R_max_self)) # (min, max) mm
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
            rho = linspace(0, r_, num=ceil(r_*self.substrate_res))
            rho = rho[rho>0]
            angles = []
            rs = []
            for r in rho:
                n = ceil(2*pi*r*self.substrate_res)
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
        
        if rotation_type == 'Planet':
            self.xyp = self.xyp_planet
        elif rotation_type == 'Solar':
            self.xyp = self.xyp_solar
            joblib_ignore.append('k')
            
        self.time_f = []
        
        if cores>1:
            #ray.init()
            #self.deposition = self.memory.cache(self.deposition_ray, ignore=['self'])
            error('parrallel does not supported now')
            self.deposition = self.memory.cache(self.deposition_serial, ignore=joblib_ignore)           
        elif cores==1:
            self.deposition = self.memory.cache(self.deposition_serial, ignore=joblib_ignore)           
        else: error('incorrect parameter "cores"')
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
    '''
    def deposition_ray(self, R, k, NR, omega):#parallel
                t0 = time.time()
                with Pool_ray(self.cores) as p:
                    result = p.starmap(self.calc, [(ij, R, k, NR, omega) for ij in self.ind])
                    I, I_err = zip(*result)
                I = reshape(I, (len(self.substrate_coords_map_x), len(self.substrate_coords_map_x[0]))) #np.reshape
                t1 = time.time()
                self.time_f.append(t1-t0)
                if self.verbose: print('%d calculation func called. computation time: %.1f s' % (len(self.time_f), self.time_f[-1]))
                return I    
    '''      
    def deposition_serial(self, R, k, NR, omega): #serial
                t0 = time.time()
                ########### INTEGRATION #################
                I, I_err = zip(*[self.calc(i, R, k, NR, omega) for i in self.ind]) #serial
                #I = reshape(I, (len(self.substrate_coords_map_x), len(self.substrate_coords_map_x[0])))
                t1 = time.time()
                self.time_f.append(t1-t0)
                if self.verbose: print('%d calculation func called. computation time: %.1f s' % (len(self.time_f), self.time_f[-1]))
                return np.array(I)

    def xyp_planet(self, i, a, R, k):
        x = R*cos(a+self.alpha0_sub)+self.rho[i]*cos(a*k + self.alpha0[i])
        y = R*sin(a+self.alpha0_sub)+self.rho[i]*sin(a*k + self.alpha0[i])
        return x, y
    
    def xyp_solar(self, i, a, R, k=1):
        x = R*cos(a+self.alpha0_sub)+self.rho[i]*cos(a + self.alpha0[i])
        y = R*sin(a+self.alpha0_sub)+self.rho[i]*sin(a + self.alpha0[i])
        return x, y
    
    def calc(self, i, R, k, NR, omega):
        I, I_err = quad(lambda a: self.F(self.xyp(i, a, R, k)), 0, NR*2*pi, #scipy.integrate.quad
                                  limit=int(round(self.max_angle_divisions*360*NR)), 
                                  epsrel=self.point_tolerance)
        
        return I/(2*pi*omega), I_err/(2*pi*omega) #Jacobian time to alpha
    '''
    def heterogeneity(self, I):
        h_1 = (1-I[len(I)//2,:].min()/I[len(I)//2,:].max())
        h_2 = (1-I[:,len(I[0])//2].min()/I[:,len(I[0])//2].max())
        return max(h_1, h_2)*100
    '''
    def heterogeneity(self, I):
        return (1-I.min()/I.max())*100
    
    def grid_I(self, I):
        I_grid = np.zeros((self.substrate_rows, self.substrate_columns))
        dy = self.substrate_y_len/(self.substrate_rows-1)
        dx = self.substrate_x_len/(self.substrate_columns-1)
        for ind in self.ind:
            i = int(ceil((self.substrate_y_len/2+self.ys[ind])/dy))
            j = int(ceil((self.substrate_x_len/2+self.xs[ind])/dx))
            I_grid[i, j] = I[ind]
        return I_grid
    
class Optimizer(QObject):
    upd_signal = pyqtSignal(str)
    def __init__(self, model):
        QObject.__init__(self)
        self.model = model
        self.log = ''
        
    def optimisation(self):
        self.log = ''
        t0 = time.time()
        mytakestep = custom_minimizer.CustomTakeStep(
                                (self.model.R_bounds[1]-self.model.R_bounds[0])*self.model.R_mc_interval, 
                                (self.model.k_bounds[1]-self.model.k_bounds[0])*self.model.k_mc_interval, 
                                (self.model.NR_bounds[1]-self.model.NR_bounds[0])*self.model.NR_mc_interval, 
                                self.model.R_min_step, self.model.k_min_step, self.model.NR_min_step, 
                                self.model.R_bounds, self.model.k_bounds, self.model.NR_bounds)
        
        mybounds = custom_minimizer.CustomBounds(self.model.R_bounds, 
                                                 self.model.k_bounds, 
                                                 self.model.NR_bounds)
    
        ret = basinhopping(self.func, self.model.x0, minimizer_kwargs=self.model.minimizer, 
                           niter=self.model.mc_iter, callback=self.print_fun, 
                           take_step=mytakestep, T=self.model.T, accept_test=mybounds)
    
        R, k, NR = ret.x
        h = ret.fun #heterogeneity
        message = "global minimum: R = %.1f, k = %.3f, NR = %.2f, heterogeneity = %.2f" % (R, k, NR, h)    
        I = self.model.deposition(R, k, NR, 1)
        t1 = time.time()        
        message +='\nFull time: %d s\nfunc calls: %d\navg func computation time: %.2f s' % (t1-t0, len(self.model.time_f), mean(self.model.time_f))
        print(message)
        self.log += (message+'\n')
        self.upd_signal.emit(message)
        return I    
    
    def func(self, x):
        c = 0
        gate=100
        delta=(0.1, 0.01)
        if x[0]<self.model.R_bounds[0]+delta[0]:
            c+=gate*(self.model.R_bounds[0]+delta[0]-x[0])
        if x[0]>self.model.R_bounds[1]-delta[0]:  
            c+=gate*(x[0]+delta[0]-self.model.R_bounds[0])
        if x[1]<self.model.k_bounds[0]+delta[1]:
            c+=gate*(self.model.k_bounds[0]+delta[1]-x[1])
        if x[1]>self.model.k_bounds[1]-delta[1]:  
            c+=gate*(x[1]+delta[1]-self.model.k_bounds[0])
        h = self.model.heterogeneity(self.model.deposition(*x, 1))
        if self.model.verbose: 
            message = 'At R = %.2f, k = %.3f, NR = %.2f ---------- heterogeneity = %.2f ' % (*x, h)
            print(message)
            self.log += (message+'\n')
            self.upd_signal.emit(message)
        return c+h
    
    def print_fun(self, x, f, accepted):
        self.model.count+=1
        if accepted == 1: s = 'accepted'
        else: s = 'rejected' 
        message = "\n##############\n%d/%d Monte-Carlo step: minimum %.2f at R = %.3f, k = %.3f, NR = %.1f was %s\n##############\n" % (self.model.count, 1+self.model.mc_iter, f, *x, s)
        print(message)
        self.log += (message+'\n')
        self.upd_signal.emit(message)