import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from scipy import interpolate
import time
import scipy.integrate as integrate
from joblib import Parallel, delayed
import math
cachedir = 'cache\\'
from joblib import Memory
import scipy.optimize as sp_opt
import custom_minimizer 
memory = Memory(cachedir, verbose=0)
pi = np.pi
t0 = time.time()

'''
####INPUTS####
'''
#path to dep profile
#sputter_profile = 'depline_Kaufman.mat'
#sputter_profile = 'ExpData/depline_exp_130mm.mat'
filename = 'depz.txt'
C = 4.46 #thickness [nm] per minute
source = 0 #Choose source of get thickness data 1 - seimtra, 0 - experiment
val = 3 #1, 2, 3 - magnetron position

alpha0_sub = 0*pi
substrate_x_len = 100 # Substrate width, mm
substrate_y_len = 100 # Substrate length, mm
substrate_x_res = 0.05 # Substrate x resolution, 1/mm
substrate_y_res = 0.05 # Substrate y resolution, 1/mm

cores = 1 # number of jobs for paralleling
verbose = True # True: print message each time when function of deposition called
delete_cache = True # True: delete history of function evaluations in the beggining 
                    #of work. Warning: if = False, some changes in the code may be ignored
point_tolerance = 5/100 # needed relative tolerance for thickness in each point
max_angle_divisions = 10 # limit of da while integration = 1 degree / max_angle_divisions

holder_inner_radius = 20   # mm
holder_outer_radius = 145  # radius sampleholder, mm
deposition_len_x = 290 # mm
deposition_len_y = 290 # mm
deposition_res_x = 1 # 1/mm
deposition_res_y = 1 # 1/mm

R_bounds = (10, 60) # (min, max) mm
k_bounds = (1, 50) # (min, max)
NR_bounds = (1, 100)
x0 = [35, 4.1, 1] #initial guess for optimisation [R0, k0]


NM = {"method":"Nelder-Mead", "options":{"disp": True, "xatol":0.01, 
                                         "fatol":0.01, 'maxfev':200}, 
                                         "bounds":(R_bounds, k_bounds, NR_bounds)}

NM_custom = {"method":custom_minimizer.minimize_custom_neldermead,
             "options":{"disp": True, "xatol":(0.1, 0.001, 0.01), 
                                         "fatol":0.01, 'maxfev':200}, 
                                         "bounds":(R_bounds, k_bounds, NR_bounds)}

Powell = {"method":"Powell", "options":{"disp": True, "xtol":0.0001, 
                                        "ftol":0.01, 'maxfev':500, 
                                        "direc":np.array([[1,0.01, 0.1],[-1,0.01,-0.1]])}, 
                                        "bounds":(R_bounds, k_bounds, NR_bounds)}
minimizer = NM_custom
R_mc_interval = 5/100 #step for MC <= R_mc_interval*(R_max_bound-R_min_bound)
k_mc_interval = 5/100 #step for MC <= k_mc_interval*(k_max_bound-k_min_bound)\
NR_mc_interval = 15/100
R_min_step = 1 #step for MC >= R_min_step
k_min_step = 0.01 #step for MC >= k_min_step
NR_min_step = 1
mc_iter = 0 # number of Monte-Carlo algoritm's iterations (number of visited local minima) 
T = 2 #"temperature" for MC algoritm

''' 
Functions and methods
'''

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y

sqr = lambda x: np.power(x, 2)

def dep_profile(X, Y, center_x, center_y, C):
    #load('depline_Kaufman.mat')
    #load('ExpData/depline_exp_130mm.mat')
    depliney = np.genfromtxt('depliney.csv',delimiter=',')
    profile_x_len = 300
    r = np.arange(0, profile_x_len, profile_x_len/len(depliney))
    f = interpolate.interp1d(r, depliney, fill_value=depliney.min(), 
                             bounds_error=False)
    
    Z = f(np.sqrt(sqr(X+center_x)+sqr(Y+center_y)))
    norm = Z.max()
    Z = C*Z/Z.max()
    t = (lambda xy: C*f(np.sqrt(sqr(xy[0]+center_x)+sqr(xy[1]+center_y)))/norm)
    return Z, t


count = 0
def print_fun(x, f, accepted):
    global count
    count+=1
    if accepted == 1: s = 'accepted'
    else: s = 'rejected' 
    print("\n##############\n%d/%d Monte-Carlo step: minimum %.2f at R = %.3f, k = %.3f, NR = %.1f was %s\n##############\n" % (count, mc_iter, f, *x, s))

'''
####GEOMETRY + INITIALIZATION####
'''
if delete_cache: memory.clear(warn=False)
else: print('WARNING: memory has not cleared, changes in the code or settings may be ignored')
F_axial = False
deposition_offset_x = -deposition_len_x/2 # mm
deposition_offset_y = -deposition_len_y/2 # mm

deposition_rect_x = [deposition_offset_x, deposition_offset_x+deposition_len_x, 
                     deposition_offset_x+deposition_len_x, deposition_offset_x, 
                     deposition_offset_x]

deposition_rect_y = [deposition_offset_y, deposition_offset_y, 
                     deposition_offset_y+deposition_len_y, 
                     deposition_offset_y+deposition_len_y, deposition_offset_y]
    
ang=np.arange(0, 2*pi,0.01) 
holder_circle_inner_x=holder_inner_radius*np.cos(ang)
holder_circle_inner_y=holder_inner_radius*np.sin(ang)
holder_circle_outer_x=holder_outer_radius*np.cos(ang)
holder_circle_outer_y=holder_outer_radius*np.sin(ang)    

#### depoition profile meshing
deposition_coords_x = np.linspace(deposition_offset_x, 
                                  deposition_offset_x+deposition_len_x, 
                                  num=math.ceil(deposition_len_x*deposition_res_x))

deposition_coords_y = np.linspace(deposition_offset_y, 
                                  deposition_offset_y+deposition_len_y, 
                                  num=math.ceil(deposition_len_y*deposition_res_y))

deposition_coords_map_x, deposition_coords_map_y = np.meshgrid(deposition_coords_x, 
                                                               deposition_coords_y)

substrate_coords_x = np.linspace(-substrate_x_len/2, substrate_x_len/2, 
                                 num=math.ceil(substrate_x_len*substrate_x_res))

substrate_coords_y = np.linspace(-substrate_y_len/2, substrate_y_len/2, 
                                 num=math.ceil(substrate_y_len*substrate_y_res))

substrate_coords_map_x, substrate_coords_map_y = np.meshgrid(substrate_coords_x, 
                                                             substrate_coords_y)

substrate_rect_x = [substrate_coords_x.min(), substrate_coords_x.max(), 
                    substrate_coords_x.max(), substrate_coords_x.min(), 
                    substrate_coords_x.min()]

substrate_rect_y = [substrate_coords_y.max(), substrate_coords_y.max(), 
                    substrate_coords_y.min(), substrate_coords_y.min(), 
                    substrate_coords_y.max()]

rho = np.sqrt(sqr(substrate_coords_map_x) + sqr(substrate_coords_map_y))
alpha0 = np.arctan2(substrate_coords_map_y, substrate_coords_map_x)
#ind = [(i, j) for i in range(len(substrate_coords_map_x)) for j in range(len(substrate_coords_map_x[i]))]
ind = [(i, len(substrate_coords_map_x[0])//2) for i in range(len(substrate_coords_map_x))]
ind = ind + [(len(substrate_coords_map_x)//2, i) for i in range(len(substrate_coords_map_x[0]))]

plt.plot(substrate_rect_x, substrate_rect_y, color='black')

plt.plot(np.reshape(substrate_coords_map_x, (-1, 1)), 
         np.reshape(substrate_coords_map_y, (-1, 1)), 'x', label='mesh point')



plt.title('Substrate')
plt.xlabel('x, mm')
plt.ylabel('y, mm')
plt.show()

def xyp(i, j, a, R, k):
    x = R*np.cos(a+alpha0_sub)+rho[i,j]*np.cos(-a*k + alpha0[i,j])
    y = R*np.sin(a+alpha0_sub)+rho[i,j]*np.sin(-a*k + alpha0[i,j])
    return x, y

def calc(ind, R, k, NR, omega):
    i, j = ind
    I, I_err = integrate.quad(lambda a: F(xyp(i, j, a, R, k)), 0, NR*2*pi, 
                              limit=int(round(max_angle_divisions*360*NR)), 
                              epsrel=point_tolerance)
    
    return I/(2*pi*omega), I_err/(2*pi*omega) #Jacobian time to alpha
        
#######
if source == 1:
    RELdeposition_coords_map_z = np.rot90(np.loadtxt(filename, skiprows=1))
    row_dep = RELdeposition_coords_map_z.max()
    deposition_coords_map_z = C*(RELdeposition_coords_map_z/row_dep)
elif source == 0: 
    if val == 1:
        center_x, center_y = 80, 59
    elif val == 2:
        center_x, center_y = 80, -59
    elif val == 3:
        center_x, center_y = -105.8, 0
    else:
        raise ValueError('Incorrect magnetron position.')
    deposition_coords_map_z, F = dep_profile(deposition_coords_map_x, 
                                             deposition_coords_map_y, 
                                             center_x, center_y, C)
    F_axial = True

if not F_axial:
    F = interpolate.RegularGridInterpolator((deposition_coords_x, deposition_coords_y), 
                                            np.transpose(deposition_coords_map_z), 
                                            bounds_error=False)
'''
########### OPTIMIZATION #################
'''

time_f = []
if cores>1:
    @memory.cache
    def deposition(R, k, NR, omega):#parallel
        t0 = time.time()
        '''
        if R+np.sqrt(substrate_x_len**2+substrate_y_len**2)/2>holder_outer_radius:
                raise ValueError('Incorrect substate out of holder border.')
        '''
        ########### INTEGRATION #################
        I, I_err = zip(*Parallel(n_jobs=cores)(delayed(calc)(ij, R, k, NR, omega) for ij in ind)) #parallel
        I = np.reshape(I, (len(substrate_coords_map_x), len(substrate_coords_map_x[0])))
        I_err = np.reshape(I_err, (len(substrate_coords_map_x), len(substrate_coords_map_x[0])))
        h_1 = (1-I[len(I)//2,:].min()/I[len(I)//2,:].max())
        h_2 = (1-I[:,len(I[0])//2].min()/I[:,len(I[0])//2].max())
        heterogeneity = max(h_1, h_2)*100
        t1 = time.time()
        time_f.append(t1-t0)
        if verbose: print('%d calculation func called. computation time: %.1f s' % (len(time_f), time_f[-1]))
        return I, heterogeneity, I_err
    
elif cores==1:
    @memory.cache
    def deposition(R, k, NR, omega): #serial
        t0 = time.time()
        if R+np.sqrt(substrate_x_len**2+substrate_y_len**2)/2>holder_outer_radius:
                raise ValueError('Incorrect substate out of holder border.')
        ########### INTEGRATION #################
        I, I_err = zip(*[calc(ij, R, k, NR, omega) for ij in ind]) #serial
        I = np.array(I)
        #I = np.reshape(I, (len(substrate_coords_map_x), len(substrate_coords_map_x[0])))
        #I_err = np.reshape(I_err, (len(substrate_coords_map_x), len(substrate_coords_map_x[0])))
        #h_1 = (1-I[len(I)//2,:].min()/I[len(I)//2,:].max())
        #h_2 = (1-I[:,len(I[0])//2].min()/I[:,len(I[0])//2].max())
        heterogeneity = (1-I.min()/I.max())*100
        t1 = time.time()
        time_f.append(t1-t0)
        if verbose: print('%d calculation func called. computation time: %.1f s' % (len(time_f), time_f[-1]))
        return I, heterogeneity, I_err
     
else: raise ValueError('incorrect parameter "cores"')

omega = 3 #speed rev/min
NR = 65
process_time = NR/omega

def func(x):
    c = 0
    gate=100
    delta=(0.1, 0.01)
    if x[0]<R_bounds[0]+delta[0]:
        c+=gate*(R_bounds[0]+delta[0]-x[0])
    if x[0]>R_bounds[1]-delta[0]:  
        c+=gate*(x[0]+delta[0]-R_bounds[0])
    if x[1]<k_bounds[0]+delta[1]:
        c+=gate*(k_bounds[0]+delta[1]-x[1])
    if x[1]>k_bounds[1]-delta[1]:  
        c+=gate*(x[1]+delta[1]-k_bounds[0])
    h = deposition(*x, omega)[1]
    if verbose: 
        print('At R = %.2f, k = %.3f, NR = %.2f ---------- heterogeneity = %.2f ' % (*x, h))
    return c+h

mytakestep = custom_minimizer.CustomTakeStep((R_bounds[1]-R_bounds[0])*R_mc_interval, 
                        (k_bounds[1]-k_bounds[0])*k_mc_interval, 
                        (NR_bounds[1]-NR_bounds[0])*NR_mc_interval, 
                        R_min_step, k_min_step, NR_min_step, 
                        R_bounds, k_bounds, NR_bounds)

mybounds = custom_minimizer.CustomBounds(R_bounds, k_bounds, NR_bounds)

ret = sp_opt.basinhopping(func, x0, minimizer_kwargs=minimizer, niter=mc_iter, 
                          callback=print_fun, take_step=mytakestep, T=T, 
                          accept_test=mybounds)

R, k, NR = ret.x
h = ret.fun #heterogeneity
print("global minimum: R = %.1f, k = %.3f, NR = %.2f, heterogeneity = %.2f" % (R, k, NR, h))

I = deposition(R, k, NR, omega)[0]
t1 = time.time()
print('Full time: %d s\nfunc calls: %d\navg func computation time: %.2f s' % (t1-t0, len(time_f), np.mean(time_f)))

'''
plt.contourf(Rs, ks, heterogeneity)
#plt.clim(9,11)
plt.colorbar()
'''
#plt.plot(ks,heterogeneity)

'''
I = np.zeros(substrate_coords_map_x.shape)
I_err = np.zeros(substrate_coords_map_x.shape)
for i in range(len(substrate_coords_map_x)):
    for j in range(len(substrate_coords_map_x[0])):
        #I[i,j], I_err[i,j] = integrate.quad(lambda a0: F(xyp(i,j,a0)), 0, 
        NR*2*pi, limit=10*360*NR, epsrel=0.01)
    
        #I[i,j], I_err[i,j] = I[i,j]/(2*pi*omega), I_err[i,j]/(2*pi*omega) #Jacobian time to alpha
        I[i,j], I_err[i,j] = calc((i,j))
'''  

'''
PLOTTING
'''

substrate_rect_x = [R+substrate_coords_x.min(), R+substrate_coords_x.max(), 
                    R+substrate_coords_x.max(), R+substrate_coords_x.min(), 
                    R+substrate_coords_x.min()]

substrate_rect_y = [substrate_coords_y.max(), substrate_coords_y.max(), 
                    substrate_coords_y.min(), substrate_coords_y.min(), 
                    substrate_coords_y.max()]

plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
plt.contourf(deposition_coords_map_x, deposition_coords_map_y, 
             deposition_coords_map_z, 100)

plt.plot(holder_circle_inner_x, holder_circle_inner_y, linewidth=2, 
         color='black', linestyle='--')

plt.plot(holder_circle_outer_x, holder_circle_outer_y, linewidth=2, 
         color='black')

plt.plot(deposition_rect_x, deposition_rect_y, linewidth=2, color='green')
plt.plot(substrate_rect_x, substrate_rect_y, color='black')
plt.plot(np.reshape(R+substrate_coords_map_x, (-1, 1)), 
         np.reshape(substrate_coords_map_y, (-1, 1)), 'x')

plt.colorbar()
plt.xlim((min(deposition_rect_x), max(deposition_rect_x)))
plt.ylim((min(deposition_rect_y), max(deposition_rect_y)))
plt.xlabel('x, mm')
plt.ylabel('y, mm')
plt.title(f'Holder orientation, R = {R}')

plt.subplot(2,2,2)
plt.contourf(substrate_coords_map_x, substrate_coords_map_y, I/I.max())
plt.clim(I.min()/I.max(), 1)
plt.colorbar()
plt.xlabel('x, mm')
plt.ylabel('y, mm')
plt.title(f'Film heterogeneity $H = {round(h,2)}\\%$')


plt.subplot(2,2,3)
i,j = 1, 2
a = np.linspace(0, NR*2*pi, num=int(round(NR*360)))
plt.plot(*xyp(i,j,a, R, k), '-')
plt.plot(*xyp(i,j,a[0], R, k), 'o')
plt.plot(*xyp(i,j,a[-1], R, k), 'x')
plt.plot(substrate_rect_x, substrate_rect_y, color='black')
plt.contourf(R+substrate_coords_map_x, substrate_coords_map_y, I)
plt.plot(holder_circle_inner_x, holder_circle_inner_y, linewidth=2, 
color='black', linestyle='--')

plt.plot(holder_circle_outer_x, holder_circle_outer_y, linewidth=2, 
color='black')

plt.xlim((min(deposition_rect_x), max(deposition_rect_x)))
plt.ylim((min(deposition_rect_y), max(deposition_rect_y)))  

