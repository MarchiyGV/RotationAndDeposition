import warnings
import scipy
import numpy
import scipy.optimize as sp_opt

class CustomTakeStep:
   def __init__(self, R_mc_interval, k_mc_interval, NR_mc_interval, R_min_step, 
                k_min_step, NR_min_step, R_bounds, k_bounds, NR_bounds):
       self.R_max_step = (R_bounds[1]-R_bounds[0])*R_mc_interval
       self.k_max_step = (k_bounds[1]-k_bounds[0])*k_mc_interval
       self.NR_max_step = (NR_bounds[1]-NR_bounds[0])*R_mc_interval
       self.R_min_step = R_min_step
       self.k_min_step = k_min_step
       self.NR_min_step = NR_min_step
       self.R_bounds = R_bounds
       self.k_bounds = k_bounds
       self.NR_bounds = NR_bounds
   def __call__(self, x):
       R_1 = self.R_min_step
       k_1 = self.k_min_step
       R_2 = self.R_max_step
       k_2 = self.k_max_step
       NR_1 = self.NR_min_step
       NR_2 = self.NR_max_step
       while True:
           d = numpy.random.uniform(R_1, R_2)*numpy.random.choice([1,-1])
           if d+x[0]>=self.R_bounds[0] and d+x[0]<self.R_bounds[1]:
               x[0] += d
               #print('MC switch: dR = %.1f' % d)
               while True:
                   d = numpy.random.uniform(k_1, k_2)*numpy.random.choice([1,-1])
                   if d+x[1]>=self.k_bounds[0] and d+x[1]<self.k_bounds[1]:
                       x[1] += d
                       #print('MC switch: dk = %.2f' % d)
                       while True:
                           d = numpy.random.uniform(NR_1, NR_2)*numpy.random.choice([1,-1])
                           if d+x[2]>=self.NR_bounds[0] and d+x[2]<self.NR_bounds[1]:
                               x[2] += d
                               #print('MC switch: dNR = %.2f' % d)
                               break
                       #print('new: R = %.1f, k = %.2f, NR = %.2f' % tuple(x))
                       break
               break
       return x
   
class CustomBounds:
    def __init__(self, R_bounds, k_bounds, NR_bounds):
        self.xmax = numpy.array((R_bounds[1], k_bounds[1], NR_bounds[1]))
        self.xmin = numpy.array((R_bounds[0], k_bounds[0], NR_bounds[0]))
    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(numpy.all(x <= self.xmax))
        tmin = bool(numpy.all(x >= self.xmin))
        return tmax and tmin

def wrap_function(function, args):
    ncalls = [0]
    if function is None:
        return ncalls, None

    def function_wrapper(*wrapper_args):
        ncalls[0] += 1
        return function(*(wrapper_args + args))

    return ncalls, function_wrapper

# standard status messages of optimizers
_status_message = {'success': 'Optimization terminated successfully.',
                   'maxfev': 'Maximum number of function evaluations has '
                              'been exceeded.',
                   'maxiter': 'Maximum number of iterations has been '
                              'exceeded.',
                   'pr_loss': 'Desired error not necessarily achieved due '
                              'to precision loss.',
                   'nan': 'NaN result encountered.'}

def minimize_custom_neldermead(func, x0, args=(), callback=None,
                         maxiter=None, maxfev=None, disp=False,
                         return_all=False, initial_simplex=None,
                         xatol=False, fatol=1e-4, adaptive=False,
                         **unknown_options):
    """
    Minimization of scalar function of one or more variables using the
    Nelder-Mead algorithm.

    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    maxiter, maxfev : int
        Maximum allowed number of iterations and function evaluations.
        Will default to ``N*200``, where ``N`` is the number of
        variables, if neither `maxiter` or `maxfev` is set. If both
        `maxiter` and `maxfev` are set, minimization will stop at the
        first reached.
    initial_simplex : array_like of shape (N + 1, N)
        Initial simplex. If given, overrides `x0`.
        ``initial_simplex[j,:]`` should contain the coordinates of
        the j-th vertex of the ``N+1`` vertices in the simplex, where
        ``N`` is the dimension.
    xatol : ndarray of floats of shape (N,), optional
        Absolute error in xopt between iterations that is acceptable for
        convergence.
    fatol : number, optional
        Absolute error in func(xopt) between iterations that is acceptable for
        convergence.
    adaptive : bool, optional
        Adapt algorithm parameters to dimensionality of problem. Useful for
        high-dimensional minimization [1]_.

    References
    ----------
    .. [1] Gao, F. and Han, L.
       Implementing the Nelder-Mead simplex algorithm with adaptive
       parameters. 2012. Computational Optimization and Applications.
       51:1, pp. 259-277

    """
    if not xatol:
        xatol = [1e-4]*len(x0)
    xatol = numpy.array(xatol)
    if 'ftol' in unknown_options:
        warnings.warn("ftol is deprecated for Nelder-Mead,"
                      " use fatol instead. If you specified both, only"
                      " fatol is used.",
                      DeprecationWarning)
        if (numpy.isclose(fatol, 1e-4) and
                not numpy.isclose(unknown_options['ftol'], 1e-4)):
            # only ftol was probably specified, use it.
            fatol = unknown_options['ftol']
        unknown_options.pop('ftol')
    if 'xtol' in unknown_options:
        warnings.warn("xtol is deprecated for Nelder-Mead,"
                      " use xatol instead. If you specified both, only"
                      " xatol is used.",
                      DeprecationWarning)
        if (numpy.isclose(xatol, 1e-4) and
                not numpy.isclose(unknown_options['xtol'], 1e-4)):
            # only xtol was probably specified, use it.
            xatol = unknown_options['xtol']
        unknown_options.pop('xtol')

    maxfun = maxfev
    retall = return_all

    fcalls, func = wrap_function(func, args)

    if adaptive:
        dim = float(len(x0))
        rho = 1
        chi = 1 + 2/dim
        psi = 0.75 - 1/(2*dim)
        sigma = 1 - 1/dim
    else:
        rho = 1
        chi = 2
        psi = 0.5
        sigma = 0.5

    nonzdelt = 0.05
    zdelt = 0.00025

    x0 = numpy.asfarray(x0).flatten()

    if initial_simplex is None:
        N = len(x0)

        sim = numpy.zeros((N + 1, N), dtype=x0.dtype)
        sim[0] = x0
        for k in range(N):
            y = numpy.array(x0, copy=True)
            if y[k] != 0:
                y[k] = (1 + nonzdelt)*y[k]
            else:
                y[k] = zdelt
            sim[k + 1] = y
    else:
        sim = numpy.asfarray(initial_simplex).copy()
        if sim.ndim != 2 or sim.shape[0] != sim.shape[1] + 1:
            raise ValueError("`initial_simplex` should be an array of shape (N+1,N)")
        if len(x0) != sim.shape[1]:
            raise ValueError("Size of `initial_simplex` is not consistent with `x0`")
        N = sim.shape[1]

    if retall:
        allvecs = [sim[0]]

    # If neither are set, then set both to default
    if maxiter is None and maxfun is None:
        maxiter = N * 200
        maxfun = N * 200
    elif maxiter is None:
        # Convert remaining Nones, to numpy.inf, unless the other is numpy.inf, in
        # which case use the default to avoid unbounded iteration
        if maxfun == numpy.inf:
            maxiter = N * 200
        else:
            maxiter = numpy.inf
    elif maxfun is None:
        if maxiter == numpy.inf:
            maxfun = N * 200
        else:
            maxfun = numpy.inf

    one2np1 = list(range(1, N + 1))
    fsim = numpy.zeros((N + 1,), float)

    for k in range(N + 1):
        fsim[k] = func(sim[k])

    ind = numpy.argsort(fsim)
    fsim = numpy.take(fsim, ind, 0)
    # sort so sim[0,:] has the lowest function value
    sim = numpy.take(sim, ind, 0)

    iterations = 1

    while (fcalls[0] < maxfun and iterations < maxiter):
        if (numpy.all(numpy.max(numpy.abs(sim[1:] - sim[0]), axis=0) <= xatol) and 
            numpy.max(numpy.abs(fsim[0] - fsim[1:])) <= fatol):
            break

        xbar = numpy.add.reduce(sim[:-1], 0) / N
        xr = (1 + rho) * xbar - rho * sim[-1]
        fxr = func(xr)
        doshrink = 0

        if fxr < fsim[0]:
            xe = (1 + rho * chi) * xbar - rho * chi * sim[-1]
            fxe = func(xe)

            if fxe < fxr:
                sim[-1] = xe
                fsim[-1] = fxe
            else:
                sim[-1] = xr
                fsim[-1] = fxr
        else:  # fsim[0] <= fxr
            if fxr < fsim[-2]:
                sim[-1] = xr
                fsim[-1] = fxr
            else:  # fxr >= fsim[-2]
                # Perform contraction
                if fxr < fsim[-1]:
                    xc = (1 + psi * rho) * xbar - psi * rho * sim[-1]
                    fxc = func(xc)

                    if fxc <= fxr:
                        sim[-1] = xc
                        fsim[-1] = fxc
                    else:
                        doshrink = 1
                else:
                    # Perform an inside contraction
                    xcc = (1 - psi) * xbar + psi * sim[-1]
                    fxcc = func(xcc)

                    if fxcc < fsim[-1]:
                        sim[-1] = xcc
                        fsim[-1] = fxcc
                    else:
                        doshrink = 1

                if doshrink:
                    for j in one2np1:
                        sim[j] = sim[0] + sigma * (sim[j] - sim[0])
                        fsim[j] = func(sim[j])

        ind = numpy.argsort(fsim)
        sim = numpy.take(sim, ind, 0)
        fsim = numpy.take(fsim, ind, 0)
        if callback is not None:
            callback(sim[0])
        iterations += 1
        if retall:
            allvecs.append(sim[0])

    x = sim[0]
    fval = numpy.min(fsim)
    warnflag = 0

    if fcalls[0] >= maxfun:
        warnflag = 1
        msg = _status_message['maxfev']
        if disp:
            #print('Warning: ' + msg)
    elif iterations >= maxiter:
        warnflag = 2
        msg = _status_message['maxiter']
        if disp:
            #print('Warning: ' + msg)
    else:
        msg = _status_message['success']
        if disp:
            #print(msg)
            #print("         Current function value: %f" % fval)
            #print("         Iterations: %d" % iterations)
            #print("         Function evaluations: %d" % fcalls[0])

    result = sp_opt.OptimizeResult(fun=fval, nit=iterations, nfev=fcalls[0],
                            status=warnflag, success=(warnflag == 0),
                            message=msg, x=x, final_simplex=(sim, fsim))
    if retall:
        result['allvecs'] = allvecs
    return result