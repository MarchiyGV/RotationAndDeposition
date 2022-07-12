import numpy as np

'''
B-spline N(x) in format: [[a0, b0, k0_0, k0_1, ...], [a1, b1, k1_0, k1_1, ...], ...]
where N(x) = sum_j ki_j * x^j within interval [ai, bi] 
'''
    
def _add_splines(c, b1, d, b2):
    """Construct c*b1 + d*b2."""
    c = np.array(c)
    d = np.array(d)
    b1 = np.array(b1)
    b2 = np.array(b2)
    if (not np.any(c)) or (not np.any(b1)):
        rr = np.tensordot(d, b2[:, 2:], axes=0)
        r = np.zeros((b2.shape[0], d.shape[0]+b2.shape[1]-2))
        for i in range(rr.shape[0]):
            for j in range(rr.shape[2]):
                r[:, i+j] += rr[i, :, j]      
        rv = np.concatenate((b2[:, :2], r), axis=1)
    elif (not np.any(d)) or (not np.any(b2)):
        rr = np.tensordot(c, b1[:, 2:], axes=0)
        r = np.zeros((b1.shape[0], c.shape[0]+b1.shape[1]-2))
        for i in range(rr.shape[0]):
            for j in range(rr.shape[2]):
                r[:, i+j] += rr[i, :, j]
        rv = np.concatenate((b1[:, :2], r), axis=1)
    else:
        result = []
        # Just combining the Piecewise without any fancy optimization
        rr = np.tensordot(c, b1[:, 2:], axes=0)
        r = np.zeros((b1.shape[0], c.shape[0]+b1.shape[1]-2))
        for i in range(rr.shape[0]):
            for j in range(rr.shape[2]):
                r[:, i+j] += rr[i, :, j]
        p1 = np.concatenate((b1[:, :2], r), axis=1)
        p1 = p1[np.argsort(p1[:,0])]
        #print('p1', p1)
        
        rr = np.tensordot(d, b2[:, 2:], axes=0)
        r = np.zeros((b2.shape[0], d.shape[0]+b2.shape[1]-2))
        for i in range(rr.shape[0]):
            for j in range(rr.shape[2]):
                r[:, i+j] += rr[i, :, j]
        p2 = np.concatenate((b2[:, :2], r), axis=1)
        p2 = p2[np.argsort(p2[:,0])]
        #print('p2', p2)
        
        p2args = p2[:, :]
        # This merging algorithm assumes the conditions in
        # p1 and p2 are sorted
        for i in range(p1.shape[0]):
            expr = p1[i, 2:]
            cond = p1[i, :2]

            lower = cond[0]
            # Check p2 for matching conditions that can be merged
            for j in range(len(p2args)):
                if np.any(p2args[j]):
                    expr2 = p2args[j, 2:]
                    cond2 = p2args[j, :2]
    
                    lower_2, upper_2 = cond2
                    if (cond2[0] == cond[0]) and (cond2[1] == cond[1]):
                        # Conditions match, join expressions
                        expr += expr2
                        # Remove matching element
                        p2args[j] = np.zeros((p2args.shape[1]))
                        # No need to check the rest
                        break
                    elif lower_2 < lower and upper_2 <= lower:
                        # Check if arg2 condition smaller than arg1,
                        # add to new_args by itself (no match expected
                        # in p1)
                        result.append(p2args[j, :])
                        p2args[j] = np.zeros((p2args.shape[1]))
                        break

            # Checked all, add expr and cond
            result.append(np.concatenate((cond, expr)))

        # Add remaining items from p2args
        for arg in p2args:
            if np.any(arg):
                result.append(arg)

        rv = np.array(result)
    
    rv = rv[np.argsort(rv[:,0])]
    return rv

def bspline_basis(d, knots, n):
    return _bspline_basis(d, d, knots, n)
    
def _bspline_basis(d, D, knots, n):
    """
    The n-th B-spline at x of degree d with knots.
    """
    d = int(d)
    D = int(D)
    n = int(n)
    n_knots = len(knots)
    n_intervals = n_knots - 1
    if n + d + 1 > n_intervals:
        raise ValueError("n + d + 1 must not exceed len(knots) - 1")
    if d == 0:
        coeffs = np.zeros((D+1))
        coeffs[0] += 1
        result = np.array([np.concatenate(([knots[n], knots[n+1]], coeffs))])
    elif d > 0:
        denom = knots[n + d + 1] - knots[n + 1]
        if denom != 0:
            B = np.array([knots[n + d + 1], -1]) / denom
            b2 = _bspline_basis(d - 1, D, knots, n + 1)
        else:
            b2 = np.array([np.concatenate(([0, 0], np.zeros((D+1))))])
            B = np.zeros((D+1))

        denom = knots[n + d] - knots[n]
        if denom != 0:
            A = np.array([- knots[n], 1]) / denom
            b1 = _bspline_basis(d - 1, D, knots, n)
        else:
            b1 = np.concatenate(([0, 0], np.zeros((D+1))))
            A = np.zeros((D+1))
        
        #print('add1  ', b1, A)
        #print('add2  ', b2, B)
        result = _add_splines(A, b1, B, b2)
        #print('result', result)
    else:
        raise ValueError("degree must be non-negative: %r" % n)

    return result[:, :(2+D+1)]


def bspline_basis_set(d, knots):
    """
    Return the ``len(knots)-d-1`` B-splines at *x* of degree *d*
    with *knots*.
    """
    n_splines = len(knots) - d - 1
    return [bspline_basis(d, tuple(knots), i) for i in range(n_splines)]

