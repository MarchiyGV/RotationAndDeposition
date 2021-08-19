# first line: 270
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
