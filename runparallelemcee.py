##### Notable Updates ###################
# August 12 2020: Brout - First Commit 
#
#
#

import callSALT2mu
import numpy as np
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import pylab as plt
import os
import corner
from multiprocessing import Pool
from multiprocessing import current_process
import emcee


debug = False
if '--debug' in sys.argv:
    debug = True

os.environ["OMP_NUM_THREADS"] = "1" #This is important for parallelization in emcee

#####################################################################################################################
############################# CONNECTION STUFF ######################################################################
#####################################################################################################################
def init_connection(index,real=True,debug=False):
    #Creates an open connection instance with SALT2mu.exe

    realdataout = 'parallel/%d_SUBPROCESS_REALDATA_OUT.DAT'%index; Path(realdataout).touch()
    simdataout = 'parallel/%d_SUBROCESS_SIM_OUT.DAT'%index; Path(simdataout).touch()
    mapsout = 'parallel/%d_PYTHONCROSSTALK_OUT.DAT'%index; Path(mapsout).touch()
    subprocess_log_data = 'parallel/%d_SUBPROCESS_LOG_DATA.STDOUT'%index; Path(subprocess_log_data).touch()
    subprocess_log_sim = 'parallel/%d_SUBPROCESS_LOG_SIM.STDOUT'%index; Path(subprocess_log_sim).touch()

    if debug:
        if real: 
            realdata = callSALT2mu.SALT2mu(
                '/project2/rkessler/PRODUCTS/SNANA_DEBUG/SNANA/bin/SALT2mu.exe SALT2mu_BS20_REALDATA_ALL_nolowz.input SUBPROCESS_FILES=%s,%s,%s','NOTHING.DAT',
                realdataout,subprocess_log_data,realdata=True,debug=True)
        else:
            realdata = -9

        connection = callSALT2mu.SALT2mu(
            '/project2/rkessler/PRODUCTS/SNANA_DEBUG/SNANA/bin/SALT2mu.exe SALT2mu_DESNN_SIM_nolowz.input SUBPROCESS_FILES=%s,%s,%s SUBPROCESS_VARNAMES_GENPDF=SIM_x1,HOST_LOGMASS,SIM_c,HOST_LOGMASS',
            mapsout,simdataout,subprocess_log_sim,debug=True)
    else:
        if real:
            realdata = callSALT2mu.SALT2mu(
                'SALT2mu.exe SALT2mu_BS20_REALDATA_ALL_nolowz.input SUBPROCESS_FILES=%s,%s,%s','NOTHING.DAT',
                realdataout,subprocess_log_data,realdata=True)
        else:
            realdata = -9

        connection = callSALT2mu.SALT2mu(
            'SALT2mu.exe SALT2mu_DESNN_SIM_nolowz.input SUBPROCESS_FILES=%s,%s,%s SUBPROCESS_VARNAMES_GENPDF=SIM_x1,HOST_LOGMASS,SIM_c,HOST_LOGMASS',
            mapsout,simdataout,subprocess_log_sim)


    if not real: 
        connection.getResult()

    return realdata, connection

def connection_prepare(connection):
    connection.iter+=1
    connection.write_iterbegin()
    return connection

def connection_next(connection):
    connection.write_iterend()
    connection.next()
    connection.getResult()
    return connection

#####################################################################################################################
#####################################################################################################################
################################ SCIENCE STARTS HERE ################################################################

def normhisttodata(datacount,simcount):
    #Helper function to 
    #normalize the simulated histogram to the total counts of the data

    datatot = np.sum(datacount)
    simtot = np.sum(simcount)
    simcount = simcount*datatot/simtot

    ww = (datacount > 0) | (simcount>0)

    poisson = np.sqrt(datacount)
    poisson[~np.isfinite(poisson)] = 1

    return datacount[ww],simcount[ww],poisson[ww],ww


carr = np.arange(-.5,.5,.001)#these are fixed so can be defined globally
xarr = np.arange(-5,5,.01)

##########################################################################################################################
def log_likelihood(theta,connection=False,returnall=False,pid=0):
    #c,cl,cr,x,xl,xr = theta
    c,cl,cr = theta
    try:
        if connection == False: #THIS IS FOR DEBUG/STANDALONE MODE
            connection = connections[(current_process()._identity[0]-1) % len(connections)]

        connection = connection_prepare(connection)
        connection.write_1D_PDF('SIM_c',c,cl,cr,carr)
        #connection.write_1D_PDF('SIM_x1',x,xl,xr,xarr) #THIS IS WHERE TO ADD MORE PARAMETERS
        connection = connection_next(connection)# NOW RUN SALT2mu with these new distributions

        if np.isnan(connection.beta):
            return -np.inf

        cbindf = connection.cbindf #THIS IS THE PANDAS DATAFRAME OF THE OUTPUT FROM SALT2mu
        realcbindf = realdata.cbindf #same for the real data (was a global variable)
    
    except BrokenPipeError:
        print('excepted') #REGENERATE THE CONNECTION 
        i = (current_process()._identity[0]-1) % len(connections)
        tc = init_connection(i,real=False)[1]
        connections[i] = tc
        return log_likelihood(theta)
    
    datacount,simcount,poisson,ww = normhisttodata(realcbindf['NEVT'],cbindf['NEVT'])

    if returnall:
         return -0.5 * np.sum((datacount - simcount) ** 2 / poisson**2),datacount,simcount,cbindf['c'][ww],poisson

    return -0.5 * np.sum((datacount - simcount) ** 2 / poisson**2)
###########################################################################################################################

def log_prior(theta):
    c,cl,cr = theta
    if -0.15 < c < 0.1 and 0.0 < cl < 0.2 and 0.0 < cr < 0.2:
        return 0.0
    return -np.inf


def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)




###################################################################################################
###################################################################################################
######## This is where things start running #######################################################
###################################################################################################


##### INITIALIZE PARAMETERS (just 3 color parameters so far) ##########
pos = np.abs(1e-3 * np.random.randn(6, 3))
pos[:,0]+= -0.03
pos[:,1]+= .044
pos[:,2]+= .044
nwalkers, ndim = pos.shape
#######################################################################


##### Run SALT2mu on the real data once! ################
realdata, _ = init_connection(-9,debug=debug)
#########################################################

#### Open a bunch of parallel connections to SALT2mu ### 
connections = []
for i in range(nwalkers*2):
    tc = init_connection(i,real=False,debug=debug)[1]
    connections.append(tc)
########################################################


if debug: ##### --debug
    #just run once through the likelihood with some parameters
    print(log_likelihood((.1,.1,.1),connection=connections[-1]))
    abort

with Pool() as pool:
    #Instantiate the sampler once (in parallel)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)

    for i in range(200):
        print('begun')
        
        #Run the sampler
        sampler.run_mcmc(pos, 20, progress=True);

        #Save the output for later
        samples = sampler.get_chain()
        pos = samples[-1,:,:]
        np.savez('samples.npz',samples=samples)



        ### THATS IT! (below is just plotting to monitor the fitting) ##########
        ########################################################################
        ########################################################################



        plt.clf()
        fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
        samples = sampler.get_chain()
        labels = ["cmean", "csigma-", "csigma+"]
        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number");
        plt.savefig('figures/chains.png')
        print('upload figures/chains.png')


        flat_samples = sampler.get_chain(discard=10, flat=True)

        plt.clf()
        fig = corner.corner(
            flat_samples, labels=labels
        );
        plt.savefig('figures/corner.png')
        print('upload figures/corner.png')


        plt.clf()
        xr = np.arange(-.3,.3,.001)
        inds = np.random.randint(len(flat_samples), size=1000)
        for ind in inds:
            sample = flat_samples[ind,:]
            arr,probs = _.get_1d_asym_gauss(sample[0],sample[1],sample[2],xr)
            probs = probs/np.sum(probs)
            plt.plot(arr,probs,c='gold',alpha=.02)
        plt.plot([],[],c='gold',alpha=.5,label='MCMC Samples')
        plt.plot(xr,_.get_1d_asym_gauss(-.062,.03,.11,xr)[1]/np.sum(_.get_1d_asym_gauss(-.062,.03,.11,xr)[1]),label='SK16 Parent Color',lw=4)
        plt.xlim(-.3,.3)
        plt.ylim(0,.01)
        plt.legend()
        plt.xlabel("Parent Color")
        plt.savefig('figures/overplotmodel.png')
        print('upload figures/overplotmodel.png')


        plt.clf()
        plt.errorbar(realdata.cbindf['c'],realdata.cbindf['NEVT'],yerr=np.sqrt(realdata.cbindf['NEVT']),fmt='o',c='k',label='REAL DATA')   
        len = len(flat_samples)
        theta = np.mean(flat_samples[int(len/2):,:],axis=0)
        tc = init_connection(i*100,real=False,debug=True)[1]
        chisq,data,simcount,simc,err = log_likelihood((theta[0],theta[1],theta[2]),returnall=True,connection=tc)
        plt.plot(simc,simcount,c='darkgreen',label='SIMULATION')
        plt.xlim(-.5,.5)
        plt.legend()
        plt.xlabel("Observed Color")
        plt.savefig('figures/overplot_observed_DATA_SIM.png')
        print('upload figures/overplot_observed_DATA_SIM.png')
