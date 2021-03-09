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
from multiprocessing import cpu_count
import emcee
import time 

resume = False
debug = False
if '--debug' in sys.argv:
    debug = True
if '--resume' in sys.argv:
    resume = True

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
                'SALT2mu.exe SALT2mu_BS20_REALDATA_ALL_nolowz.input SUBPROCESS_FILES=%s,%s,%s','NOTHING.DAT',
                realdataout,subprocess_log_data,realdata=True,debug=True)
        else:
            realdata = 0

        connection = callSALT2mu.SALT2mu(
            'SALT2mu.exe SALT2mu_DESNN_SIM_nolowz.input SUBPROCESS_FILES=%s,%s,%s SUBPROCESS_VARNAMES_GENPDF=SIM_x1,HOST_LOGMASS,SIM_c,HOST_LOGMASS',
            mapsout,simdataout,subprocess_log_sim,debug=True)
    else:
        if real: #Will always run SALT2mu on real data the first time through. Redoes RUNTEST_SUBPROCESS_BS20DATA 
            realdata = callSALT2mu.SALT2mu(
                'SALT2mu.exe SALT2mu_BS20_REALDATA_ALL_nolowz.input SUBPROCESS_FILES=%s,%s,%s','NOTHING.DAT',
                realdataout,subprocess_log_data,realdata=True)
        else:
            realdata = 0
#Then calls it on the simulation. Redoes RUNTEST_SUBPROCESS_SIM, essentially.
        connection = callSALT2mu.SALT2mu( 
            'SALT2mu.exe SALT2mu_DESNN_SIM_nolowz.input SUBPROCESS_FILES=%s,%s,%s SUBPROCESS_VARNAMES_GENPDF=SIM_x1,HOST_LOGMASS,SIM_c,HOST_LOGMASS',
            mapsout,simdataout,subprocess_log_sim)


    if not real: #connection is an object that is equal to SUBPROCESS_SIM/DATA
        connection.getResult() #Gets result, as it were

    return realdata, connection

def connection_prepare(connection): #probably works. Iteration issues, needs to line up with SALT2mu and such. 
    connection.iter+=1 #tick up iteration by one 
    connection.write_iterbegin() #open SOMETHING.DAT for that iteration
    return connection

def connection_next(connection): #Happens at the end of each iteration. 
    connection.write_iterend()
    print('wrote end')
    connection.next()
    print('submitted next iter')
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
    poisson[datacount == 0] = 1 
    poisson[~np.isfinite(poisson)] = 1

    return datacount[ww],simcount[ww],poisson[ww],ww


carr = np.arange(-.5,.5,.001)#these are fixed so can be defined globally
xarr = np.arange(-5,5,.01)

##########################################################################################################################
def log_likelihood(theta,connection=False,returnall=False,pid=0):
    print('inside loglike')
    #c,cl,cr,x,xl,xr = theta
    c,cl,cr = theta
    try:
        if connection == False: #For MCMC running, will pick up a connection
            connection = connections[(current_process()._identity[0]-1) % len(connections)] 
            print('here1')

        connection = connection_prepare(connection) #cycle iteration, open SOMETHING.DAT
        print('writing 1d pdf')
        connection.write_1D_PDF('SIM_c',c,cl,cr,carr) #This writes SOMETHING.DAT
        #connection.write_1D_PDF('SIM_x1',x,xl,xr,xarr) #THIS IS WHERE TO ADD MORE PARAMETERS
        print('next')
        connection = connection_next(connection)# NOW RUN SALT2mu with these new distributions 
        print('got result')
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
    print("for", theta, "we found an LL of", -0.5 * np.sum((datacount - simcount) ** 2 / poisson**2))
    sys.stdout.flush()
    if returnall:
         return -0.5 * np.sum((datacount - simcount) ** 2 / poisson**2),datacount,simcount,cbindf['c'][ww],poisson

    return -0.5 * np.sum((datacount - simcount) ** 2 / poisson**2)
###########################################################################################################################

def log_prior(theta):
    c,cl,cr = theta
    if -0.15 < c <= 0.1 and 0.0 < cl < 0.2 and 0.0 < cr < 0.2:
        return 0.0
    else:
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
pos = np.abs(0.001 * np.random.randn(6, 3))
pos[:,0]+= -0.03
pos[:,1]+= .044
pos[:,2]+= .044
nwalkers, ndim = pos.shape

if resume == True: #This needs to be generalised to more than just a 6x3 grid, but it works for now
    past_results = np.load("samples.npz")
    pos[:,0] = np.random.normal(past_results.f.arr_0[-1,:,0], 0.05) #mean
    pos[:,1] = np.random.normal(past_results.f.arr_0[-1,:,1], 0.05) #cl
    pos[:,2] = np.random.normal(past_results.f.arr_0[-1,:,2], 0.05) #cr

#######################################################################
##### Run SALT2mu on the real data once! ################
realdata, _ = init_connection(0,debug=debug)
#########################################################

#### Open a bunch of parallel connections to SALT2mu ### 
print(nwalkers)
connections = []
if debug: nwalkers = .5
else: pass
for i in range(int(nwalkers*2)): #set this back to nwalkers*2 at some point
    tc = init_connection(i,real=False,debug=debug)[1]
    connections.append(tc)
########################################################

if debug: ##### --debug
    #just run once through the likelihood with some parameters
    print(log_likelihood((-.062,.11,.03),connection=connections[-1],returnall=True))
    abort #deliberately switched the 0.11 (stdr) and 0.03 (stdl) to get wrong value

with Pool() as pool: #this is where Brodie comes in to get mc running in parallel on batch jobs. 
    #Instantiate the sampler once (in parallel)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)

    for qb in range(200):
        print("Starting loop iteration", qb)
        print('begun', cpu_count(), "CPUs")
        sys.stdout.flush()
        #Run the sampler
        starttime = time.time()
        sampler.run_mcmc(pos, 200, progress=True) #There used to be a semicolon here for some reason 
        #May need to implement a proper burn-in 
        endtime = time.time()
        multi_time = endtime - starttime 
        print("Multiprocessing took {0:.1f} seconds".format(multi_time))
        sys.stdout.flush() 

        #Save the output for later
        samples = sampler.get_chain()
        pos = samples[-1,:,:]
        np.savez('samples.npz',samples)
        print(pos, "New samples as input for next run")

        ### THATS IT! (below is just plotting to monitor the fitting) ##########
        ########################################################################
        ########################################################################



        plt.clf()
        fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
        samples = sampler.get_chain()
        labels = ["cmean", "csigma-", "csigma+"]
        for it in range(ndim):
            ax = axes[it]
            ax.plot(samples[:, :, it], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[it])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number");
        plt.savefig('figures/chains.png')
        print('upload figures/chains.png')
        plt.close()

        flat_samples = sampler.get_chain(discard=10, flat=True)

        plt.clf()
        fig = corner.corner(
            flat_samples, labels=labels
        );
        plt.savefig('figures/corner.png')
        print('upload figures/corner.png')
        plt.close()

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
        plt.close()

        plt.clf()
        plt.errorbar(realdata.cbindf['c'],realdata.cbindf['NEVT'],yerr=np.sqrt(realdata.cbindf['NEVT']),fmt='o',c='k',label='REAL DATA')   
        theta = np.mean(flat_samples[-5:,:],axis=0)
        tc = init_connection(i*100,real=False,debug=True)[1]
        chisq,data,simcount,simc,err = log_likelihood((theta[0],theta[1],theta[2]),returnall=True,connection=tc)
        plt.plot(simc,simcount,c='darkgreen',label='SIMULATION')
        plt.xlim(-.5,.5)
        plt.legend()
        plt.xlabel("Observed Color")
        plt.savefig('figures/overplot_observed_DATA_SIM.png')
        print('upload figures/overplot_observed_DATA_SIM.png')
        plt.close()
