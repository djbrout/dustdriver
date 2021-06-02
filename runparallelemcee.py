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
import psutil

JOBNAME_SALT2mu = "SALT2mu.exe"   # public default code
#JOBNAME_SALT2mu = "/home/rkessler/SNANA/bin/SALT2mu.exe"  # RK debug

inp_params = ['c', 'x1']

resume = False
debug = False
if '--debug' in sys.argv:
    debug = True
if '--resume' in sys.argv:
    resume = True

os.environ["OMP_NUM_THREADS"] = "1" #This is important for parallelization in emcee

####################################################################################################################
################################## MISCELLANY ######################################################################
####################################################################################################################
paramdict = {'c':['c_m', 'c_l', 'c_r'], 'x1':['x_m', 'x_l', 'x_r']}


def pconv(inp_params):
    inpfull = []
    for i in inp_params:
        inpfull.append(paramdict[i])
    inpfull = [item for sublist in inpfull for item in sublist]
    return inpfull

def xconv(inp_param):
    if inp_param == 'c':
        return np.linspace(-.3,.3,12)
    elif inp_param == 'x1':
        return np.linspace(-3,3,12)

def dffixer(df, RET):    
    cpops = []
    xpops =[]

    for q in np.unique(df.ibin_x1.values):
        cpops.append(np.sum(df[df.ibin_x1 == q].NEVT))
    for q in np.unique(df.ibin_c.values):
        xpops.append(np.sum(df[df.ibin_c == q].NEVT))
    if RET == 'HIST':
        return np.array(cpops), np.array(xpops)


def LL_Creator(inparr, returnall_2=False): #takes a list of arrays - eg [[data_1, sim_1],[data_2, sim_2]] and gives an LL. 
    if returnall_2:
        datacount_list = []
        simcount_list = []
        poisson_list = []
        ww_list = []
    LL_list = []
    for i in inparr:
        datacount,simcount,poisson,ww = normhisttodata(i[0], i[1])
        LL_c = -0.5 * np.sum((datacount - simcount) ** 2 / poisson**2) 
        LL_list.append(LL_c)
        if returnall_2:
            datacount_list.append(datacount)
            simcount_list.append(simcount)
            poisson_list.append(poisson)
            ww_list.append(ww)
    LL_list = np.array(LL_list)
    if not returnall_2:
        return np.sum(LL_list)
    else:
        return np.sum(LL_list), datacount_list, simcount_list, poisson_list, ww_list

def pltting_func(sampler, inp_params):
    labels = pconv(inp_params)  
    ndim = len(labels)
    plt.clf()                               
    fig, axes = plt.subplots(ndim, figsize=(3*ndim, 7), sharex=True)                                   
    samples = sampler.get_chain()                                        
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
    #plt.show()
    
    theta = np.mean(flat_samples[-5:,:],axis=0)
    tc = init_connection(i*100,real=False,debug=True)[1]                                        
    chisq, datacount_list, simcount_list, poisson_list, ww_list = log_likelihood((theta),returnall=True,connection=tc)
    
    for th in range(len(inp_params)):
        plt.clf() 
        datax = xconv(inp_params[th])
        plt.errorbar(datax, datacount_list[th] ,yerr=(poisson_list[th]),fmt='o',c='k',label='REAL DATA')   
        plt.plot(datax, simcount_list[th], c='darkgreen',label='SIMULATION')    
        #plt.xlim(-.5,.5)
        plt.legend()           
        plt.xlabel("Observed Variable")                                                                
        plt.savefig(f'figures/overplot_observed_DATA_SIM_{inp_param}.png')                                       
        print(f'upload figures/overplot_observed_DATA_SIM_{inp_param}.png')                              
        plt.close() 
    
    return 'Done'


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

    arg_outtable = f"\'x1(12,-3:3)*c(12,-0.3:0.3)\'"
    if debug:
        if real:
            cmd = f"{JOBNAME_SALT2mu} SALT2mu_BS20_REALDATA_ALL_nolowz.input " \
                  f"SUBPROCESS_FILES=%s,%s,%s " \
                  f"SUBPROCESS_OUTPUT_TABLE={arg_outtable}"
            realdata = callSALT2mu.SALT2mu(cmd, 'NOTHING.DAT', realdataout, 
                                           subprocess_log_data, realdata=True, debug=True)
        else:
            realdata = 0

        cmd = f"{JOBNAME_SALT2mu} SALT2mu_DESNN_SIM_nolowz.input SUBPROCESS_FILES=%s,%s,%s "\
              f"SUBPROCESS_VARNAMES_GENPDF=SIM_x1,HOST_LOGMASS,SIM_c,HOST_LOGMASS " \
              f"SUBPROCESS_OUTPUT_TABLE={arg_outtable}"
        connection = callSALT2mu.SALT2mu(cmd, mapsout,simdataout,subprocess_log_sim, debug=True )

    else:
        if real: #Will always run SALT2mu on real data the first time through. Redoes RUNTEST_SUBPROCESS_BS20DATA 
            cmd = f"{JOBNAME_SALT2mu} SALT2mu_BS20_REALDATA_ALL_nolowz.input " \
                  f"SUBPROCESS_FILES=%s,%s,%s " \
                  f"SUBPROCESS_OUTPUT_TABLE={arg_outtable}"
            realdata = callSALT2mu.SALT2mu(cmd, 'NOTHING.DAT', realdataout,
                                           subprocess_log_data, realdata=True )
        else:
            realdata = 0
            #Then calls it on the simulation. Redoes RUNTEST_SUBPROCESS_SIM, essentially.
        cmd = f"{JOBNAME_SALT2mu} SALT2mu_DESNN_SIM_nolowz.input SUBPROCESS_FILES=%s,%s,%s "\
              f"SUBPROCESS_VARNAMES_GENPDF=SIM_x1,HOST_LOGMASS,SIM_c,HOST_LOGMASS " \
              f"SUBPROCESS_OUTPUT_TABLE={arg_outtable}"
        connection = callSALT2mu.SALT2mu(cmd, mapsout,simdataout,subprocess_log_sim)

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
    datacount = np.array(datacount)
    simcount = np.array(simcount)
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
    print('inside loglike',flush=True)
    c,cl,cr,x,xl,xr = theta #will need to fix this up as well, as to not have it hard coded.
    #c,cl,cr = theta
    try:
        if connection == False: #For MCMC running, will pick up a connection
            connection = connections[(current_process()._identity[0]-1) % len(connections)] 
            print('here1',flush=True)

        connection = connection_prepare(connection) #cycle iteration, open SOMETHING.DAT
        print('writing 1d pdf',flush=True)
        connection.write_1D_PDF('SIM_c',[c,cl,cr],carr) #This writes SOMETHING.DAT
        connection.write_1D_PDF('SIM_x1',[x,xl,xr],xarr) #THIS IS WHERE TO ADD MORE PARAMETERS
        print('next',flush=True)
        connection = connection_next(connection)# NOW RUN SALT2mu with these new distributions 
        print('got result', flush=True)
        try:
            if np.isnan(connection.beta):
                return -np.inf
        except AttributeError:
            print("We tripped an AttributeError here.")
            return -np.inf

        bindf = connection.bindf #THIS IS THE PANDAS DATAFRAME OF THE OUTPUT FROM SALT2mu
        bindf = bindf.dropna()
        sim_c, sim_x = dffixer(bindf, 'HIST')
        realbindf = realdata.bindf #same for the real data (was a global variable)
        realbindf = realbindf.dropna()
        real_c, real_x = dffixer(realbindf, 'HIST')
        resparr = [[real_c, sim_c],[real_x, sim_x]] #Need to generate this programmatically. 
        #I don't love this recasting, it's a memory hog, but it's temporary.
    
    except BrokenPipeError:
        print('excepted') #REGENERATE THE CONNECTION 
        i = (current_process()._identity[0]-1) % len(connections)
        tc = init_connection(i,real=False)[1]
        connections[i] = tc
        return log_likelihood(theta)
    
    sys.stdout.flush()
    #LL_Creator(resparr, returnall)
    #need to convert this into a function defined earlier for best ease of use
    #datacount_c,simcount_c,poisson_c,ww_c = normhisttodata(real_c, sim_c)
    #datacount_x,simcount_x,poisson_x,ww_x = normhisttodata(real_x, sim_x)
    #print("for", theta, "we found an LL of", -0.5 * np.sum((datacount - simcount) ** 2 / poisson**2))
    #sys.stdout.flush()
    #LL_c = -0.5 * np.sum((datacount_c - simcount_c) ** 2 / poisson_c**2)
    #LL_x = -0.5 * np.sum((datacount_x - simcount_x) ** 2 / poisson_x**2)
    #LL = LL_c + LL_x
    if returnall:
        out_result = LL_Creator(resparr)  
        print("for", theta, "we found an LL of", out_result)            
        sys.stdout.flush()
        return LL_Creator(resparr, returnall)
        #return LL,datacount,simcount,bindf['c'][ww],poisson
    else:
        out_result = LL_Creator(resparr)
        print("for", theta, "we found an LL of", out_result)
        sys.stdout.flush()
        return out_result

###########################################################################################################################

def log_prior(theta):
    c,cl,cr,x,xl,xr = theta
    #c,cl,cr = theta
    if -0.15 < c <= 0.1 and 0.0 < cl < 0.2 and 0.0 < cr < 0.2:
        return 0.0
    elif -2 < x <= 2 and 0.0 < xl < 3 and 0.0 < xr < 3:
        return 0.0
    else:
        return -np.inf


def log_probability(theta):
    print(psutil.virtual_memory().available * 100 / psutil.virtual_memory().total, "is the percentage of memory used")
    lp = log_prior(theta)
    if not np.isfinite(lp):
        print('we returned -inf')
        sys.stdout.flush()
        return -np.inf
    else:
        print('we did successfully call log_likelihood')
        sys.stdout.flush()
        return lp + log_likelihood(theta)



###################################################################################################
###################################################################################################
######## This is where things start running #######################################################
###################################################################################################


##### INITIALIZE PARAMETERS (just 3 color parameters so far) ########## 
#Need to generalise this to n parameters and 2n walkers or what-have-you


nwalkers=12
ndim=6

pos = np.abs(0.001 * np.random.randn(nwalkers, ndim))
pos[:,0]+= np.random.normal(-0.03,0.03)
pos[:,1]+= np.abs(np.random.normal(.044,0.03))
pos[:,2]+= np.abs(np.random.normal(.044, 0.03))

pos[:,3]+= np.random.normal(0, 0.5) 
pos[:,4]+= np.abs(np.random.normal(1, 0.5))
pos[:,5]+= np.abs(np.random.normal(0, 0.5))
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
if debug:
    print('we are in debug mode now')
    nwalkers = 1
else: 
    pass

for i in range(int(nwalkers)): #set this back to nwalkers*2 at some point
    print('generated', i, 'walker.')
    sys.stdout.flush()
    tc = init_connection(i,real=False,debug=debug)[1] #set this back to debug=debug
    connections.append(tc)

print(connections)
sys.stdout.flush()
########################################################

if debug: ##### --debug
    #just run once through the likelihood with some parameters
    for i in range(200):
        #print(log_probability((-.1, 0.01, 0.17)))
        print(log_likelihood((-.1, 0.01, 0.17, 0,1,1),connection=connections[-1],returnall=True))

    abort #deliberately switched the 0.11 (stdr) and 0.03 (stdl) to get wrong value


#filename = "tutorial.h5"
#backend = emcee.backends.HDFBackend(filename)
##backend.reset(nwalkers, ndim)

with Pool() as pool: #this is where Brodie comes in to get mc running in parallel on batch jobs. 
    #Instantiate the sampler once (in parallel)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)

    for qb in range(2):
        print("Starting loop iteration", qb)
        print('begun', cpu_count(), "CPUs with", nwalkers, ndim, "walkers and dimensions")
        print(psutil.virtual_memory().available * 100 / psutil.virtual_memory().total, "is the percentage of memory used")
        sys.stdout.flush()
        #Run the sampler
        starttime = time.time()
        sampler.run_mcmc(pos, 150, progress=True) #There used to be a semicolon here for some reason 
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

        pltting_func(sampler, inp_params)

"""
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
        plt.errorbar(realdata.bindf['c'],realdata.bindf['NEVT'],yerr=np.sqrt(realdata.bindf['NEVT']),fmt='o',c='k',label='REAL DATA')   
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
"""
