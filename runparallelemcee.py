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

#JOBNAME_SALT2mu = "SALT2mu.exe"   # public default code
JOBNAME_SALT2mu = "/project2/rkessler/SURVEYS/SDSS/USERS/BAP37/SNANA_PRIV_INSTALL/SNANA/bin/SALT2mu.exe"  # RK debug

inp_params = ['c', 'RV']

resume = False
debug = False
if '--debug' in sys.argv:
    debug = True
if '--resume' in sys.argv:
    resume = True

os.environ["OMP_NUM_THREADS"] = "1" #This is important for parallelization in emcee

####################################################################################################################
################################## DICTIONARY ######################################################################
####################################################################################################################

#paramdict is hard coded to take the input parameters and expand into the necessary variables to properly model those 
paramdict = {'c':['c_m', 'c_std'], 'x1':['x1_m', 'x1_l', 'x1_r'], 'EBV':['EBV_Tau_low','EBV_Tau_high'], 'RV':['RV_m_low','RV_l_low','RV_r_low','RV_m_high','RV_l_high','RV_r_high']}

#cleandict is ironically named at this point as it's gotten more and more unwieldy. It is designed to contain the following:
#first entry is starting mean value for walkers. Second is the walker std. Third is whether or not the value needs to be positive (eg stds). Fourth is a list containing the lower and upper valid bounds for that parameter.
cleandict = {'c_m':[-0.03, 0.03, False, [-0.3,0.3]], 'c_l':[0.044, 0.03, True, [0.01,0.2]], 'c_r':[0.044, 0.03, True, [0.01,0.2]], 'c_std':[0.044, 0.03, True, [0.01, 0.2]],
             'x1_m':[0,0.05, False, [-2,2]], 'x1_l':[1,1, True, [0.01,2]], 'x1_r':[1,1, True, [0.01,2]], 
             'EBV_Tau_low':[0.01, 0.01, True, [0.01, 0.2]], 'EBV_Tau_high':[0.01, 0.01, True, [0.001, 0.2]],
             'RV_m_low':[2, 0.5, True, [0.8, 4]], 'RV_l_low':[1, 0.5, True, [0.1, 4]], 'RV_r_low':[1, 0.5, True, [0.1, 4]],
             'RV_m_high':[2, 0.5, True, [0.8, 4]], 'RV_l_high':[1, 0.5, True, [0.1, 4]], 'RV_r_high':[1, 0.5, True, [0.1, 4]]}


simdic = {'c':'SIM_c', 'x1':'SIM_x1', "HOST_LOGMASS":"HOST_LOGMASS", 'RV':'SIM_RV', 'EBV':'SIM_EBV'} #converts inp_param into SALT2mu readable format 
arrdic = {'c':np.arange(-.5,.5,.001), 'x1':np.arange(-5,5,.01), 'RV':np.arange(0,8,0.1), 'EBV':np.arange(0.0,1,.02)} #arrays.

####################################################################################################################        
################################## MISCELLANY ######################################################################             
####################################################################################################################  

def thetaconverter(theta): #takes in theta and returns a dictionary of what cuts to make when reading/writing theta
    thetadict = {}
    extparams = pconv(inp_params) #expanded list of all variables. len is ndim.
    for p in inp_params:
        thetalist = []
        for n,ep in enumerate(extparams):
            if p in ep: #for instance, if 'c' is in 'c_l', then this records that position. 
                thetalist.append(n) 
        thetadict[p] = thetalist 
    return thetadict #key gives location of relevant parameters in extparams 

def thetawriter(theta, key): #this does the splitting that thetaconverter sets up. Used in log_likelihood 
    thetadict = thetaconverter(theta)
    lowbound = thetadict[key][0]
    highbound = thetadict[key][-1]+1
    return (theta[lowbound:highbound])


def input_cleaner(inp_params, cleandict): #this function takes in the input parameters and generates the walkers with appropriate dimensions, starting points, walkers, and step size 
    plist = pconv(inp_params)
    pos = np.abs(0.1 * np.random.randn(len(plist)*2, len(plist)))
    for entry in range(len(plist)):
        newpos_param = cleandict[plist[entry]]
        pos[:,entry] = np.random.normal(newpos_param[0], newpos_param[1], len(pos[:,entry]))
        if newpos_param[2]: 
            pos[:,entry] = np.abs(pos[:,entry])
    return pos, len(plist)*2, len(plist)
    

def pconv(inp_params): #converts simple input list of parameters into the expanded list that characterises the sample 
    inpfull = []
    for i in inp_params:
        inpfull.append(paramdict[i])
    inpfull = [item for sublist in inpfull for item in sublist]
    return inpfull

def xconv(inp_param): #gnarly but needed for plotting 
    if inp_param == 'c':
        return np.linspace(-.3,.3,12)
    elif inp_param == 'x1':
        return np.linspace(-3,3,12)

def dffixer(df, RET):
    cpops = []  
    rmspops = []
    
    dflow = df.loc[df.ibin_HOST_LOGMASS == 0]
    dfhigh = df.loc[df.ibin_HOST_LOGMASS == 1]
    
    lowNEVT = dflow.NEVT.values
    highNEVT = dfhigh.NEVT.values
    lowrespops = dflow.MURES_SUM_WGT.values
    highrespops = dfhigh.MURES_SUM_WGT.values
    lowRMS = dflow.MURES_SQSUM.values
    highRMS = dfhigh.MURES_SQSUM.values
    
    
    for q in np.unique(df.ibin_c.values):                                       
        #print(q, np.sum(df.loc[df.ibin_c == q].NEVT))    
        cpops.append(np.sum(df.loc[df.ibin_c == q].NEVT))
        
    cpops = np.array(cpops)
    
    lowRMS = np.sqrt(lowRMS/lowNEVT - ((dflow.MURES_SUM/lowNEVT)**2))
    highRMS = np.sqrt(highRMS/highNEVT - ((dfhigh.MURES_SUM/highNEVT)**2))
    
    if RET == 'HIST':         
        return cpops 
    elif RET == 'ANALYSIS':
        return (cpops), (highrespops/dfhigh.SUM_WGT.values), (lowrespops/dflow.SUM_WGT.values), (highRMS), (lowRMS)
    else:
        return 'No output'


def LL_Creator(inparr, simbeta, returnall_2=False): #takes a list of arrays - eg [[data_1, sim_1],[data_2, sim_2]] and gives an LL. 
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
    LL_Beta = -0.5 * ((realbeta - simbeta) ** 2 / realbetaerr**2)    

    if not returnall_2:
        return np.sum(LL_list)+LL_Beta
    else:
        return np.sum(LL_list)+LL_Beta, datacount_list, simcount_list, poisson_list, ww_list

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
        plt.savefig(f'figures/overplot_observed_DATA_SIM_{inp_params[th]}.png')                            
        print(f'upload figures/overplot_observed_DATA_SIM_{inp_params[th]}.png')                              
        plt.close() 
    
    return 'Done'


#####################################################################################################################
############################# CONNECTION STUFF ######################################################################
#####################################################################################################################

#why is data being regenerated each time?
def init_connection(index,real=True,debug=False):
    #Creates an open connection instance with SALT2mu.exe

    realdataout = 'parallel/%d_SUBPROCESS_REALDATA_OUT.DAT'%index; Path(realdataout).touch()
    simdataout = 'parallel/%d_SUBROCESS_SIM_OUT.DAT'%index; Path(simdataout).touch()
    mapsout = 'parallel/%d_PYTHONCROSSTALK_OUT.DAT'%index; Path(mapsout).touch()
    subprocess_log_data = 'parallel/%d_SUBPROCESS_LOG_DATA.STDOUT'%index; Path(subprocess_log_data).touch()
    subprocess_log_sim = 'parallel/%d_SUBPROCESS_LOG_SIM.STDOUT'%index; Path(subprocess_log_sim).touch()

    arg_outtable = f"\'x1(12,-3:3)*c(12,-0.3:0.3)\'"
#    arg_outtable = f"screm"
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
#    c,cl,cr,x,xl,xr = theta #will need to fix this up as well, as to not have it hard coded.
    #c,cl,cr = theta
    thetadict = thetaconverter(theta)
    try:
        if connection == False: #For MCMC running, will pick up a connection
            sys.stdout.flush()
            connection = connections[(current_process()._identity[0]-1)]#formerly connections[(current_process()._identity[0]-1) % len(connections)] 
            print('DEBUG!', os.getpid(), 'PID', (current_process()._identity[0]-1), 'Connection')
            sys.stdout.flush()

        connection = connection_prepare(connection) #cycle iteration, open SOMETHING.DAT
        print('writing 1d pdf',flush=True)
        #connection.write_1D_PDF('SIM_c',[c,cl,cr],carr) #This writes SOMETHING.DAT
        #connection.write_1D_PDF('SIM_x1',[x,xl,xr],xarr) #THIS IS WHERE TO ADD MORE PARAMETERS
        for inp in inp_params: #TODO - need to generalise to 2d functions as well
            if 'RV' in inp:
                connection.write_2D_Mass_PDF(simdic[inp], thetawriter(theta, inp), arrdic[inp])
                #connection.write_2D_PDF(simdic[inp], thetawriter(theta, inp), arrdic[inp])
            elif 'EBV' in inp:
                connection.write_2D_MassEBV_PDF(simdic[inp], thetawriter(theta, inp), arrdic[inp])
            else:
                print(inp)
                connection.write_1D_PDF(simdic[inp], thetawriter(theta, inp), arrdic[inp]) 
        print('next',flush=True)
        connection = connection_next(connection)# NOW RUN SALT2mu with these new distributions 
        print('got result', flush=True)
        try:
            if np.isnan(connection.beta):
                print('WARNING! oops negative infinity!')
                newcon = (current_process()._identity[0]-1) #% see above at original connection generator, this has been changed  
                tc = init_connection(newcon,real=False)[1]                      
                connections[newcon] = tc 
                return -np.inf
        except AttributeError:
            print("WARNING! We tripped an AttributeError here.")
            if debug:
                tc = init_connection(0,real=False,debug=debug)[1] #set this back to debug=debug                       
                connections[0] = tc
                tempin = np.abs(random.choice([1, .1]) * np.random.randn(nwalkers, ndim))[0]             
                print('inp', tempin)                 
                return -np.inf
            else:
                newcon = (current_process()._identity[0]-1) #% see above at original connection generator, this has been changed 
                tc = init_connection(newcon,real=False)[1]                            
                connections[newcon] = tc   
                return -np.inf
        #ANALYSIS returns c, highres, lowres, rms
        try: #TODO - need to generalise for more parameters and add more options than just HIST 
            bindf = connection.bindf #THIS IS THE PANDAS DATAFRAME OF THE OUTPUT FROM SALT2mu
            bindf = bindf.dropna()
            sim_vals = dffixer(bindf, 'ANALYSIS')
            realbindf = realdata.bindf #same for the real data (was a global variable)
            realbindf = realbindf.dropna()
            real_vals = dffixer(realbindf, 'ANALYSIS')
            resparr = []
            for lin in range(len(real_vals)):
                resparr.append([real_vals[lin],sim_vals[lin]])
            #resparr = [[real_c, sim_c],[real_hires, sim_hires], [real_lores, sim_lores], [real_rms, sim_rms]] #Need to generate this programmatically. 
            #print('DEBUG!', resparr)
        #I don't love this recasting, it's a memory hog, but it's temporary.
        except:
            print('WARNING! something went wrong in reading in stuff for the LL calc')
            return -np.inf

    except BrokenPipeError:
        if debug:
            print('WARNING! we landed in a Broken Pipe error')
            tc = init_connection(0,real=False,debug=debug)[1] #set this back to debug=debug                                
            tempin = np.abs(random.choice([0.001, .1]) * np.random.randn(nwalkers, ndim))[0]            
            print('inp', tempin) 
            return log_likelihood(tempin, connection=tc,returnall=True)
        else:
            print('WARNING! Slurm Broken Pipe Error!') #REGENERATE THE CONNECTION 
            print('before regenerating')
            newcon = (current_process()._identity[0]-1) #% see above at original connection generator, this has been changed 
            tc = init_connection(newcon,real=False)[1]
            connections[newcon] = tc
            return log_likelihood(theta)
    
    sys.stdout.flush()
    if returnall:
        out_result = LL_Creator(resparr, connection.beta)  
        print("for", theta, "we found an LL of", out_result)            
        sys.stdout.flush()
        return LL_Creator(resparr, connection.beta, returnall)
    else:
        out_result = LL_Creator(resparr, connection.beta)
        print("for", theta, "we found an LL of", out_result)
        sys.stdout.flush()
        return out_result

###########################################################################################################################

#def log_prior(theta): #I need to rewrite this to be less awful
#    c,cl,cr,x,xl,xr = theta
    #c,cl,cr = theta
#    if -0.15 < c <= 0.1 and 0.01 < cl < 0.2 and 0.01 < cr < 0.2 and -2 < x <= 2 and 0.01 < xl < 3 and 0.01 < xr < 3:
#        return 0.0
#    elif -2 < x <= 2 and 0.0 < xl < 3 and 0.0 < xr < 3:
#        return 0.0
#    else:
#        return -np.inf

def logprior(theta): #goes through expanded input parameters and checks that they are all within range. If any are not, returns negative infinity.
    thetadict = thetaconverter(theta) 
    tlist = False #if all parameters are good, this remains false 
    for key in thetadict.keys(): 
        temp_ps = (thetawriter(theta, key)) #I hate this but it works. Creates expanded list for this parameter
        for t in range(len(temp_ps)): #then goes through
            lowb = cleandict[plist[t]][3][0] 
            highb = cleandict[plist[t]][3][1]
            if  not lowb < temp_ps[t] < highb: # and compares to valid boundaries.
                tlist = True
    if tlist:
        return -np.inf
    else:
        return 0


def log_probability(theta):
    #print(psutil.virtual_memory().available * 100 / psutil.virtual_memory().total, "is the percentage of memory used")
    lp = log_prior(theta)
    if not np.isfinite(lp):
        print('WARNING! We returned -inf')
        sys.stdout.flush()
        return -np.inf
    else:
        #print('we did successfully call log_likelihood')
        sys.stdout.flush()
        return lp + log_likelihood(theta)



###################################################################################################
###################################################################################################
######## This is where things start running #######################################################
###################################################################################################


##### INITIALIZE PARAMETERS (just 3 color parameters so far) ########## 
#Need to generalise this to n parameters and 2n walkers or what-have-you

#ndim = len(pconv(inp_params))
#nwalkers = ndim*2

pos, nwalkers, ndim = input_cleaner(inp_params, cleandict)

if resume == True: #This needs to be generalised to more than just a 6x3 grid, but it works for now
    past_results = np.load("samples.npz")
    pos[:,0] = np.random.normal(past_results.f.arr_0[-1,:,0], 0.05) #mean
    pos[:,1] = np.random.normal(past_results.f.arr_0[-1,:,1], 0.05) #cl
    pos[:,2] = np.random.normal(past_results.f.arr_0[-1,:,2], 0.05) #cr
    pos[:,3] = np.random.normal(past_results.f.arr_0[-1,:,3], 0.05) #mean x1     
    pos[:,4] = np.random.normal(past_results.f.arr_0[-1,:,4], 0.05) #xl                                                        
    pos[:,5] = np.random.normal(past_results.f.arr_0[-1,:,5], 0.05) #xr  

print(pos)

#######################################################################
##### Run SALT2mu on the real data once! ################
realdata, _ = init_connection(0,debug=debug)
#########################################################
realbeta = realdata.beta
realbetaerr = realdata.betaerr

#### Open a bunch of parallel connections to SALT2mu ### 
connections = []
if debug:
    print('we are in debug mode now')
    nwalkers = 1
else: 
    pass

nconn = nwalkers

for i in range(int(nconn)): #set this back to nwalkers*2 at some point
    print('generated', i, 'walker.')
    sys.stdout.flush()
    tc = init_connection(i,real=False,debug=debug)[1] #set this back to debug=debug
    connections.append(tc)

########################################################

if debug: ##### --debug
    import random
    #just run once through the likelihood with some parameters
    for i in range(200):
        #connections = [] 
        #tc = init_connection(i,real=False,debug=debug)[1] #set this back to debug=debug                 
        #connections.append(tc) 
        #print(log_probability((-.1, 0.01, 0.17)))
        tempin = np.abs(random.choice([0.1, .1]) * np.random.randn(nwalkers, ndim))[0]
        print('inp', tempin)
        print(log_likelihood(tempin, connection=connections[-1],returnall=True))
    abort #deliberately switched the 0.11 (stdr) and 0.03 (stdl) to get wrong value


#filename = "tutorial.h5"
#backend = emcee.backends.HDFBackend(filename)
##backend.reset(nwalkers, ndim)

with Pool(nconn) as pool: #this is where Brodie comes in to get mc running in parallel on batch jobs. 
    #Instantiate the sampler once (in parallel)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)

    for qb in range(1):
        print("Starting loop iteration", qb)
        print('begun', cpu_count(), "CPUs with", nwalkers, ndim, "walkers and dimensions")
        #print(psutil.virtual_memory().available * 100 / psutil.virtual_memory().total, "is the percentage of memory used")
        sys.stdout.flush()
        #Run the sampler
        starttime = time.time()
        sampler.run_mcmc(pos, 500, progress=True) #There used to be a semicolon here for some reason 
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
