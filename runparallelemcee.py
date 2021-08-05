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

#################################################################################################################
############################################## INPUT STUFF ######################################################
#################################################################################################################

JOBNAME_SALT2mu = "SALT2mu.exe"   # public default code
#JOBNAME_SALT2mu = "/home/bap37/SNANA/bin/SALT2mu.exe" 

os.environ["OMP_NUM_THREADS"] = "1" #This is important for parallelization in emcee 


#switching between bound and unbound involves editing inp_params, tempin, sim_input, and the init_connection file
inp_params = ['c', 'RV', 'EBVZ', 'beta']
#tempin = [-0.07372071,  0.05470099,  2.14175912,  1.06861188,  1.54413064,  0.39776982, 0.12459312,  0.13753596, 2.0, .35]
tempin = [-0.084,  0.042, 2.75,  1.3,  1.5,  1.3, .13, .13, .13, .21, 2.0, .35] 
#tempin = [-0.07372071,  .02,  2.14175912,  1.06861188,  1.54413064,  0.39776982, 0.12459312,  0.13753596, .12, .14, 2.0, .35]
tempin = [-0.07478121,  0.05038964,  1.88575588,  2.26321848,  1.72849619,  1.13732639, 0.08549067,  0.11739132,  0.1115375,   0.12054095,  2.01261982,  0.31081618]  

ncbins = 6
data_input= f"SALT2mu_ALL_DATA.input"
sim_input = f"SALT2mu_ALL_BOUND.input"
previous_samples = 'chains/SALT2mu_ALL_DATA-samples.npz'

##################################################################################################################
############################################# ARGPARSE ###########################################################
##################################################################################################################

splitEBV = False

if 'EBVZ' in inp_params:
    splitEBV = True

doplot = False 
resume = False
debug = False
stretch = False
wfit = False 
single = False
shotnoise = False 

if '--debug' in sys.argv:
    debug = True
if '--resume' in sys.argv:
    resume = True
if '--stretch' in sys.argv:
    stretch = True
if '--doplot' in sys.argv:
    doplot = True
    debug = True 
if '--wfit' in sys.argv:
    wfit = True
    resume = True
if '--single' in sys.argv:
    single = True
    debug = True
if '--shotnoise' in sys.argv:
    shotnoise = True
    debug = True
    wfit = True
####################################################################################################################
################################## DICTIONARY ######################################################################
####################################################################################################################

#paramdict is hard coded to take the input parameters and expand into the necessary variables to properly model those 
paramdict = {'c':['c_m', 'c_std'], 'x1':['x1_m', 'x1_l', 'x1_r'], 'EBV':['EBV_Tau_low','EBV_Tau_high'], 'RV':['RV_m_low','RV_std_low', 'RV_m_high','RV_std_high'], 'beta':['beta_m','beta_std'], 'EBVZ':['EBVZL_Tau_low','EBVZL_Tau_high', 'EBVZH_Tau_low','EBVZH_Tau_high']}

#cleandict is ironically named at this point as it's gotten more and more unwieldy. It is designed to contain the following:
#first entry is starting mean value for walkers. Second is the walker std. Third is whether or not the value needs to be positive (eg stds). Fourth is a list containing the lower and upper valid bounds for that parameter.
cleandict = {'c_m':[-0.03, 0.03, False, [-0.3,0.3]], 'c_l':[0.044, 0.03, True, [0.01,0.2]], 'c_r':[0.044, 0.03, True, [0.01,0.2]], 'c_std':[0.044, 0.03, True, [0.01, 0.2]],
             'x1_m':[0,0.05, False, [-2,2]], 'x1_l':[1,1, True, [0.01,2]], 'x1_r':[1,1, True, [0.01,2]], 
             'EBV_Tau_low':[0.11, 0.02, True, [0.08, 0.2]], 'EBV_Tau_high':[0.13, 0.02, True, [0.08, 0.2]],
             'RV_m_low':[2, 0.5, True, [0.8, 4]], 'RV_l_low':[1, 0.5, True, [0.1, 4]], 'RV_r_low':[1, 0.5, True, [0.1, 4]], 'RV_std_low':[1, 0.5, True, [0.1,4]],
             'RV_m_high':[2, 0.5, True, [0.8, 4]], 'RV_l_high':[1, 0.5, True, [0.1, 4]], 'RV_r_high':[1, 0.5, True, [0.1, 4]], 'RV_std_high':[1, 0.5, True, [0.1,4]],
             'beta_m':[2, 0.5, False, [1,3]], 'beta_std':[.2, .1, False, [0, 1]],
             'EBVZL_Tau_low':[0.11, 0.02, True, [0.08, 0.2]], 'EBVZL_Tau_high':[0.13, 0.02, True, [0.08, 0.2]],
             'EBVZH_Tau_low':[0.11, 0.02, True, [0.08, 0.2]], 'EBVZH_Tau_high':[0.13, 0.02, True, [0.08, 0.2]],
}


simdic = {'c':'SIM_c', 'x1':'SIM_x1', "HOST_LOGMASS":"HOST_LOGMASS", 'RV':'SIM_RV', 'EBV':'SIM_EBV', 'beta':'SIM_beta', 'EBVZ':'SIM_EBV'} #converts inp_param into SALT2mu readable format 
arrdic = {'c':np.arange(-.5,.5,.001), 'x1':np.arange(-5,5,.01), 'RV':np.arange(0,8,0.1), 'EBV':np.arange(0.0,1.5,.02),
          'EBVZ':np.arange(0.0,1.5,.02)} #arrays.

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

def thetawriter(theta, key, names=False): #this does the splitting that thetaconverter sets up. Used in log_likelihood 
    thetadict = thetaconverter(theta)
    lowbound = thetadict[key][0]
    highbound = thetadict[key][-1]+1
    if names:
        return names[lowbound:highbound]
    else:
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
    lowrespops = dflow.MURES_SUM.values
    highrespops = dfhigh.MURES_SUM.values
    lowRMS = dflow.MURES_SQSUM.values
    highRMS = dfhigh.MURES_SQSUM.values
    
    
    for q in np.unique(df.ibin_c.values):                                       
        #print(q, np.sum(df.loc[df.ibin_c == q].NEVT))    
        cpops.append(np.sum(df.loc[df.ibin_c == q].NEVT))
        
    cpops = np.array(cpops)
    
    lowRMS = np.sqrt(lowRMS/lowNEVT - ((dflow.MURES_SUM.values/lowNEVT)**2))
    highRMS = np.sqrt(highRMS/highNEVT - ((dfhigh.MURES_SUM.values/highNEVT)**2))
    
    if RET == 'HIST':         
        return cpops 
    elif RET == 'ANALYSIS':
        return (cpops), (highrespops/dfhigh.NEVT.values), (lowrespops/dflow.NEVT.values), (highRMS), (lowRMS), (highNEVT), (lowNEVT)
    else:
        return 'No output'

def LL_Creator(inparr, simbeta, simsigint, returnall_2=False): #takes a list of arrays - eg [[data_1, sim_1],[data_2, sim_2]] and gives an LL      
    if returnall_2:     
        datacount_list = []         
        simcount_list = []       
        poisson_list = []    
    LL_list = []
    print('real beta, sim beta, real beta error', realbeta, simbeta, realbetaerr)
    sys.stdout.flush()
    LL_Beta = -0.5 * ((realbeta - simbeta) ** 2 / realbetaerr**2) 
    LL_sigint = -0.5 * ((realsigint - simsigint) ** 2 / realsiginterr**2) * 2
    thetaconverter(inp_params)
    for n, i in enumerate(inparr):    
        if n == 0: #colour
            datacount,simcount,poisson,ww = normhisttodata(i[0], i[1])              
        elif n == 1: #Hi mass MURES
            datacount = i[0]
            simcount  = i[1]
            poisson = inparr[3][0]/np.sqrt(inparr[-2][0]) #uses RMS/sqrt(N) as error
        elif n == 2: #lo mass MURES
            datacount = i[0]
            simcount  = i[1]
            poisson = inparr[4][0]/np.sqrt(inparr[-1][0]) #uses RMS/sqrt(N) as error
        elif (n == 3): #Hi mass RMS
            datacount = i[0]
            simcount = i[1]
            poisson = i[0]/np.sqrt(2*inparr[-2][0]) #The error on the error is sigma/sqrt(2N)
        elif (n == 4): #low mass RMS
            datacount = i[0]    
            simcount = i[1]   
            poisson = i[0]/np.sqrt(2*inparr[-1][0])

        LL_c = -0.5 * np.sum((datacount - simcount) ** 2 / poisson**2)                                                                
        LL_list.append(LL_c)      
        if returnall_2:       
            datacount_list.append(datacount)   
            simcount_list.append(simcount)  
            poisson_list.append(poisson) 

    LL_list = LL_list[:-2]
    LL_list.append(LL_Beta)
    LL_list.append(LL_sigint)
    LL_list = np.array(LL_list) 
    if not returnall_2:        
        return np.nansum(LL_list)                                                                                                       
    else:    
        return (LL_list), datacount_list, simcount_list, poisson_list 

def pltting_func(samples, inp_params):
    labels = pconv(inp_params)  
    ndim = len(labels)
    plt.clf()                               
    fig, axes = plt.subplots(ndim, figsize=(3*ndim, 7), sharex=True) 
    for it in range(ndim): 
        ax = axes[it]      
        ax.plot(samples[:, :, it], "k", alpha=0.3)                                              
        ax.set_xlim(0, len(samples))                                                            
        ax.set_ylabel(labels[it])                                                               
        ax.yaxis.set_label_coords(-0.1, 0.5)                                                    
                               
    axes[-1].set_xlabel("step number");                                                         
    plt.savefig('figures/'+data_input.split('.')[0]+'-chains.png')                                                           
    print('upload figures/chains.png')   
    plt.close()
                               
    flat_samples = samples.reshape(-1, samples.shape[-1])                           

    plt.clf()              
    fig = corner.corner(   
        flat_samples, labels=labels, smooth=True                                                             
    );                     
    plt.savefig('figures/'+data_input.split('.')[0]+'-corner.png')                                              
    print('upload figures/corner.png')                                                          
    plt.close()
    #plt.show()

def Criteria_Plotter(theta):
    tc = init_connection(299,real=False,debug=True)[1]
    try:
        chisq, datacount_list, simcount_list, poisson_list = log_likelihood((theta),returnall=True,connection=tc)
    except TypeError:
        print(f"LL was not returned after running log_likelihood, which is likely due to bad parameters. Will skip plotting.")
        return 
    cbins = np.linspace(-0.2,0.25, ncbins)
    if debug: print('RESULT!', chisq, flush=True)
    sys.stdout.flush()

    ######## Colour Histogram
    plt.clf()
    plt.errorbar(cbins, datacount_list[0], yerr =(poisson_list[0]), fmt='o', c='k', label='REAL DATA')
    plt.plot(cbins, simcount_list[0], c='darkgreen',label='SIMULATION')
    plt.legend()
    plt.xlabel('c')
    plt.ylabel('Count')
    plt.savefig('figures/'+data_input.split('.')[0]+'overplot_observed_DATA_SIM_c.png')         
    print('upload figures/overplot_observed_DATA_SIM_c.png')   
    plt.close() 

    ####### MURES hi and lo
    plt.clf()          
    if debug: print('DEBUG!',len(cbins), len(datacount_list[1]), len(poisson_list[1]))
    sys.stdout.flush()
    plt.errorbar(cbins, datacount_list[1], yerr =(poisson_list[1]), fmt='o', c='k', label='REAL DATA HIMASS')            
    plt.plot(cbins, simcount_list[1], c='tab:orange',label='SIMULATION HIMASS')      

    plt.errorbar(cbins, datacount_list[2], yerr =(poisson_list[1]), fmt='o', c='tab:green', label='REAL DATA LOWMASS')
    plt.plot(cbins, simcount_list[2], c='tab:blue',label='SIMULATION LOWMASS') 
    plt.legend()        
    plt.xlabel('c')   
    plt.ylabel('MURES')   
    plt.savefig('figures/'+data_input.split('.')[0]+'overplot_observed_DATA_SIM_cvMURES.png')   
    print('upload figures/overplot_observed_DATA_SIM_cvMURES.png')        
    plt.close()  

    ####### RMS hi and lo 
    plt.clf()    
    plt.errorbar(cbins, datacount_list[3], yerr =(poisson_list[3]), fmt='o', c='k', label='REAL DATA HIMASS')   
    plt.plot(cbins, simcount_list[3], c='tab:orange',label='SIMULATION HIMASS')  

    plt.errorbar(cbins, datacount_list[4], yerr =(poisson_list[4]), fmt='o', c='tab:green', label='REAL DATA LOWMASS') 
    plt.plot(cbins, simcount_list[4], c='tab:blue',label='SIMULATION LOWMASS')
    plt.legend()     
    plt.xlabel('c')   
    plt.ylabel('RMS')   
    plt.savefig('figures/'+data_input.split('.')[0]+'overplot_observed_DATA_SIM_cvRMS.png')   
    print('upload figures/overplot_observed_DATA_SIM_cvRMS.png')         
    plt.close() 
    
    return 'Done'


#####################################################################################################################
############################# CONNECTION STUFF ######################################################################
#####################################################################################################################

#why is data being regenerated each time?
def init_connection(index,real=True,debug=False):
    #Creates an open connection instance with SALT2mu.exe

    directory = 'parallel'
    if wfit:
        directory = 'M0DIF'

    if wfit:
        realdataout = f'{directory}/%d_SUBPROCESS_REALDATA_OUT.DAT'%index; Path(realdataout).touch()
        simdataout = f'{directory}/%d_SUBROCESS_SIM_OUT.DAT'%index; Path(simdataout).touch()
        mapsout = f'{directory}/%d_PYTHONCROSSTALK_OUT.DAT'%index; Path(mapsout).touch()
        subprocess_log_data = f'{directory}/%d_SUBPROCESS_LOG_DATA.STDOUT'%index; Path(subprocess_log_data).touch()
        subprocess_log_sim = f'{directory}/%d_SUBPROCESS_LOG_SIM.STDOUT'%index; Path(subprocess_log_sim).touch()  
    else:
        realdataout = f'{directory}/%d_SUBPROCESS_REALDATA_OUT.DAT'%index; Path(realdataout).touch()
        simdataout = f'{directory}/%d_SUBROCESS_SIM_OUT.DAT'%index; Path(simdataout).touch()
        mapsout = f'{directory}/%d_PYTHONCROSSTALK_OUT.DAT'%index; Path(mapsout).touch()
        subprocess_log_data = f'{directory}/%d_SUBPROCESS_LOG_DATA.STDOUT'%index; Path(subprocess_log_data).touch()
        subprocess_log_sim = f'{directory}/%d_SUBPROCESS_LOG_SIM.STDOUT'%index; Path(subprocess_log_sim).touch()

    arg_outtable = f"\'c(6,-0.2:0.25)*HOST_LOGMASS(2,0:20)\'"

    #data_input= f"SALT2mu_HIZ_DATA.input"
    #sim_input = f"SALT2mu_HIZ_SIM.input"


#    arg_outtable = f"screm"

    if wfit:
        if real:
            cmd = f"{JOBNAME_SALT2mu} {data_input} " \
                  f"SUBPROCESS_FILES=%s,%s,%s " \
                  f"SUBPROCESS_OUTPUT_TABLE={arg_outtable}" 
            realdata = callSALT2mu.SALT2mu(cmd, 'NOTHING.DAT', realdataout,
                                           subprocess_log_data, realdata=True, debug=True) 
        else: 
            realdata = 0 
        cmd = f"{JOBNAME_SALT2mu} {sim_input} SUBPROCESS_FILES=%s,%s,%s "\
              f"SUBPROCESS_VARNAMES_GENPDF=SIM_x1,HOST_LOGMASS,SIM_c,SIM_RV,SIM_EBV,SIM_ZCMB,SIM_beta " \
              f"SUBPROCESS_OUTPUT_TABLE={arg_outtable} " \
              f"SUBPROCESS_OPTMASK=6"
        connection = callSALT2mu.SALT2mu(cmd, mapsout,simdataout,subprocess_log_sim, debug=True )

    elif debug:
        if real:
            cmd = f"{JOBNAME_SALT2mu} {data_input} " \
                  f"SUBPROCESS_FILES=%s,%s,%s " \
                  f"SUBPROCESS_OUTPUT_TABLE={arg_outtable} " \
                  f"SUBPROCESS_OPTMASK=1" 
            realdata = callSALT2mu.SALT2mu(cmd, 'NOTHING.DAT', realdataout, 
                                           subprocess_log_data, realdata=True, debug=True)
        else:
            realdata = 0

        cmd = f"{JOBNAME_SALT2mu} {sim_input} SUBPROCESS_FILES=%s,%s,%s "\
              f"SUBPROCESS_VARNAMES_GENPDF=SIM_x1,HOST_LOGMASS,SIM_c,SIM_RV,SIM_EBV,SIM_ZCMB,SIM_beta " \
              f"SUBPROCESS_OUTPUT_TABLE={arg_outtable} " \
              f"SUBPROCESS_OPTMASK=1 " \
              f"SUBPROCESS_SIMREF_FILE=/scratch/midway2/rkessler/PIPPIN_OUTPUT/HIGH-REDSHIFT-BOUND/1_SIM/SIMDES_4D_BS20/PIP_HIGH-REDSHIFT-BOUND_SIMDES_4D_BS20.input"
        connection = callSALT2mu.SALT2mu(cmd, mapsout,simdataout,subprocess_log_sim, debug=True )

    else:
        if real: #Will always run SALT2mu on real data the first time through. Redoes RUNTEST_SUBPROCESS_BS20DATA 
            cmd = f"{JOBNAME_SALT2mu} {data_input} " \
                  f"SUBPROCESS_FILES=%s,%s,%s " \
                  f"SUBPROCESS_OUTPUT_TABLE={arg_outtable}"
            realdata = callSALT2mu.SALT2mu(cmd, 'NOTHING.DAT', realdataout,
                                           subprocess_log_data, realdata=True )
        else:
            realdata = 0
            #Then calls it on the simulation. Redoes RUNTEST_SUBPROCESS_SIM, essentially.
        cmd = f"{JOBNAME_SALT2mu} {sim_input} SUBPROCESS_FILES=%s,%s,%s "\
              f"SUBPROCESS_VARNAMES_GENPDF=SIM_x1,HOST_LOGMASS,SIM_c,SIM_RV,SIM_EBV,SIM_ZCMB,SIM_beta " \
              f"SUBPROCESS_OUTPUT_TABLE={arg_outtable} " \
              f"SUBPROCESS_OPTMASK=4 " \
              f"SUBPROCESS_SIMREF_FILE=/scratch/midway2/rkessler/PIPPIN_OUTPUT/HIGH-REDSHIFT-BOUND/1_SIM/SIMDES_4D_BS20/PIP_HIGH-REDSHIFT-BOUND_SIMDES_4D_BS20.input"
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

    ww = (datacount != 0) | (simcount != 0)

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
            #print('DEBUG!', os.getpid(), 'PID', (current_process()._identity[0]-1), 'Connection')
            sys.stdout.flush()

        connection = connection_prepare(connection) #cycle iteration, open SOMETHING.DAT
        print('writing 1d pdf',flush=True)
        #connection.write_1D_PDF('SIM_c',[c,cl,cr],carr) #This writes SOMETHING.DAT
        #connection.write_1D_PDF('SIM_x1',[x,xl,xr],xarr) #THIS IS WHERE TO ADD MORE PARAMETERS
        for inp in inp_params: #TODO - need to generalise to 2d functions as well
            if 'RV' in inp:
                #connection.write_2D_Mass_PDF(simdic[inp], thetawriter(theta, inp), arrdic[inp])
                connection.write_2D_Mass_PDF_SYMMETRIC(simdic[inp], thetawriter(theta, inp), arrdic[inp])
            elif 'EBV' in inp:
                if splitEBV: #need to hold here to pass off both EBVZH and EBVZL simultaneously 
                    connection.write_3D_MassEBV_PDF(simdic[inp], thetawriter(theta, inp), arrdic[inp])
                else:
                    connection.write_2D_MassEBV_PDF(simdic[inp], thetawriter(theta, inp), arrdic[inp])
            elif ('alpha' in inp) or ('beta' in inp):
                #print(inp, 'allegedly only beta or alpha')
                connection.write_SALT2(inp, thetawriter(theta, inp))
            else:
                #print(inp)
                connection.write_1D_PDF(simdic[inp], thetawriter(theta, inp), arrdic[inp]) 
        if stretch:
            connection.write_1D_PDF('SIM_x1',[0.973, 1.472, 0.22], arrdic['x1'])
            #0.973 ± 0.105 1.472 ± 0.080 0.222 ± 0.076
        print('next',flush=True)
        #AAAAAAAA
        connection = connection_next(connection)# NOW RUN SALT2mu with these new distributions 
        print('got result', flush=True)
        ####AAAAAAAAA
        if connection.maxprob > 1.001:    
            print(connection.maxprob, 'MAXPROB parameter greater than 1! Coming up against the bounding function! Returning -np.inf to account, caught right after connection', flush=True)    
            return -np.inf 
        #print(connection.maxprob, flush=True)
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
        try: 
            bindf = connection.bindf #THIS IS THE PANDAS DATAFRAME OF THE OUTPUT FROM SALT2mu
            bindf = bindf.dropna()
            sim_vals = dffixer(bindf, 'ANALYSIS')
            realbindf = realdata.bindf #same for the real data (was a global variable)
            realbindf = realbindf.dropna()
            real_vals = dffixer(realbindf, 'ANALYSIS')
            resparr = []
            for lin in range(len(real_vals)):
                resparr.append([real_vals[lin],sim_vals[lin]])
        #I don't love this recasting, it's a memory hog, but it's temporary.
        except:
            print('WARNING! something went wrong in reading in stuff for the LL calc')
            return -np.inf

    except BrokenPipeError:
        if debug:
            print('WARNING! we landed in a Broken Pipe error')
            quit()
            tc = init_connection(299,real=False,debug=True)[1]
            return log_likelihood(tempin, connection=tc,returnall=True)
        else:
            print('WARNING! Slurm Broken Pipe Error!') #REGENERATE THE CONNECTION 
            print('before regenerating')
            newcon = (current_process()._identity[0]-1) #% see above at original connection generator, this has been changed 
            tc = init_connection(newcon,real=False)[1]
            connections[newcon] = tc
            return log_likelihood(theta,connection=tc)
    
    sys.stdout.flush()

    if returnall:
        out_result = LL_Creator(resparr, connection.beta, connection.sigint)  
        print("for", theta, "we found an LL of", out_result)            
        sys.stdout.flush()
        return LL_Creator(resparr, connection.beta, connection.sigint, returnall)
    else:
        out_result = LL_Creator(resparr, connection.beta, connection.sigint)
        print("for", theta, "we found an LL of", out_result)
        sys.stdout.flush()
        return out_result

###########################################################################################################################

def log_prior(theta, debug=False): #goes through expanded input parameters and checks that they are all within range. If any are not, returns negative infinity.
    thetadict = thetaconverter(theta)
    plist = pconv(inp_params) 
    if debug: print('plist', plist)
    tlist = False #if all parameters are good, this remains false 
    for key in thetadict.keys(): 
        if debug: print('key', key)
        temp_ps = (thetawriter(theta, key)) #I hate this but it works. Creates expanded list for this parameter
        if debug: print('temp_ps', temp_ps)
        plist_n = (thetawriter(theta, key, names=plist))
        for t in range(len(temp_ps)): #then goes through
            if debug: print('plist name', plist_n[t])
            lowb = cleandict[plist_n[t]][3][0] 
            highb = cleandict[plist_n[t]][3][1]
            if debug: print(lowb, temp_ps[t], highb)
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
        print('WARNING! We returned -inf from small parameters!')
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
    past_results = np.load(previous_samples)
    pos2 = past_results.f.arr_0[-1,:,:]
    if pos2.shape != pos.shape:
        print('The previously saved samples are not of the same dimensionality as those you are trying to fit for...')
        print('Quitting...')
        quit()
    else:
        pos = pos2

#print(pos)

#######################################################################
##### Run SALT2mu on the real data once! ################

if debug:
    realdata, _ = init_connection(299,debug=debug)
else:
    realdata, _ = init_connection(0,debug=debug)
#########################################################
realbeta = realdata.beta
realbetaerr = realdata.betaerr
realsigint = realdata.sigint
realsiginterr = 0.01


if doplot:    
    past_results = np.load(previous_samples)            
    past_results = past_results.f.arr_0
    Criteria_Plotter(tempin)               
    pltting_func(past_results, inp_params) 
    print('done')
    quit()

elif single:           
    print('inp', tempin)          
    Criteria_Plotter(tempin)      
    quit()  

if shotnoise:
    tc = init_connection(299,real=False,debug=True)[1]
    for tp in range(200):
        chisq, datacount_list, simcount_list, poisson_list = log_likelihood((tempin),returnall=True,connection=tc) 
        newfile = 'SHOTNOISE/%d_SIMDATA_OUT.M0DIF'%tp;
        os.rename('OUT_SIM.M0DIF', newfile)   
    quit()

if wfit:
    npoints = 200
    past_results = np.load(previous_samples)
    past_results = past_results.f.arr_0#[-npoints:,:,:]
    past_results = past_results.reshape(-1,past_results.shape[-1])
    #past_results = np.unique(past_results, axis=0)#[::-npoint_step]
    templen = len(past_results[:,0])
    npoint_step = int(templen/(npoints))
    tempin = past_results[::npoint_step,:]
    tc = init_connection(299,real=False,debug=True)[1]
    f = open('LL_List', 'w') #' '.join([str(elem) for elem in listy])
    f.write('Index LL '+str( ' '.join([str(elem) for elem in pconv(inp_params)]))+' \n')
    f.close()
    for i in range(npoints):
        f = open('LL_List', 'a')
        print('starting M0DIF ', i)
        #print(tempin[i])
        chisq, datacount_list, simcount_list, poisson_list = log_likelihood((tempin[i]),returnall=True,connection=tc)  
        print('RESULT!', chisq, flush=True) 
        bigstr = ''
        bigstr += str(i)+' '+str(np.sum(chisq))
        for t in tempin[i]:
            bigstr += ' '+str(t)
        bigstr += ' \n'
        f.write(bigstr)
        print(str(i)+','+str(np.sum(chisq))+','+str(tempin[i]))
        sys.stdout.flush() 
        newfile = 'M0DIF/%d_SIMDATA_OUT.M0DIF'%i;
        os.rename('OUT_SIM.M0DIF', newfile)
        f.close()
    quit()

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
    print('inp', tempin)
    #Criteria_Plotter(tempin)
    for tp in range(20):
        print(f"loop {tp}")
        log_likelihood((tempin),returnall=True,connection=tc)
    abort #deliberately switched the 0.11 (stdr) and 0.03 (stdl) to get wrong value


with Pool(nconn) as pool: #this is where Brodie comes in to get mc running in parallel on batch jobs. 
    #Instantiate the sampler once (in parallel)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)

    for qb in range(20):
        print("Starting loop iteration", qb)
        print('begun', cpu_count(), "CPUs with", nwalkers, ndim, "walkers and dimensions")
        #print(psutil.virtual_memory().available * 100 / psutil.virtual_memory().total, "is the percentage of memory used")
        sys.stdout.flush()
        #Run the sampler
        if qb == 0:
            state = sampler.run_mcmc(pos, 100, progress=True) #There used to be a semicolon here for some reason 
            print('Finished burn-in!')
            sampler.reset()
            sampler.run_mcmc(state, 100, progress=True)
        else:
            state = sampler.run_mcmc(state, 100, progress=True)

        #May need to implement a proper burn-in 
        sys.stdout.flush() 

        #Save the output for later
        samples = sampler.get_chain()
        pos = samples[-1,:,:]
        np.savez(data_input.split('.')[0]+'-samples.npz',samples)
        #print(pos, "New samples as input for next run")

        print(pos[0])
        pltting_func(samples, inp_params)                                  
        Criteria_Plotter(pos[0])                                                                          
                            

        ### THATS IT! (below is just plotting to monitor the fitting) ##########
        ########################################################################
        ########################################################################

