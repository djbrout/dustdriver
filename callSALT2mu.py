import subprocess
import time
import sys
import numpy as np
import pandas as pd
from io import StringIO
import os
import logging

def setup_custom_logger(name, screen=False):
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    handler = logging.FileHandler(f'logs/log_{name}.log', mode='a+')
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    if screen:
        logger.addHandler(screen_handler)
    return logger
    #END setup_custom_logger

class SALT2mu: #I understand classes better now
    def __init__(self,command,mapsout,SALT2muout,log,realdata=False,debug=False): #setting all sorts of class values
        #print(command%(mapsout,SALT2muout,log))
        self.logger = setup_custom_logger('walker_'+os.path.basename(mapsout).split('_')[0])
        self.iter=-1
        self.debug = debug #Boolean. Default False.
        self.ready = 'Enter expected ITERATION number'
        self.ready2 = 'ITERATION=%d'
        self.done = 'Graceful Program Exit. Bye.'
        self.initready = 'Finished SUBPROCESS_INIT'
        self.crosstalkfile = open(mapsout,'w')
        self.SALT2muoutputs = open(SALT2muout,'r') #An output file

        self.command = command

        self.logger.info('Init SALT2mu instance. ')
        self.logger.info('## ================================== ##')
        self.logger.info(f'Command: {self.command}')
        self.logger.info(f'mapsout: {mapsout}')
        self.logger.info(f'Debug mode={self.debug}')

        if self.debug:
            self.command = self.command + ' write_yaml=1'

        if realdata: #this is awful )
            self.logger.info('Running realdata=True')
            os.system(command%(mapsout,SALT2muout,log)) #command is an input variable formatted as a string and passed to system
            self.logger.info('Command being run: ' + (command%(mapsout,SALT2muout,log))) 
            self.getData() #calls getData
        else:
            self.logger.info('Running realdata=False')

            self.process = subprocess.Popen((self.command%(mapsout,SALT2muout,log)).split(), 
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               stdin=subprocess.PIPE,
                               bufsize=0,
                               universal_newlines=True)
            self.logger.info('Command being run: ' + (self.command%(mapsout,SALT2muout,log))) 
            self.stdout_iterator = iter(self.process.stdout.readline, "")
        #END __init__

    def next(self): #ticks up iteration by one 
        print('writing next iter to stdin')
        self.process.stdin.write('%d\n'%self.iter)  
        #END next
        
    def quit(self): #sets iteration input to -1, which causes a quit somewhere
        self.process.stdin.write('-1\n')
        for stdout_line in iter(self.process.stdout.readline, ""): 
            print(stdout_line)
        #END quit

    def getResult(self):
        start = False #ok fine
        self.stdout_iterator = iter(self.process.stdout.readline, "") #what
        #print(self.stdout_iterator)
        for stdout_line in self.stdout_iterator:
            if self.debug: 
                self.logger.debug(stdout_line)
            if str(self.done) in stdout_line:
                return
            #if str(self.ready) in stdout_line:
            if self.iter == -1:
                if str(self.ready) in stdout_line:
                    #self.data = self.getData()
                    return
            else:
                #if self.ready2%(self.iter-1) in stdout_line:
                if str(self.ready) in stdout_line:
                    print('getting data')
                    self.data = self.getData()                
                    return 
        #END getResult

    def getData(self): #reads in the current SALT2mu output fitres and scrapes data.
        self.SALT2muoutputs.seek(0) #sets pointer to top of file
        text = self.SALT2muoutputs.read() #reads in the text
        self.alpha = float(text.split('alpha0')[1].split()[1]) 
        self.alphaerr = float(text.split('alpha0')[1].split()[3])
        self.beta = float(text.split('beta0')[1].split()[1])
        self.betaerr = float(text.split('beta0')[1].split()[3])
        self.maxprob = float(text.split('MAXPROB_RATIO')[1].split()[1])
        self.headerinfo = self.NAndR(StringIO(text))
        self.sigint = float(text.split('sigint')[1].split()[1])
        self.bindf = pd.read_csv(StringIO(text), header=None, skiprows=self.headerinfo[1],names=self.headerinfo[0], delim_whitespace=True, comment='#')
        return True
        #END getData

    def get_1d_asym_gauss(self,mean,lhs,rhs,arr): #creates a 1d, asymmetric Gaussian
        probs = np.exp(-.5*((arr-mean)/lhs)**2) 
        probs[arr>mean] = np.exp(-.5*((arr[arr>mean]-mean)/rhs)**2)
        probs = probs/np.max(probs)
        return arr,probs #x and y
        #END get_1d_asym_gauss

    def get_1d_exponential(self, tau, arr):
        probs = (tau**-1)*np.exp(-arr/tau)
        probs = probs/np.max(probs)
        return arr, probs
        #END get_1d_exponential

    def get_1d_lognormal(self, mu, std, arr):
        probs = np.exp(mu + std*arr)
        probs = probs/np.max(probs)
        return arr, probs
        #END get_1d_lognormal

    def writeheader(self,names): #writes the header - DEPRECATED
        self.crosstalkfile.write('VARNAMES:')
        for name in names:
            self.crosstalkfile.write(' '+name)
        self.crosstalkfile.write(' PROB\n')
        return 
        #END writeheader

    def writegenericheader(self, inp, varnames):
        self.crosstalkfile.write(f'VARNAMES: {inp}')
        for name in varnames: self.crosstalkfile.write(' '+name); 
        self.crosstalkfile.write(' PROB \n')
        return
        #END writegenericheader

    def write3Dprobs(self,arr,z,mass,probs): #Writes a 3D probability    
        bigstr = ''    
        for a,p in zip(arr,probs): 
            bigstr+='PDF: %.3f %.2f %.2f %.3f\n'%(a,z,mass,p)        
        self.crosstalkfile.write(bigstr)        
        #END write3Dprobs

    def write2Dprobs(self,arr,mass,probs): #Writes a 2D probability 
        bigstr = ''
        for a,p in zip(arr,probs):
            bigstr+='PDF: %.3f %.2f %.3f\n'%(a,mass,p)
        self.crosstalkfile.write(bigstr)
        #END write2Dprobs

    def write1Dprobs(self,arr,probs): #Writes a 1D probability 
        bigstr = ''
        for a,p in zip(arr,probs):
            bigstr+='PDF: %.3f %.3f\n'%(a,p)
        bigstr+='\n'
        self.crosstalkfile.write(bigstr)
        #END write1Dprobs

    def write_iterbegin(self):
        self.crosstalkfile.truncate(0)
        self.crosstalkfile.seek(0)
        self.crosstalkfile.write('ITERATION_BEGIN: %d\n'%self.iter)
        #END write_iterbegin

    def write_iterend(self):
        self.crosstalkfile.write('ITERATION_END: %d\n'%self.iter)
        self.crosstalkfile.flush()
        #END write_iterend

    def write_SALT2(self, name, PARAMS):
        for tm in range(3):    
            self.crosstalkfile.write("\n") 
        mean = PARAMS[0]
        std = PARAMS[1]
        self.crosstalkfile.write(f"GENPEAK_SIM_{name}: {mean} \n")
        self.crosstalkfile.write(f"GENSIGMA_SIM_{name}: {std} {std} \n")
        if name == 'alpha':
            self.crosstalkfile.write(f"GENRANGE_SIM_{name}: .1 .2 \n")
        else:
            self.crosstalkfile.write(f"GENRANGE_SIM_{name}: .4 3 \n")
        for tm in range(3):
            self.crosstalkfile.write("\n")
        #END write_SALT2

    def write_1D_PDF(self,varname, PARAMS, arr): #DEPRECATED
        self.writeheader([varname])
        try:
            mean,lhs,rhs = PARAMS
        except ValueError:
            mean = PARAMS[0]
            lhs = PARAMS[1]
            rhs = lhs
        arr,probs = self.get_1d_asym_gauss(mean,lhs,rhs,arr)
        self.write1Dprobs(arr,probs)
        #END write_1D_PDF

    def writeRVLOGMASS_PDF(self,mean,lhs,rhs): #DEPRECATED
        self.writeheader(['RV','LOGMASS'])
        for mass in massarr:
            if mass < 10:
                arr,probs = self.get_1d_asym_gauss(mean,lhs,rhs)
                self.write2Dprobs(arr,mass,probs)
            else:
                arr,probs = self.get_1d_asym_gauss(mean,lhs,rhs)
            probs[arr<.5] = 0
            self.write2Dprobs(arr,mass,probs)
        #END writeRVLOGMASS_PDF

    def write_2D_Mass_PDF(self, varname, PARAMS, arr): #DEPRECATED
        self.writeheader([varname, 'HOST_LOGMASS'])
        Lmean, Llhs, Lrhs = PARAMS[0:3]
        Hmean, Hlhs, Hrhs = PARAMS[3:]        
        for mass in massarr:               
            if mass < 10:                                                                
                arr,probs = self.get_1d_asym_gauss(Lmean,Llhs,Lrhs,arr)                
                probs[arr < .4] = 0
                self.write2Dprobs(arr,mass,probs)
            else:                          
                arr,probs = self.get_1d_asym_gauss(Hmean,Hlhs,Hrhs,arr)                 
                probs[arr < .4] = 0
                self.write2Dprobs(arr,mass,probs)
        self.crosstalkfile.write('\n')
        #END write_2D_Mass_PDF

    def write_2D_Mass_PDF_SYMMETRIC(self, varname, PARAMS, arr):  #DEPRECATED
        self.writeheader([varname, 'HOST_LOGMASS'])                 
        Lmean, Llhs = PARAMS[0:2]
        Lrhs = Llhs
        Hmean, Hlhs = PARAMS[2:]
        Hrhs = Hlhs
        for mass in massarr:      
            if mass < 10:           
                arr,probs = self.get_1d_asym_gauss(Lmean,Llhs,Lrhs,arr)   
                probs[arr < .4] = 0        
                self.write2Dprobs(arr,mass,probs)     
            else:          
                arr,probs = self.get_1d_asym_gauss(Hmean,Hlhs,Hrhs,arr)  
                probs[arr < .4] = 0        
                self.write2Dprobs(arr,mass,probs)      
        self.crosstalkfile.write('\n') 
        #END write_2D_Mass_PDF_SYMMETRIC

    def write_2D_MassEBV_PDF(self, varname, PARAMS, arr):  #DEPRECATED
        self.writeheader([varname, 'HOST_LOGMASS'])       
        LTau = PARAMS[0]
        HTau = PARAMS[1]  
        for mass in massarr:    
            if mass < 10:      
                arr,probs = self.get_1d_exponential(LTau,arr)     
                self.write2Dprobs(arr,mass,probs)
            else:             
                arr,probs = self.get_1d_exponential(HTau,arr)         
                self.write2Dprobs(arr,mass,probs)
        self.crosstalkfile.write('\n')
        #END write_2D_MassEBV_PDF

    def write_2D_LOGNORMAL_PDF(self, varname, PARAMS, arr): #DEPRECATED
        self.writeheader([varname, 'HOST_LOGMASS']) 
        Lmu, Lstd = PARAMS[0,1]
        Hmu, Hstd = PARAMS[2,3]    
        for mass in massarr:   
            if mass < 10:   
                arr,probs = self.get_1d_lognormal(Lmu, Lstd, arr) 
                self.write2Dprobs(arr,mass,probs)  
            else: 
                arr,probs = self.get_1d_lognormal(Hmu, Hstd, arr)
                self.write2Dprobs(arr,mass,probs)
        self.crosstalkfile.write('\n')  
        #END write_2D_LOGNORMAL_PDF

    def write_3D_MassEBV_PDF(self, varname, PARAMS, arr):   #for when EBV needs a z split - DEPRECATED
        self.writeheader([varname, 'SIM_ZCMB', 'HOST_LOGMASS']) #needs work   
        LZ_LTau = PARAMS[0]    
        LZ_HTau = PARAMS[1]   
        HZ_LTau = PARAMS[2]
        HZ_HTau = PARAMS[3]
        for z in zarr:
            if np.around(z,3) < 0.1:
                for mass in massarr:      
                    if mass < 10:  
                        arr,probs = self.get_1d_exponential(LZ_LTau,arr)    
                        self.write3Dprobs(arr,z,mass,probs)  
                    else:  
                        arr,probs = self.get_1d_exponential(LZ_HTau,arr) 
                        self.write3Dprobs(arr,z,mass,probs)    
            else:
                for mass in massarr: 
                    if mass < 10:           
                        arr,probs = self.get_1d_exponential(HZ_LTau,arr)     
                        self.write3Dprobs(arr,z,mass,probs)           
                    else:    
                        arr,probs = self.get_1d_exponential(HZ_HTau,arr) 
                        self.write3Dprobs(arr,z,mass,probs)  
        self.crosstalkfile.write('\n') 
        #END write_3D_MassEBV_PDF

    def write_3D_LOGNORMAL_PDF(self, varname, PARAMS, arr):   #for when EBV needs a z split - DEPRECATED
        self.writeheader([varname, 'SIM_ZCMB', 'HOST_LOGMASS']) #needs work  
        LZ_Lmu, LZ_Lstd, LZ_Hmu, LZ_Hstd, HZ_Lmu, HZ_Lstd, HZ_Hmu, HZ_Hstd = PARAMS 
        for z in zarr:    
            if np.around(z,3) < 0.1:      
                for mass in massarr:  
                    if mass < 10:      
                        arr,probs = self.get_1d_lognormal(LZ_Lmu,LZ_Lstd,arr)      
                        self.write3Dprobs(arr,z,mass,probs)     
                    else:      
                        arr,probs = self.get_1d_lognormal(LZ_Hmu,LZ_Hstd,arr)   
                        self.write3Dprobs(arr,z,mass,probs)                
            else:    
                for mass in massarr:    
                    if mass < 10:  
                        arr,probs = self.get_1d_lognormal(HZ_Lmu,HZ_Lstd,arr)    
                        self.write3Dprobs(arr,z,mass,probs)  
                    else:    
                        arr,probs = self.get_1d_lognormal(HZ_Hmu,HZ_Hstd,arr) 
                        self.write3Dprobs(arr,z,mass,probs)   
        self.crosstalkfile.write('\n') 
        #END write_3D_LOGNORMAL_PDF

    def write_2D_PDF(self, varname, LOWPARAMS, HIGHPARAMS, arr):#DEPRECATED
        self.writeheader([varname[0], varname[1]])                    
        Lmean, Llhs, Lrhs = LOWPARAMS
        Hmean, Hlhs, Hrhs = HIGHPARAMS 
        for mass in arr: 
            if mass < 10:                        
                arr,probs = self.get_1d_asym_gauss(Lmean,Llhs,Lrhs)                  
            else:           
                arr,probs = self.get_1d_asym_gauss(Hmean,Hlhs,Hrhs)                             
            probs[arr<.5] = 0                             
            self.write2Dprobs(arr,mass,probs) 
        #END write_2D_PDF

    def NAndR(self, fp):
        for i, line in enumerate(fp):
            if line.startswith('VARNAMES:'):
                line = line.replace(',',' ')
                line = line.replace('\n','')
                Names = line.split()
            elif line.startswith('SN') or line.startswith('ROW:') or line.startswith('GAL'):
                Startrow = i
                break  
        return Names, Startrow 
        #END NAndR
        
    def write_generic_PDF(self, inp, SPLIT, PARAMS, SHAPE, SHAPEDICT,SIMDICT, arr):
        if ('alpha' in inp) or ('beta' in inp): #quick check to see if it's one of the weird parameters.
            self.write_SALT2(inp, PARAMS)
            return 'Done' #Ends the function early because beta/alpha don't work as below.
        splits = [] #Creates empty array for consistency
        if inp in SPLIT.keys(): splits = list(SPLIT[inp].keys()) #if splits exist for this inp, lists
        splits = [SIMDICT[i] for i in splits] #convert to SNANA readable format
        splitloc = [SPLIT[inp][i] for i in splits]
        self.writegenericheader(SIMDICT[inp], splits) #Done writing the header.
        if len(splits) == 0:
            arrs,probs= self.shape_assigner(PARAMS, SHAPE, arr[0])
            if inp == 'RV': probs[arrs < .4] = 0
            self.write1Dprobs(arrs, probs)
        elif len(splits) == 1:
            SPLITPARAMS = self.shape_interpret(PARAMS, inp, SHAPE, SHAPEDICT, SPLIT)
            for sp1 in arr[1]:
                if sp1 < splitloc[0]:
                    arrs,probs= self.shape_assigner(SPLITPARAMS[0], SHAPE, arr[0])
                    if inp == 'RV': probs[arrs < .4] = 0
                    self.write2Dprobs(arrs,sp1, probs)
                else:
                    arrs,probs= self.shape_assigner(SPLITPARAMS[1], SHAPE, arr[0])
                    if inp == 'RV': probs[arrs < .4] = 0
                    self.write2Dprobs(arrs, sp1, probs)
        elif len(splits) == 2:
            SPLITPARAMS = self.shape_interpret(PARAMS, inp, SHAPE, SHAPEDICT, SPLIT)
            for sp1 in arr[1]: #starts first split "low" version 
                if np.around(sp1,3) < splitloc[0]: #if first split condition is met
                    for sp2 in arr[2]: #Second split 
                        if np.around(sp2,1) < splitloc[1]: #Start second split "low" version
                            arrs,probs= self.shape_assigner(SPLITPARAMS[0], SHAPE, arr[0])
                            if inp == 'RV': probs[arrs < .4] = 0
                            self.write3Dprobs(arrs,sp1,sp2,probs)
                        else:
                            arrs,probs= self.shape_assigner(SPLITPARAMS[1], SHAPE, arr[0])
                            if inp == 'RV': probs[arrs < .4] = 0
                            self.write3Dprobs(arrs,sp1,sp2,probs)
                else: #starts first split "high" version
                    for sp2 in arr[2]: #Start second split
                        if np.around(sp2,1) < splitloc[1]: #Starts second split "low" version
                            arrs,probs= self.shape_assigner(SPLITPARAMS[2], SHAPE, arr[0])
                            if inp == 'RV': probs[arrs < .4] = 0
                            self.write3Dprobs(arrs,sp1,sp2,probs)
                        else:
                            arrs,probs= self.shape_assigner(SPLITPARAMS[3], SHAPE, arr[0])
                            if inp == 'RV': probs[arrs < .4] = 0
                            self.write3Dprobs(arrs,sp1,sp2,probs)
        self.crosstalkfile.write('\n')
        #END write_generic_PDF

    def shape_assigner(self, PARAMS, SHAPE, arr):
        if SHAPE == 'Gaussian':
            return self.get_1d_asym_gauss(PARAMS[0], PARAMS[1], PARAMS[1],arr)
        elif SHAPE == 'Exponential':
            return self.get_1d_exponential(PARAMS[0], arr)
        elif SHAPE == 'LogNormal':
            return self.get_1d_lognormal(PARAMS[0], PARAMS[1], arr)
        elif SHAPE == 'Skewed Gaussian':
            return self.get_1d_asym_gauss(PARAMS[0], PARAMS[1], PARAMS[2],arr)
        #END shape_assigner

    def shape_interpret(self, PARAMS, inp, SHAPE, SHAPEDICT, SPLIT):       
        temp = []                                                             
        for i in range(len(SPLIT[inp].keys())*2):  
            temp.append(PARAMS[0+i*len(SHAPEDICT[SHAPE]):(i+1)*len(SHAPEDICT[SHAPE])]) 
        return temp 
        #END shape_interpret
