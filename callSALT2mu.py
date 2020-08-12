import subprocess
import time
import sys
import numpy as np
import pandas as pd
from io import StringIO
import os

massarr = np.arange(5.05,15.,.1)

class SALT2mu:
    def __init__(self,command,mapsout,SALT2muout,log,realdata=False,debug=False):
        #print(command%(mapsout,SALT2muout,log))
        self.iter=-1
        self.debug=debug
        self.ready = 'Enter expected ITERATION number'
        self.ready2 = 'ITERATION=%d'
        self.done = 'Graceful Program Exit. Bye.'
        self.initready = 'Finished SUBPROCESS_INIT'
        self.crosstalkfile = open(mapsout,'w')
        self.SALT2muoutputs = open(SALT2muout,'r')
        if realdata:
            os.system(command%(mapsout,SALT2muout,log))
            self.getData()
        else:
            self.process = subprocess.Popen((command%(mapsout,SALT2muout,log)).split(), 
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               stdin=subprocess.PIPE,
                               bufsize=0,
                               universal_newlines=True)
            self.stdout_iterator = iter(self.process.stdout.readline, "")
            #self.getResult()  

            #self.call()
            #self.next()
            #self.getResult()

    #def call(self):
    #    self.getResult()
    #    return self.data

    def next(self):
        self.process.stdin.write('%d\n'%self.iter)  
        
    def quit(self):
        self.process.stdin.write('-1\n')
        for stdout_line in iter(self.process.stdout.readline, ""):
            print(stdout_line)

    def getResult(self):
        start = False
        for stdout_line in self.stdout_iterator:
            if self.debug: print(stdout_line)
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
                    #print('getting data')
                    self.data = self.getData()                
                    return 

    def getData(self):
        self.SALT2muoutputs.seek(0)
        text = self.SALT2muoutputs.read()
        self.alpha = float(text.split('alpha0')[1].split()[1])
        self.alphaerr = float(text.split('alpha0')[1].split()[3])
        self.beta = float(text.split('beta0')[1].split()[1])
        self.betaerr = float(text.split('beta0')[1].split()[3])
        self.cbindf = pd.read_csv(StringIO(text),comment='#',delim_whitespace=True)
        return True

    def get_1d_asym_gauss(self,mean,lhs,rhs,arr):
        probs = np.exp(-.5*((arr-mean)/lhs)**2)
        probs[arr>mean] = np.exp(-.5*((arr[arr>mean]-mean)/rhs)**2)
        probs = probs/np.max(probs)
        return arr,probs

    def writeheader(self,names):
        self.crosstalkfile.write('VARNAMES:')
        for name in names:
            self.crosstalkfile.write(' '+name)
        self.crosstalkfile.write(' PROB\n')
        return 

    def write2Dprobs(self,arr,mass,probs):
        bigstr = ''
        for a,p in zip(arr,probs):
            bigstr+='PDF: %.3f %.2f %.3f\n'%(a,mass,p)
        self.crosstalkfile.write(bigstr)

    def write1Dprobs(self,arr,probs):
        bigstr = ''
        for a,p in zip(arr,probs):
            bigstr+='PDF: %.3f %.3f\n'%(a,p)
        self.crosstalkfile.write(bigstr)

    def write_iterbegin(self):
        self.crosstalkfile.truncate(0)
        self.crosstalkfile.seek(0)
        self.crosstalkfile.write('ITERATION_BEGIN: %d\n'%self.iter)

    def write_iterend(self):
        self.crosstalkfile.write('ITERATION_END: %d\n'%self.iter)
        self.crosstalkfile.flush()

    def write_1D_PDF(self,varname,mean,lhs,rhs,arr):
        self.writeheader([varname])
        arr,probs = self.get_1d_asym_gauss(mean,lhs,rhs,arr)
        self.write1Dprobs(arr,probs)

    def writeRVLOGMASS_PDF(self,mean,lhs,rhs):
        self.writeheader(['RV','LOGMASS'])
        for mass in massarr:
            if mass < 10:
                arr,probs = self.get_1d_asym_gauss(mean,lhs,rhs)
            else:
                arr,probs = self.get_1d_asym_gauss(mean,lhs,rhs)
            
            probs[arr<.5] = 0
            self.write2Dprobs(arr,mass,probs)

