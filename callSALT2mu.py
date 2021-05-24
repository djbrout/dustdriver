import subprocess
import time
import sys
import numpy as np
import pandas as pd
from io import StringIO
import os

massarr = np.arange(5.05,15.,.1)

class SALT2mu: #I understand classes better now
    def __init__(self,command,mapsout,SALT2muout,log,realdata=False,debug=False): #setting all sorts of class values
        #print(command%(mapsout,SALT2muout,log))
        self.iter=-1
        self.debug=debug #Boolean. Default False.
        self.ready = 'Enter expected ITERATION number'
        self.ready2 = 'ITERATION=%d'
        self.done = 'Graceful Program Exit. Bye.'
        self.initready = 'Finished SUBPROCESS_INIT'
        self.crosstalkfile = open(mapsout,'w')
        self.SALT2muoutputs = open(SALT2muout,'r') #An output file
        if realdata: #this is awful 
            os.system(command%(mapsout,SALT2muout,log)) #command is an input variable formatted as a string and passed to system
            #print(command%(mapsout,SALT2muout,log)) 
            self.getData() #calls getData
        else:
            #print(command%(mapsout,SALT2muout,log))
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

    def next(self): #ticks up iteration by one 
        print('writing next iter to stdin')
        self.process.stdin.write('%d\n'%self.iter)  
        
    def quit(self): #sets iteration input to -1, which causes a quit somewhere
        self.process.stdin.write('-1\n')
        for stdout_line in iter(self.process.stdout.readline, ""): 
            print(stdout_line)

    def getResult(self): #
        start = False #ok fine
        self.stdout_iterator = iter(self.process.stdout.readline, "") #what
        #print(self.stdout_iterator)
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
                    print('getting data')
                    self.data = self.getData()                
                    return 

    def getData(self): #reads in the current SALT2mu output fitres and scrapes data.
        self.SALT2muoutputs.seek(0) #sets pointer to top of file
        text = self.SALT2muoutputs.read() #reads in the text
        print(text)
        self.alpha = float(text.split('alpha0')[1].split()[1]) 
        self.alphaerr = float(text.split('alpha0')[1].split()[3])
        self.beta = float(text.split('beta0')[1].split()[1])
        self.betaerr = float(text.split('beta0')[1].split()[3])
        self.cbindf = pd.read_csv(StringIO(text),comment='#',delim_whitespace=True) #this appears to be the fitres file 
        return True

    def get_1d_asym_gauss(self,mean,lhs,rhs,arr): #creates a 1d, asymmetric Gaussian
        probs = np.exp(-.5*((arr-mean)/lhs)**2) 
        probs[arr>mean] = np.exp(-.5*((arr[arr>mean]-mean)/rhs)**2)
        probs = probs/np.max(probs)
        return arr,probs #x and y

    def writeheader(self,names): #writes the header 
        self.crosstalkfile.write('VARNAMES:')
        for name in names:
            self.crosstalkfile.write(' '+name)
        self.crosstalkfile.write(' PROB\n')
        return 

    def write2Dprobs(self,arr,mass,probs): #Writes a 2D probability 
        bigstr = ''
        for a,p in zip(arr,probs):
            bigstr+='PDF: %.3f %.2f %.3f\n'%(a,mass,p)
        self.crosstalkfile.write(bigstr)

    def write1Dprobs(self,arr,probs): #Writes a 1D probability 
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

