###Base e modelo 
import subprocess 
import sys
import numpy as np
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpld3
from scipy.integrate import odeint
import lmfit
from lmfit.lineshapes import gaussian, lorentzian
import chaospy as cp
from scipy.integrate import odeint
from lmfit import minimize, Parameters, Parameter, report_fit
from SALib.sample import saltelli
from SALib.analyze import sobol
import timeit
import re

class TTCellModel:
    tf=1000
    ti=0
    dt=0.01
    dtS=1
    parametersN={"gK1":-100,"gKs":-100,"gKr":-100,"gNa":-100,"gbna":-100,"gCal":-100,"gbca":-100,"gto":-100}
    def parametize(self,ps):
        params={}
        i=0;
        for val  in (TTCellModel.parametersN):
            try:
                params[val]=ps[val]
                i+=1
            except:
                params[val]=-100
        return params;
    
    def __init__(self,params):
        self.parameters = self.parametize(params)

    
    @staticmethod
    def getSimSize(): #Returns size of result vector for given simulation size parameters, usefull for knowing beforehand the number of datapoints to compare
        n=TTCellModel("").run().shape
        print(n)
        return n
    
    
    @staticmethod
    def setSizeParameters(ti,tf,dt,dtS):
        TTCellModel.ti=ti
        TTCellModel.tf=tf
        TTCellModel.dt=dt
        TTCellModel.dtS=dtS
        return TTCellModel.getSimSize()
    @staticmethod   #runs the model once for the given size parameters and returns the time points at wich there is evalution
    def getEvalPoints():
        n=TTCellModel("").run()
        tss= np.zeros(n.shape[0])
        for i,timepoint in enumerate(n[:,0]):
            tss[i]=float(timepoint[0]);
            
        return tss
    
    
        
    def ads(self,sol,repoCofs): ##calculo da velocidade de repolarização
        k=0
        i=0;
        out={}
        x=sol
        flag=0
        try: 
            try:
                x=sol[0:+TTCellModel.tf,0].ravel().transpose()
            except:
                x=sol
            index=0
            for value in x:
                index+=1  
                if(value==x.max()):
                        flag=1                
                        out[len(repoCofs)]=index +TTCellModel.ti 
                if(flag==1):
                        k+=1
                if(flag==1 and repoCofs[i]*x.min() >= value):
                        out[i]=k 
                        i+=1
                if(i>=len(repoCofs)):
                        break
        except:
            print("ADCALCERROR")
            print(x)       
        return out
    
    def callCppmodel(self,params):     
        args="C:\\s\\uriel-numeric\\Release\\cardiac-cell-solver.exe " +"--tf="+str(TTCellModel.tf)+" --ti="+str(TTCellModel.ti)+" --dt="+str(TTCellModel.dt)+" --dt_save="+str(TTCellModel.dtS)  
        for value in params:
            if(params[value]!=-100):
                args+= " "
                args+=" --"+value+"="+(str(params[value]))[:9]
        output = subprocess.Popen(args,stdout=subprocess.PIPE)
        matrix={}
        try:
            string = output.stdout.read().decode("utf-8")
            matrix = np.matrix(string)
            return matrix
        
        except:
            print(args)
            print(params)
            print("\n")
            
        return matrix
    
    def plot_sir(self, r, labels):
        
        x=(r[:,0])
        y=(r[:,1])
        plt.plot(x, y,label=labels[0])
        ads=self.ads(y,[0.5,0.75,0.9])
        try:
            plt.axvline(x=ads[3], label='Depolarization')
            plt.axvline(x=ads[0]+ads[3], label='APD90')
            plt.axvline(x=ads[1]+ads[3], label='APD50')
            plt.axvline(x=ads[2]+ads[3], label='APD75')
        except:
            print("no repo")
            
        plt.xlabel("tempo")
        plt.ylabel("Variação no potencial")
        plt.legend(loc='best')
        plt.show()
        
        # parametros 
  
    def run(self):  
        x = self.callCppmodel(self.parameters)
        return x
    
   
    
    