# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 02:58:23 2021

@author: yanbw
"""

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
from modelTT import TTCellModel
import math
from sklearn import linear_model as lm
import csv


def Sobol():
    problem = {
    "num_vars": 6,
    "names": labels,
    "bounds": [ [gK1*0.9,gK1*1.1], [gKs*0.9,gKs*1.1], [gKr*0.9,gKr*1.1],[gto*0.9,gto*1.1],[gNa*0.9,gNa*1.1],[gCal*0.9,gCal*1.1]],
}
    Nsobol=1000
    param_vals = saltelli.sample(problem,Nsobol,  calc_second_order=False)
    Ns = param_vals.shape[0]
    Y=np.empty([Ns])
    for i in range(Ns):
        Y[i]=cp.call(fitted_polynomial,param_vals[i])  
    sensitivity = sobol.analyze(problem,Y , calc_second_order=False)
    return sensitivity['S1']

def readF(fn):
    X=[]
    file = open(fn, 'r')
    for row in file:
        X.append([float(x) for x in row.split(',')])
    return X



def calcula_loo(y, poly_exp, samples,model):
    """ 
    LOO for a scalar quantity: y 
    """
    #print ("\n\nPerforming Leave One Out cross validation.")    
    nsamp = np.shape(y)[0]
    #print(nsamp)
    #print(y)

    deltas = np.zeros(nsamp)
    samps = samples.T

    start = timeit.default_timer()

    for i in range(nsamp):
        indices = np.linspace(0,nsamp-1,nsamp, dtype=np.int32)
        indices = np.delete(indices,i)
        subs_samples = samps[indices,:].copy()
        subs_y =[ y[i] for i in (indices)]

        subs_poly = cp.fit_regression (poly_exp,subs_samples.T,subs_y,model=model,retall=False) 
        yhat = cp.call(subs_poly, samps[i,:])
        deltas[i] = ((y[i] - yhat))**2

    y_std = np.std(y)
    err = np.mean(deltas)/np.var(y)
    acc = 1.0 - np.mean(deltas)/np.var(y)
    
    
    stop = timeit.default_timer()

    return err


#Load validation files

##Reference Value for Sobol Indexes Error calculation
sensitivity={}
sensitivity['S5']=[1.55538786e-02, 1.82850519e-01, 9.61438769e-02, 2.99173145e-03,
 1.56245611e-05, 7.60818660e-01]
sensitivity['S9']=[ 5.31159406e-02 , 1.68227767e-01 , 1.13413410e-01, -3.96239115e-04,
  1.25679367e-03, 7.20783737e-01 ]
sensitivity['SVM']=[3.98217241e-04 , 3.09025984e-05, -4.28701329e-05 , 1.58306471e-05,
  9.94885997e-01 , 3.35410661e-04 ]
sensitivity['SVR']=[7.48449580e-01,3.65262866e-03, 2.01777153e-03 ,4.39363109e-04,
 1.82630159e-05, 2.43027627e-01]

#Validation Samples
X=readF("Xval.txt")
samplesVal=np.zeros((len(X),6))
for i,sample in enumerate(X):       ##must be matrix not list
    for k,y in enumerate(sample):
        samplesVal[i][k]=y


##Load Result File
f = open('resultstrial.csv', 'w',newline='')

# create the csv writer
writer = csv.writer(f)
row=['QOI',	'Method', 'Degree','Val. error',' LOOERROR','Max Sobol Error','Mean Sobol Error','Ns']
writer.writerow(row)



Y = readF("Yval.txt")
Y= np.array(Y)

Yval={
      
     "ADP90":Y[:,0],
     "ADP50":Y[:,1],
     "dVmax":Y[:,2],
     "Vrest":Y[:,3],
 
     
     }


#Simulation Parameters
labels=["gK1","gKs","gKr","gto","gNa","gCal"]
TTCellModel.setParametersOfInterest(["gK1","gKs","gKr","gto","gNa","gCal"])
nPar=6
#size parameteres
ti=3000
tf=4000
dt=0.01
dtS=1
size=TTCellModel.setSizeParameters(ti, tf, dt, dtS)  
Timepoints=TTCellModel.getEvalPoints()



#Parameters of interest X
gK1=  5.4050e+00 
gKs= 0.245 
gKr = 0.096 
gNa = 1.48380e+01 
gbna =  2.90e-04
gCal =   1.750e-04 
gbca = 5.920e-04
gto =  2.940e-01 

low=0.9
high=1.1

gK1d  = cp.Uniform(gK1*low,gK1*high)
gKsd  = cp.Uniform(gKs*low,gKs*high)
gKrd  = cp.Uniform(gKr*low,gKr*high)
gtod =  cp.Uniform(gto*low,gto*high)
gNad  = cp.Uniform(gNa*low,gNa*high)
gCald = cp.Uniform(gCal*low,gCal*high)
dist = cp.J(gK1d,gKsd,gKrd,gtod,gNad,gCald)

p = 2 # polynomial degree
Np = math.factorial ( nPar+ p ) / ( math.factorial (nPar) * math.factorial (p))
m = 6      # multiplicative factor
Ns = 500#m * Np # number of samples

pmin,pmax=2,6

print("Samples",Ns) 
print("Degree",p) 

#Sample the parameter distribution
samples = dist.sample(Ns,rule="latin_hypercube", seed=1234)




#Run model for true response
start = timeit.default_timer()
sols=[TTCellModel(sample).run() for sample in samples.T]
ads50=[sol["ADP50"] for sol in sols]
ads90=[sol["ADP90"] for sol in sols]
dVmaxs=[sol["dVmax"] for sol in sols]
vrest=[sol["Vrepos"] for sol in sols]


qoi={
      "ADP50":ads50,
     # "ADP90":ads90,
     # "Vrest":vrest,
     #"dVmax":dVmaxs,
     
     }

stop = timeit.default_timer()


print('Time to run model: ', stop - start) 







kws = {"fit_intercept": False}
models = {

    

    "lars": lm.Lars(normalize=False,**kws,eps=0.75,precompute=False),
    "least squares": lm.LinearRegression(**kws),
     "lasso lars": lm.LassoLars(alpha=0.1, **kws),
     "ridge": lm.Ridge(alpha=0.1, **kws),
     "bayesian ridge": lm.BayesianRidge(**kws),

  


  
}



        # pols=[]
        # for P in list(range(pmin,pmax+1,1)):  
        #     poly_exp = cp.generate_expansion(P, dist,rule="three_terms_recurrence")
        #     fitted_polynomial = cp.fit_regression (poly_exp,samples,dataset,model=model,retall=False)     
        #     loos[P-pmin]=calcula_loo(dataset,poly_exp,samples,model)
        #     print("{:e}".format(loos[P-pmin]),"\n")
        #     pols.append(fitted_polynomial)
        
        
        

for label, model in models.items():
    
    print("Beggining ", label)


##Adpative algorithm chooses best fit in deegree range
  
    
    for qlabel,dataset in qoi.items():
        
        
        loos= np.zeros((pmax-pmin+1))
        pols=[]
        for P in list(range(pmin,pmax+1,1)):  
            poly_exp = cp.generate_expansion(P, dist,rule="three_terms_recurrence")
            fitted_polynomial = cp.fit_regression (poly_exp,samples,dataset,model=model,retall=False)     
            loos[P-pmin]=calcula_loo(dataset,poly_exp,samples,model)
            print("{:e}".format(loos[P-pmin]),"\n")
            pols.append(fitted_polynomial)
        
        
        
        #Choose best fitted poly exp in degree range
        degreeIdx=loos.argmin()
        loo=loos[degreeIdx]
        fitted_polynomial=pols[degreeIdx]
        
        loos
        ##Calculate Sobol Error
        #s1f=np.array(sensitivity['S9'])
        #sms=Sobol()
        avgE=0#np.mean(abs(s1f- sms))
        maxE=0#np.max(abs(s1f- sms))
        
        #Caluclate Validation Error
        YPCE=[cp.call(fitted_polynomial,sample) for sample in X]
        nErr=np.mean((YPCE-Yval[qlabel])**2)/np.var(Yval[qlabel])
        
        
        row=[qlabel,label,degreeIdx+pmin,nErr,loo,maxE,avgE,Ns]
        writer.writerow(row)
        
    # Y=l.predict(X)
    # d=Yval["ADP90"]-Y
    # np.mean(d**2)/np.var(Yval["ADP90"])

    
    # # fig, axs = plt.subplots(2, 2)
    # # plt=axs[0,0]
    # # plt.set_title('ADP90')
    # # Ytrue=Yval[:,0]
    # # plt.scatter(Ytrue,YPCE)
    # # plt.plot(Ytrue,Ytrue,"black",linewidth=2)
    # # plt.plot()
    # # plt.set(xlabel="Y_true",ylabel="Y_pred")
    
   
    
    
    
# close the file
f.close()
    
    
    
    
    
    
