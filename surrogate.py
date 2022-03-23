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

import csv

def readF(fn):
    X=[]
    file = open(fn, 'r')
    for row in file:
        X.append([float(x) for x in row.split(',')])
    return X



def calcula_loo(y, poly_exp, samples):
    """ 
    LOO for a scalar quantity: y 
    """
    #print ("\n\nPerforming Leave One Out cross validation.")    
    nsamp = np.shape(y)[0]
    #print(nsamp)
    #print(y)

    deltas = np.empty(nsamp)
    samps = samples.T

    start = timeit.default_timer()

    for i in range(nsamp):
        indices = np.linspace(0,nsamp-1,nsamp, dtype=np.int32)
        indices = np.delete(indices,i)
        subs_samples = samps[indices,:].copy()
        subs_y =[ y[i] for i in (indices)]

        subs_poly = cp.fit_regression(poly_exp, subs_samples.T, subs_y)
        yhat = cp.call(subs_poly, samps[i,:])
        deltas[i] = ((y[i] - yhat))**2

    y_std = np.std(y)
    err = np.mean(deltas)/np.var(y)
    print("Err:",err)
    acc = 1.0 - np.mean(deltas)/np.var(y)
    
    
    stop = timeit.default_timer()
    print('Time to LOO: ', stop - start) 

    return acc


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




Y = readF("Yval.txt")
Yval = np.array(Y)
#Simulation Parameters
labels=["gK1","gKs","gKr","gto","gNa","gCal"]
TTCellModel.setParametersOfInterest(["gK1","gKs","gKr","gto","gNa","gCal"])
nPar=6
#size parameteres
ti=3000
tf=3500
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

p = 3 # polynomial degree
Np = math.factorial ( nPar+ p ) / ( math.factorial (nPar) * math.factorial (p))
m = 2  # multiplicative factor
Ns = m * Np # number of samples

print("Samples",Ns) 
print("Degree",p) 

#Sample the parameter distribution
samples = dist.sample(Ns,rule="latin_hypercube", seed=1234)
print(samples.shape)



#Run model for true response
start = timeit.default_timer()
sols=[TTCellModel(sample).run() for sample in samples.T]
stop = timeit.default_timer()


print('Time to run model: ', stop - start) 

#Build PCE and fit coefficients
poly_exp = cp.orth_ttr (p,dist)
ads50=[sol["ADP50"] for sol in sols]
ads90=[sol["ADP90"] for sol in sols]
dvMaxs=[sol["dVmax"] for sol in sols]
vrest=[sol["Vrepos"] for sol in sols]
surr_model50 = cp.fit_regression (poly_exp,samples,ads50)
surr_model90 = cp.fit_regression (poly_exp,samples,ads90)
surr_modeldvMax = cp.fit_regression (poly_exp,samples,dvMaxs)
surr_modelVrest = cp.fit_regression (poly_exp,samples,vrest)





start = timeit.default_timer()

#Calculate LOO, Sobol Error and Validation Error for each QOI




##Load Result File
f = open('results.csv', 'a')

# create the csv writer
writer = csv.writer(f)

# write a row to the csv file

#row=['QOI',	'Method', 'Degree','Val. error',' LOOERROR','Max Sobol Error','Mean Sobol Error','Ns']
#writer.writerow(row)


print("SOBOL + LOO ERRO CALC")
print("\n")


##AD50
s1f=np.array(sensitivity['S5'])
sms=cp.Sens_m (surr_model50,dist)
avgE=np.mean(abs(s1f- sms))
maxE=np.max(abs(s1f- sms))
loo=calcula_loo(ads50,poly_exp,samples)
YPCE=[cp.call(surr_model50,x) for x in X]
nErr=np.mean((YPCE-Yval[:,0])**2)/np.var(Yval[:,0])

row=['AD50','CPLS',p,nErr,loo,maxE,avgE,Ns]
writer.writerow(row)



##AD90
s1f=np.array(sensitivity['S5'])
sms=cp.Sens_m (surr_model50,dist)
avgE=np.mean(abs(s1f- sms))
maxE=np.max(abs(s1f- sms))
loo=calcula_loo(ads90,poly_exp,samples)
YPCE=[cp.call(surr_model50,x) for x in X]
nErr=np.mean((YPCE-Yval[:,1])**2)/np.var(Yval[:,1])

row=['AD90','CPLS',p,nErr,loo,maxE,avgE,Ns]
writer.writerow(row)

##AD50
s1f=np.array(sensitivity['S5'])
sms=cp.Sens_m (surr_model50,dist)
avgE=np.mean(abs(s1f- sms))
maxE=np.max(abs(s1f- sms))
loo=calcula_loo(ads50,poly_exp,samples)
YPCE=[cp.call(surr_modeldvMax,x) for x in X]
nErr=np.mean((YPCE-Yval[:,2])**2)/np.var(Yval[:,2])

row=['Vrest','CPLS',p,nErr,loo,maxE,avgE,Ns]
writer.writerow(row)

##AD50
s1f=np.array(sensitivity['S5'])
sms=cp.Sens_m (surr_model50,dist)
avgE=np.mean(abs(s1f- sms))
maxE=np.max(abs(s1f- sms))
loo=calcula_loo(ads50,poly_exp,samples)
YPCE=[cp.call(surr_modelVrest,x) for x in X]
nErr=np.mean((YPCE-Yval[:,3])**2)/np.var(Yval[:,3])

row=['dVmax','CPLS',p,nErr,loo,maxE,avgE,Ns]
writer.writerow(row)




stop = timeit.default_timer()


print("\n")
print('Time to run Sobol: + LOO ', stop - start) 








# close the file
f.close()






