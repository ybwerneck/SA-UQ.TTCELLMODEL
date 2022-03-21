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

sensitivity={}
sensitivity['S5']=[0.01686099 , 0.17251719,  0.09176353 , -0.00206405,  0.0007281,   0.71546501 ]
sensitivity['S9']=[ 0.05775064 , 0.16151245 ,  0.11953077 ,  0.003076  ,  0.00179271 , 0.66013521 ]
sensitivity['SVM']=[1.26269597e-04, 4.98022874e-06, 8.04921510e-05, 1.35415023e-05, 9.96958800e-01, 2.53308786e-05 ]
sensitivity['SVR']=[7.85707445e-01 , 2.84286540e-03,  1.55199393e-03,  4.51826826e-04,
 -1.20794094e-04,  2.29738177e-01]



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


labels=["gK1","gKs","gKr","gto","gNa","gCal"]

TTCellModel.setParametersOfInterest(["gK1","gKs","gKr","gto","gNa","gCal"])


nPar=6
#Simulation size parameteres
ti=3000
tf=3400
dt=0.01
dtS=1
size=TTCellModel.setSizeParameters(ti, tf, dt, dtS)  #returns excpeted size of simulation output given size parameters
Timepoints=TTCellModel.getEvalPoints()

#gPC method parameters
p = 2 # polynomial degree
Np = math.factorial ( nPar+ p ) / ( math.factorial (nPar) * math.factorial (p))
m = 2  # multiplicative factor
Ns = m * Np # number of samples

#U Parameters
#Variaveis de Interesse
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

print("Samples",Ns) 
print("Degree",p) 

samples = dist.sample(Ns,rule="latin_hypercube", seed=1234)
print(samples.shape)


start = timeit.default_timer()

sols=[TTCellModel(sample).run() for sample in samples.T]

stop = timeit.default_timer()

print('Time to run model: ', stop - start) 


poly_exp = cp.orth_ttr (p,dist)

ads50=[sol["ADP50"] for sol in sols]
ads90=[sol["ADP90"] for sol in sols]
dvMaxs=[sol["dVmax"] for sol in sols]
vrest=[sol["Vrepos"] for sol in sols]
surr_model50 = cp.fit_regression (poly_exp,samples,ads50)
surr_model90 = cp.fit_regression (poly_exp,samples,ads90)
surr_modeldvMax = cp.fit_regression (poly_exp,samples,dvMaxs)
surr_modelVrest = cp.fit_regression (poly_exp,samples,vrest)





#print(calcula_loo(evals,poly_exp,samples))


#mean = cp.E ( surr_model , dist )
#std = cp.Std ( surr_model , dist )
#sm = cp.Sens_m ( surr_model , dist)
#st = cp.Sens_t ( surr_model , dist )



start = timeit.default_timer()


##Calc dif Sobol

print("SOBOL + LOO ERRO CALC")

#print("FX")
#stf=np.array(sensitivity['STf'])
#s1f=np.array(sensitivity['S1f'])3
#sms=[np.mean(sm.T[0][:,i]) for i,val in enumerate(labels)]
#sts=[np.mean(st.T[0][:,i]) for i,val in enumerate(labels)]
#print("S1")

#print("MAX ERR:",(np.max(abs(s1f- sms))))
#print("MEAN ERR:",np.mean(abs(s1f- sms)))


#fig = plt.figure()
#y_pos = np.arange(len(labels))
#plt.bar(y_pos,sms)
#plt.xticks(y_pos, labels)
#plt.xlabel("SOBOL INDEX FOR WAVEFORM")

plt.show()


print("\n")
print("AD50")




s1f=np.array(sensitivity['S5'])
sms=cp.Sens_m (surr_model50,dist)

print("MAX ERR:",(np.max(abs(s1f- sms))))
print("MEAN ERR:",np.mean(abs(s1f- sms)))



fig = plt.figure()
y_pos = np.arange(len(labels))
plt.bar(y_pos,sms)
plt.xticks(y_pos, labels)
plt.xlabel("SOBOL INDEX FOR AD50")
plt.show()
print("LOOad50 =",calcula_loo(ads50,poly_exp,samples))


print("\n")
print("AD90")




s1f=np.array(sensitivity['S9'])
sms=cp.Sens_m (surr_model90,dist)

print("MAX ERR:",(np.max(abs(s1f- sms))))
print("MEAN ERR:",np.mean(abs(s1f- sms)))



fig = plt.figure()
y_pos = np.arange(len(labels))
plt.bar(y_pos,sms)
plt.xticks(y_pos, labels)
plt.xlabel("SOBOL INDEX FOR AD90")
plt.show()

print("LOO ad90 =",calcula_loo(ads90,poly_exp,samples))



print("\n")
print("dvMax")



s1f=np.array(sensitivity['SVM'])
sms=cp.Sens_m (surr_modeldvMax,dist)

print("MAX ERR:",(np.max(abs(s1f- sms))))
print("MEAN ERR:",np.mean(abs(s1f- sms)))



fig = plt.figure()
y_pos = np.arange(len(labels))
plt.bar(y_pos,sms)
plt.xticks(y_pos, labels)
plt.xlabel("SOBOL INDEX FOR DVMax")
plt.show()
print("LOO dvmax =",calcula_loo(dvMaxs,poly_exp,samples))


print("\n")
print("vRest")





s1f=np.array(sensitivity['SVR'])
sms= cp.Sens_m (surr_modelVrest,dist)

print("MAX ERR:",(np.max(abs(s1f- sms))))
print("MEAN ERR:",np.mean(abs(s1f- sms)))




fig = plt.figure()
y_pos = np.arange(len(labels))
plt.bar(y_pos,sms)
plt.xticks(y_pos, labels)
plt.xlabel("SOBOL INDEX FOR Rest pontetial")
plt.show()


print("LOO vrest =",calcula_loo(vrest,poly_exp,samples))

stop = timeit.default_timer()


print("\n")
print('Time to run Sobol: + LOO ', stop - start) 















