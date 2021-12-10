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


def calcula_loo(y, poly_exp, samples):
    nsamp = samples.shape[1]
    
    deltas = np.empty(nsamp)
    samps = samples.T

    for i in range(nsamp):
        indices = np.linspace(0,nsamp-1,nsamp, dtype=int)
        indices = np.delete(indices,i)
        subs_samples = samps[indices, :].copy()
        subs_y =[ y[i] for i in (indices)]

        subs_poly = cp.fit_regression(poly_exp, subs_samples.T, subs_y)
        yhat = cp.call(subs_poly, samps[i,:])
        deltas[i] = np.mean(abs(y[i] - yhat)) ##Delta E para o polinomio gerado excluindo cada sample
        
    y_std=np.zeros((y[0].shape[0]))
    
    for x in range(y[0].shape[0]):
        y_std[x]=np.std([y[i][x] for i in (range(nsamp))]) # STD for each poin in QOI
    
    acc = 1.0 - np.mean(deltas)/np.mean(y_std)  ## E= deltAS/nsamp, V= mean(STD in each QOI)
    return acc

labels={"gK1","gKs","gKr","gNa","gbna","gCal","gbca","gto"}


TTCellModel.setParametersOfInterest(["gK1","gKs","gKr","gNa","gbna","gCal","gbca","gto"])

nPar=8
#Simulation size parameteres
ti=0
tf=500
dt=0.01
dtS=1
size=TTCellModel.setSizeParameters(ti, tf, dt, dtS)  #returns excpeted size of simulation output given size parameters
Timepoints=TTCellModel.getEvalPoints()

#gPC method parameters
p = 3 # polynomial degree
Np = math.factorial ( nPar+ p ) / ( math.factorial (nPar) * math.factorial (p))
m = 3 # multiplicative factor
Ns = m * Np # number of samples

#U Parameters
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
gKsd  = cp.Uniform(gKs*0.9,gKs*1.1)
gKrd  = cp.Uniform(gKr*low,gKr*high)
gNad  = cp.Uniform(gNa*low,gNa*high)
gbnad = cp.Uniform(gbna*low,gbna*high)
gCald = cp.Uniform(gCal*low,gCal*high)
gbcad = cp.Uniform(gbca*low,gbca*high)
gtod =  cp.Uniform(gto*low,gto*high)
dist = cp.J(gK1d,gKsd,gKrd,gNad,gbnad,gCald,gbcad,gtod)

samples = dist.sample(Ns)
print(samples.shape)

sols=[TTCellModel(sample).run() for sample in samples.T]
evals=[sol[:,1] for sol in sols]

poly_exp = cp.orth_ttr (p,dist)

surr_model = cp.fit_regression (poly_exp,samples,evals)

print("LOO=")
print(calcula_loo(evals,poly_exp,samples))
mean = cp.E ( surr_model , dist )
std = cp.Std ( surr_model , dist )
sm = cp.Sens_m ( surr_model , dist)
st = cp.Sens_t ( surr_model , dist )

sms=[np.mean(sm.T[0][:,i]) for i,val in enumerate(labels)]
plt.plot(Timepoints,mean, lw=2, color='blue', label='W (mean)')
plt.fill_between(Timepoints,mean[:,0] - std[:,0],mean[:,0] + std[:,0], facecolor='blue', alpha=0.5, label='S (std)')
plt.xlabel("tempo")
plt.ylabel("Variação no potencial")
plt.legend(loc='best')
plt.show()

fig = plt.figure()
y_pos = np.arange(len(labels))
plt.bar(y_pos,sms)
plt.xticks(y_pos, labels)
plt.show()