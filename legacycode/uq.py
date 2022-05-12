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

#Choosing Parameters of Interest

TTCellModel.setParametersOfInterest(["gK1","gKs","gKr","gto","gNa","gCal"])



#Simulation size parameteres


ti=0
tf=500
dt=0.01
dtS=1
size=int(TTCellModel.setSizeParameters(ti, tf, dt, dtS)[0])  #returns excpeted size of simulation output given size parameters
Timepoints=TTCellModel.getEvalPoints()
Ns = 100

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
gNad  = cp.Uniform(gNa*0.3,gNa*1.7)
gCald = cp.Uniform(gCal*low,gCal*high)
dist = cp.J(gK1d,gKsd,gKrd,gtod,gNad,gCald)


nPar=6
p = 4 # polynomial degree
Np = math.factorial ( nPar+ p ) / ( math.factorial (nPar) * math.factorial (p))
m = 2  # multiplicative factor
Ns = m * Np # number of samples

samples = dist.sample(Ns)
sols=[TTCellModel(sample).run() for sample in samples.T]
dvMaxstrainer=[sol["dVmax"] for sol in sols]
poly_exp = cp.orth_ttr (p,dist)
surr_modeldvMax = cp.fit_regression (poly_exp,samples,dvMaxstrainer)



start1 = timeit.default_timer()
samples = dist.sample(1500)
sols=[cp.call(surr_modeldvMax,sample) for sample in samples.T]
dvMaxsSurr=sols

stop1 = timeit.default_timer()

print('Time surr model', stop1 - start1) 
    
start1 = timeit.default_timer()

samples = dist.sample(1500)
sols=[TTCellModel(sample).run() for sample in samples.T]
dvMaxsModel=[sol["dVmax"] for sol in sols]
stop1 = timeit.default_timer()

print('Time True model', stop1 - start1) 
print("Surr Model", np.mean(dvMaxsSurr),np.std(dvMaxsSurr))
print("True Model", np.mean(dvMaxsModel),np.std(dvMaxsModel))








