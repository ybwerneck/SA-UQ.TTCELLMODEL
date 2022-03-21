# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 14:26:55 2021

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

TTCellModel.setParametersOfInterest(["gK1","gKs","gKr","gto","gNa","gCal"])

#Variaveis de Interesse
gK1=  5.4050e+00
gKs= 0.245
gKr = 0.096
gNa = 1.48380e+01
gbna =  2.90e-04
gCal =   1.750e-04
gbca = 5.920e-04
gto =  2.940e-01


#Simulation size parameters
ti=10000
tf=10500
dt=0.01
dtSave=1
size=TTCellModel.setSizeParameters(ti, tf, dt,dtSave)[0]

#Sobol
start = timeit.default_timer()
print("COMEÇANDO SOBOL")
nsamples=100
problem = {
    "num_vars": 6,
    "names": [ "gK1","gKs","gKr","gto","gNa","gCal"],
    "bounds": [ [gK1*0.9,gK1*1.1], [gKs*0.9,gKs*1.1], [gKr*0.9,gKr*1.1],[gto*0.9,gto*1.1],[gNa*0.9,gNa*1.1],[gCal*0.9,gCal*1.1]],
}
param_vals = saltelli.sample(problem,nsamples,  calc_second_order=True)
Ns = param_vals.shape[0]
Y = np.empty([Ns])
print("N=")
print(Ns)
ad90s=np.empty([Ns])
ad50s=np.empty([Ns])
dVMaxs=np.empty([Ns])
vrest=np.empty([Ns])
for i in range(Ns):
     # resolve o problema SIR
    model=TTCellModel(param_vals[i]);
    r=model.run()
    sol = r["Wf"][:,1]
    Y[i] = sol[size-1]
    if(i%20==0):
        print(i/Ns)
    try:
        ad90s[i]=r["ADP90"]
        ad50s[i]=r["ADP50"]
        dVMaxs[i]=r["dVmax"]
        vrest[i]=r["Vrepos"]
    except:
        ad90s[i]=0
        ad50s[i]=0



# ADPS E VEL DE DEPOLARIZAÇÃO
# ADP50
sensitivity = sobol.analyze(problem,ad50s , calc_second_order=True)
print("Indice de Sobol princial or de primeira ordem relacionado a AD50")
print(sensitivity['S1'])
print("Indice de alta ordem")
print(sensitivity['ST'])
print()

# ADP90
sensitivity = sobol.analyze(problem,ad90s , calc_second_order=True)
print("Indice de Sobol princial or de primeira ordem relacionado a AD90")
print(sensitivity['S1'])
print("Indice de de alta ordem")
print(sensitivity['ST'])
print()

# dvmax
sensitivity = sobol.analyze(problem, dVMaxs, calc_second_order=True)
print("Indice de Sobol princial or de primeira ordem relacionado a dvMax")
print(sensitivity['S1'])
print("Indice de Sobol total ou de alta ordem")
print(sensitivity['ST'])
print()


# vrest
sensitivity = sobol.analyze(problem, vrest , calc_second_order=True)
print("Indice de Sobol princial or de primeira ordem relacionado a Vrest")
print(sensitivity['S1'])
print("Indice de Sobol total ou de alta ordem")
print(sensitivity['ST'])
print()

stop = timeit.default_timer()
print('Time ALL RUN: ', stop - start) 


