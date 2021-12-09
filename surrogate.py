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



#Simulation size parameteres
ti=0
tf=1000
dt=0.01
dtS=1
size=TTCellModel.setSizeParameters(ti, tf, dt, dtS)  #returns excpeted size of simulation output given size parameters
Timepoints=TTCellModel.getEvalPoints()

#gPC method parameters
p = 2 # polynomial degree
Np = math.factorial (2 + p ) / ( math.factorial (2) * math.factorial (p))
m = 3 # multiplicative factor
Ns = m * Np # number of samples

#Parameters of Interest
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

dist = cp.J(gK1d,gKsd,gKrd,gNad)


samples = dist.sample(Ns)

sols=[TTCellModel(sample).run() for sample in samples.T]
evals=[sol[:,1] for sol in sols]

poly_exp = cp . orth_ttr (p , dist)

surr_model = cp . fit_regression ( poly_exp , samples , evals)

mean = cp.E ( surr_model , dist )
std = cp.Std ( surr_model , dist )
sm = cp.Sens_m ( surr_model , dist)
st = cp.Sens_t ( surr_model , dist )

plt.plot(Timepoints,mean, lw=2, color='blue', label='W (mean)')
plt.xlabel("tempo")
plt.ylabel("Variação no potencial")
plt.legend(loc='best')
plt.show()
