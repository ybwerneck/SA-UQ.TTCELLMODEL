# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 19:00:50 2022

@author: yanbw
"""

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
TTCellModel.setParametersOfInterest(["gNa"])


#Simulation size parameteres


ti=0
tf=500
dt=0.01
dtS=1
size=int(TTCellModel.setSizeParameters(ti, tf, dt, dtS)[0])  #returns excpeted size of simulation output given size parameters
Timepoints=TTCellModel.getEvalPoints()
Ns = 100

#Parameters of I
gK1=  5.4050e+00
gKs= 0.245
gKr = 0.096
gNa = 1.48380e+01
gbna =  2.90e-04
gCal =   1.750e-04
gbca = 5.920e-04
gto =  2.940e-01

low=0.3
high=1.1


gNa  = cp.Uniform(gNa*low,gNa*high)

dist = cp.J(gNa)


samples = dist.sample(Ns)

sols=[TTCellModel(sample).run() for sample in samples.T]
evals=[sol["Wf"][:,1] for sol in sols]
dtmaxs=[sol["dVmax"] for sol in sols]
exp_s = np.zeros(size)
var_s = np.zeros(size)
tss= np.zeros(size)

