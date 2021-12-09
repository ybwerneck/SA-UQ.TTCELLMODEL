# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 22:44:43 2021

@author: yanbw
"""

from modelTT import TTCellModel


##Simple model use, usefull for
ti=0
tf=10000
dt=0.01
dtS=1
TTCellModel.setSizeParameters(ti, tf, dt, dtS)
model=TTCellModel("")
model.plot_sir(model.run(),"W")