# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 12:51:32 2021

@author: yanbw
"""


import subprocess 
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
import math
from math import factorial

def readF(fn):
    X=[]
    file = open(fn, 'r')
    for row in file:
        X.append([float(x) for x in row.split()])
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
    samps = samples

    for i in range(nsamp):
        indices = np.linspace(0,nsamp-1,nsamp, dtype=np.int32)
        indices = np.delete(indices,i)
        subs_samples = samps[indices, :].copy()
        subs_y =[ y[i] for i in (indices)]
        subs_poly = cp.fit_regression(poly_exp, subs_samples.T, subs_y)
        yhat = cp.call(subs_poly, samps[i,:])
        deltas[i] = ((y[i] - yhat))**2

    y_std = np.std(y)
    err = np.mean(deltas)/np.var(y)
    print("Err:",err)
    acc = 1.0 - np.mean(deltas)/np.var(y)
    return acc


if __name__ == "__main__":

    nPar = 6
    p = 5 # polynomial degree

    if(len(sys.argv)==2):
        p = int(sys.argv[1])
        print(p)

    X=readF("X.txt")
    samples=np.zeros((len(X),6))
    for i,sample in enumerate(X):       ##must be matrix not list
        for k,y in enumerate(sample):
            samples[i][k]=y

    Y = readF("Y.txt")
    Y = np.array(Y)

    print(len(X))
    print(len(Y))
    print("Polynomial Degree:",p)

    start = timeit.default_timer()

 
    Z1 = cp.Uniform(0.75, 1.25)   # q1
    Z2 = cp.Uniform(0.75, 1.25)   # q2
    Z3 = cp.Uniform(0.75, 1.25)   # q3
    Z4 = cp.Uniform(0.75, 1.25)   # q4
    Z5 = cp.Uniform(0.75, 1.25)   # q4
    Z6 = cp.Uniform(0.75, 1.25)   # q4    
    dist = cp.J(Z1,Z2,Z3,Z4,Z5,Z6)
    
    m =1
    Na = int(factorial(nPar+p)/(factorial(nPar)*factorial(p)))
    ns = (Na*m)
    weights = cp.generate_quadrature(1, dist, rule=("gaussian"))
    indices = np.random.choice(samples.shape[0], ns, replace=True)



    #print(indices)

    if(ns<200):
        ns=200
        new_samples = samples
        new_Y = Y
    else:
        new_samples = samples[indices]
        new_Y = Y[indices]

    print("Samples taken for fitting:",ns)

 #print(new_samples)
    #print(new_Y)
        
    poly_exp = cp.generate_expansion(p, dist,rule="three_terms_recurrence")

    #poly_exp = cp.expansion.stieltjes(p,dist)

    #surr_model = cp.fit_regression (poly_exp, samples.T, Y)
    surr_model = cp.fit_regression(poly_exp, new_samples.T, new_Y) ##linear regression fitting

    print("Base len :",len(surr_model))
    stop = timeit.default_timer()

    print('Time to build model: ', stop - start) 
    start = timeit.default_timer()


    #mean = cp.E ( surr_model , dist )
    #std = cp.Std ( surr_model , dist )
    #sm = cp.Sens_m ( surr_model , dist)
    #st = cp.Sens_t ( surr_model , dist )



    #print("Mean Value",mean)
    #print("STD Value",std)
    #print("FIRST ORDER SOBOL \n",sm)
    

    #print("Total SOBOL \n",st)
    stop = timeit.default_timer()

    print('Time analytics: ', stop - start) 
    
    X=readF("Xval.txt")
    samples=np.zeros((len(X),6))
    for i,sample in enumerate(X):       ##must be matrix not list
        for k,y in enumerate(sample):
            samples[i][k]=y

    Y = readF("Yval.txt")
    Y = np.array(Y)
    
    Ytrue=Y
    
    Y_pred=[cp.call(surr_model,sample) for sample in samples]
    plt.scatter(Ytrue,Y_pred)
    plt.plot(Ytrue,Ytrue,"black",linewidth=2)
    plt.plot()
    plt.xlabel("Y_true")
    plt.ylabel("Y_pred")
    plt.legend(loc='best')
    plt.show()
    
    Yshist = np.zeros((2,1000))
    for i,val in enumerate(Ytrue):
        Yshist[0][i]=Ytrue[i]
        Yshist[1][i]=Y_pred[i][0]
        
    plt.hist(Yshist[0],color=["blue"],bins=20,label=["True model Response"])
    plt.hist(Yshist[1],color=["orange"],bins=20,label=["PCE prediction"])
    plt.xlim(50,210)
    plt.legend(loc='best')
    plt.xlabel("Y")
    plt.ylabel("Counts'")
    plt.show()
    
    start = timeit.default_timer()

    #loo = calcula_loo(Y, poly_exp, samples)
    loo = calcula_loo(new_Y, poly_exp, new_samples)
    
    print("LOO",loo)
    stop = timeit.default_timer()

    print('Time loo', stop - start) 
   
