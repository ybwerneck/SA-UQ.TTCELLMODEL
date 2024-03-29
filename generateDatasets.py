# -*- coding: utf-8 -*-
"""
Created on Sat May  7 00:17:43 2022

@author: yanbw
"""

import os
import numpy as np
import chaospy as cp
import timeit
from modelTT import TTCellModel
import math
import csv
import ray
import copy
from scipy.spatial import KDTree as kd
from utils import *
#DEF MODEL
labels=["gK1","gKs","gKr","gto","gNa","gCal"]
#MODEL WRAPPER
def Model(sample):
    TTCellModel.setParametersOfInterest(["gK1","gKs","gKr","gto","gNa","gCal"])
    #size parameteres
    ti=000
    tf=1000
    dt=0.01
    dtS=1
    size=TTCellModel.setSizeParameters(ti, tf, dt, dtS)  
    return TTCellModel(sample).run()

def generateDataset(nx,ny,folder,Ns,out):
    #PARAMETERS
    
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
    
    
    
    #Fitting parameters
    nPar=6
    #m * Np # number of samples
    Nv=21*Ns
    
    #normalization options
    normalizarqoi=ny
    normalizarX=nx
    normalizationMethod='normal'  #normal or skt
    
    
    if(True):
        print('Generating Sets')
        print("Samples",Ns)
        print("Samples Validation ",Nv)
        

    
    f=folder
    
    
    #Sample the parameter distributio
    start = timeit.default_timer()
    samples = dist.sample(Ns,rule="latin_hypercube")
    samplesaux=copy.copy(samples)
    stop = timeit.default_timer()
    if(out):        
       print('Time to sample Dist: ',stop-start)
    
    
    start = timeit.default_timer()
    samplesV = dist.sample(Nv ,rule="latin_hypercube")
    samplesVaux=copy.copy(samplesV)
    stop = timeit.default_timer()
    if(out):
        print('Time to sample Dist V: ',stop-start)
    
    
    #Run training set
    
    start = timeit.default_timer()
    sols=runModelParallel(samples,Model)
    stop = timeit.default_timer()
    
    ads50=[sols[i]["ADP50"] for i in range(Ns)]
    ads90=[sols[i]["ADP90"] for i in range(Ns)]
    dVmaxs=[sols[i]["dVmax"] for i in range(Ns)]
    vrest= [sols[i]["Vrepos"] for i in range(Ns)]
    
    qoi={
         "ADP50":ads50,
         "ADP90":ads90,
         "Vrest":vrest,
         "dVmax":dVmaxs,
         
         }
    
    
    print('Time to run Model training Set: ',stop-start)
    
    
    
    
    
    #Run validation set
    
    start = timeit.default_timer()
    sols=runModelParallel(samplesV,Model)
    stop = timeit.default_timer()
    print('Time to run Model Validation set: ',stop-start)
    
    ads50=[sols[i]["ADP50"] for i in range(Nv)]
    ads90=[sols[i]["ADP90"] for i in range(Nv)]
    dVmaxs=[sols[i]["dVmax"] for i in range(Nv)]
    vrest= [sols[i]["Vrepos"] for i in range(Nv)]
    
    qoiV={
         "ADP50":ads50,
         "ADP90":ads90,
         "Vrest":vrest,
         "dVmax":dVmaxs,
         
         }
    
    
    
    
    
    # #NORMALIZE
    
    for i in range(nPar): #even if the output is a non-normalized array, a normalized array is used for treating the set
        samples[i],samplesV[i]=normalizeTwoArrays(samples[i], samplesV[i],0,1, normalizationMethod)
    
    if(normalizarqoi):
         for qlabel in qoi:  
            qoi[qlabel],qoiV[qlabel]=normalizeTwoArrays(qoi[qlabel],qoiV[qlabel], 0,1,normalizationMethod)
    
    
    
    #Treat validation set
    if(out):
            
        print("Treating validation dataset")
        print("Removing closest points")
        
        
    
    #REMOVE CLOSES POINT IN VALIDATION SET TO EACH POINT IN TRAINING SET
    difs,i,retrys=np.zeros(Ns),0,0
    start = timeit.default_timer()
    idtoremv=np.zeros(Ns,dtype=int)-1
    svt=samplesV.T
    kdt=kd(samplesV.T)
    for sample in samples.T:
        t=0 # trys
        flag=True
        while(flag):
             hit,ii = kdt.query(sample,k=t+1) #find the closes point, if the point is already mark to be removed, select the next closest until find a point not marked
             if(t>=1):
                 ii=ii[t]
                 difs[i]=hit[t]
             else :
                 difs[i]=hit
                
             if(False==np.any(idtoremv==ii)):    
                 flag=False
                 break;
             else:
                 retrys=retrys+1
                 t=t+1
        
       
        idtoremv[i]=ii
        i=i+1    
    
    
    if(normalizarX==False):
        svt=np.delete(samplesVaux.T,idtoremv,0)
    else:
        svt=np.delete(svt,idtoremv,0)
        
    for ql,dt in qoiV.items():
        qoiV[ql]=np.delete(dt,idtoremv,0)
        
        
        
        
    if(out):
            
        print("AVG DISTANCE",np.mean(difs))
        print("MAX DISTANCE",np.max(difs))
        print("Min DISTANCE",np.min(difs))
        print("EXACT MATCHES",np.count_nonzero(difs<=0.01)) ##0.01 tolerance
        print("Retrys",retrys)
    samplesV=svt.T
    
    
    
    
    
    ##Write results
    
    
    try:   
        os.mkdir(f)
        
    except:
        if(out):
        
            print("Updating")
    
    try:
        os.mkdir(f+"/validation")
    except:
        if(out):
        
            print("Updating")
    
    file=open(f+"X.csv","w", newline='') 
    writer = csv.writer(file)
    if(normalizarX==False):
        samples=samplesaux
        
    for row in samples.T:
        writer.writerow(row)
    file.close()
    
    file=open(f+"/validation/"+"X.csv","w",newline='')
    writer = csv.writer(file)
    for row in samplesV.T:
        writer.writerow(row)
    file.close()
    
    
    
    for qlabel in qoi:
        Y=qoi[qlabel]
        Yval=qoiV[qlabel]
        file=open(f+qlabel+".csv","w", newline='')
        np.savetxt(file,Y, fmt='%.8f')
        file.close()
        file=open(f+"/validation/"+qlabel+".csv","w", newline='')
        np.savetxt(file,Yval, fmt='%.8f')
        file.close()
    
    
    
    
       
    
    




