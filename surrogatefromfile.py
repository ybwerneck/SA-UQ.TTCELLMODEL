# -*- coding: utf-8 -*-
"""
Created on Mon May  9 12:40:26 2022

@author: yanbw
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 02:58:23 2021

@author: yanbw
"""
import os


import numpy as np
import matplotlib.pyplot as plt
import chaospy as cp
import timeit
from sklearn import linear_model as lm
import csv
from utils import runModelParallel as runModel
import utils
#MODEL WRAPPER -> PCE MODEL
def ModelPCE(exp):
    def Model(sample): 
        return cp.call(exp,sample)
    return Model




#Load validation files

##Reference Value for Sobol Indexes Error calculation

Ns = 50
pmin,pmax=2,4

qoi={
     "ADP50",
     #"ADP90",
     #"Vrest",
     #"dVmax"
     
}

folder="datasets/teste/"+str(Ns)+"/"

try:
    os.mkdir(folder+"results/")
except:
    print("Updating results")
    
    
    
    
    
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


##Load Result File 
f = open(folder+'/results/numeric.csv', 'w',newline='')


# create the csv writer
writer = csv.writer(f)

row=['QOI',	'Method', 'Degree','Val. error',' LOOERROR','Max Sobol Error','Mean Sobol Error','Ns','Timeselected','Timemax','Timeselected','Timemax']
writer.writerow(row)

#Load datasets

#Training Samples
nPar=6
samples=np.empty((Ns,nPar))
X=utils.readF(folder+"X.csv")
samplesVal=np.zeros((len(X),6))
for i,sample in enumerate(X):       ##must be matrix not list
    for k,y in enumerate(sample):
        samples[i][k]=y
        
Y={}
for qlabel in qoi:
    Y[qlabel]=utils.readF(folder+qlabel+".csv")



#Validation Samples

Xv=utils.readF(folder+"validation/"+"X.csv")
samplesVal=np.zeros((len(Xv),6))
for i,sample in enumerate(X):       ##must be matrix not list
    for k,y in enumerate(sample):
        samplesVal[i][k]=y

Yval={}
for qlabel in qoi:
    Yval[qlabel]=utils.readF(folder+"validation/"+qlabel+".csv")








#Sample the parameter distribution





alpha=1
eps=0.75
kws = {"fit_intercept": False,"normalize":False}
models = {

    
     "OLS CP": None,
   
     "LARS": lm.Lars(**kws,eps=eps),
     "OMP"+str(alpha):
         lm.OrthogonalMatchingPursuit(n_nonzero_coefs=3, **kws),
    "OLS SKT": lm.LinearRegression(**kws),
    "ridge"+str(alpha): lm.Ridge(alpha=alpha, **kws),
    # "bayesian ridge": lm.BayesianRidge(**kws),
    "elastic net "+str(alpha): lm.ElasticNet(alpha=alpha, **kws),
    "lasso"+str(alpha): lm.Lasso(alpha=alpha, **kws),
    #"lasso lars"+str(alpha): lm.LassoLars(alpha=alpha, **kws),
    
    

  
}

##
pltxs=2
pltys=0

while(pltys*pltxs<len(models)):
    pltys=pltys+1
    



for qlabel,dataset in Y.items():
    print('\n',"QOI: ", qlabel,'\n')      
##Adpative algorithm chooses best fit in deegree range
    timeL=0
    
    fig,plotslot=plt.subplots(pltxs,pltys)
    plotsaux=[]
    plots=[]
    
    
    try:
        for row in plotslot:
            for frame in row:
                plotsaux.append(frame)
    except:
        for frame in plotslot:
            plotsaux.append(frame)
        
    
    for i in range(0,len(plotsaux)):
        plots.append(plotsaux.pop())
    pltidx=0
    fig.suptitle(qlabel)
    for label, model in models.items():
        
        print('\n--------------',"\n")
        print("Beggining ", label)
        loos= np.zeros((pmax-pmin+1))
        gF= np.zeros((pmax-pmin+1))
        timeL= np.zeros((pmax-pmin+1))
        
        pols=[]
        for P in list(range(pmin,pmax+1,1)):             
            print('\n')
            print('D=',P)
            #generate and fit expansion            
            start = timeit.default_timer()
            poly_exp = cp.generate_expansion(P, dist,rule="three_terms_recurrence")
            fp = cp.fit_regression (poly_exp,samples.T,dataset,model=model)  
            stop = timeit.default_timer()
            time=stop-start
            print('Time to generate exp: ',time) 
            #calculate loo error
            gF[P-pmin]=time
            start = timeit.default_timer()
            loos[P-pmin]=utils.calcula_loo(dataset,poly_exp,samples.T,model)
            stop = timeit.default_timer()
            timeL[P-pmin]=stop-start
            print('Time to LOO: ',timeL[P-pmin],'LOO: ',loos[P-pmin]) 


            pols.append(fp)
            
            print('\n')
        
        
        #Choose best fitted poly exp in degree range->lowest loo error
        degreeIdx=loos.argmin()
        loo=loos[degreeIdx]
        fitted_polynomial=pols[degreeIdx]
        ##
        print('AA picked D= ',degreeIdx+pmin," Generate Validation Results") 
        
        ##Calculate Sobol Error
        #s1f=np.array(sensitivity['S9'])
        #sms=Sobol()
        avgE=0#np.mean(abs(s1f- sms))
        maxE=0#np.max(abs(s1f- sms)) 
        
        #Caluclate Validation Error
        start = timeit.default_timer()
        #YPCE=[cp.call(fitted_polynomial,sample) for sample in Xv]
        YPCE=runModel(np.array(Xv).T,ModelPCE(fitted_polynomial))
        YPCE=np.array([YPCE[idxl] for idxl in (YPCE)]).flatten()
        YVAL=np.array(Yval[qlabel]).flatten()
        YPCE=np.array(YPCE)
        nErr=np.mean((YPCE-YVAL)**2)/np.var(YVAL)
        
        
        stop = timeit.default_timer()
        time=stop-start
        print('Time to Validate: ',time)   
        row=[qlabel,label,degreeIdx+pmin,              
        f"{nErr:.2E}",f"{loo:.2E}",maxE,avgE,Ns,timeL[degreeIdx],timeL[timeL.argmax()],gF[degreeIdx],gF[gF.argmax()]]
        writer.writerow(row)       
        print('--------------',"\n")
        
        ##PLOT RESULTS
        
        p=plots.pop()
        pltidx=pltidx+1
        p.set_title(label)
       
        p.plot(YVAL,YVAL,"black",linewidth=2)
        p.scatter(YVAL,YPCE)
     
        p.set(xlabel="Y_true",ylabel="Y_pred")
        p.get_figure().savefig(folder+"results/"+qlabel+"_validation_results.png")
    
    
    
# close the file
f.close()
    