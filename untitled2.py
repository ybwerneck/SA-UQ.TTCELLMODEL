# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 01:28:15 2022

@author: yanbw
"""

from surrogatefromfile import surrogatefromfile as sol
from generateDatasets import generateDataset as generate_prob
import os
import utils
inst={
      "qoinn":((True,False)),
      "xnn":((False,True)),
      "nn":((False,False)),
      "n":((True,True))
      }


Ns=500

folder="datasets/auto/"+str(Ns)+"/"


utils.init()
for i in inst:
    xn=inst[i][0]
    yn=inst[i][1]
    try:
        os.mkdir(folder+i)
    except:
        print("updating")
    
    generate_prob(xn, yn, folder+i+"/", Ns,False)
    sol(xn,yn,folder+i+"/",Ns)