import matplotlib
matplotlib.use('Agg')
from mpi4py import MPI
import numpy as np
import Majorana_module as Maj
import sys
import re
import matplotlib.pyplot as plt

NS_dict = {'TV':1,'a':1,'mu':.2,'mumax':1,'alpha_R':5, 'Delta_0':0.2,'Delta_c':0.2,'epsilon':1,'wireLength':100, 'mu_lead':25.0, 'Nbarrier':2,'Ebarrier':10.0, 'Gamma':0.0001, 'QD':0, 'VD':0.4, 'dotLength':20, 'SE':'no', 'Vz':0.0, 'voltage':0.0,'smoothpot':0, 'gamma':0,'multiband':0,'leadpos':0,'peakpos':.25,'sigma':1};
NS_dict['Vz']=.36;
junction=Maj.NSjunction(NS_dict);
Maj.conductance(NS_dict,junction);
print(Maj.conductance(NS_dict,junction))