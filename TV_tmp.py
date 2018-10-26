import matplotlib
matplotlib.use('Agg')
import numpy as np
import Majorana_module as Maj
import sys
import re
import matplotlib.pyplot as plt

np.warnings.filterwarnings('ignore');
NS_dict = {'a':1,'mu':1,'mumax':4,'alpha_R':5, 'Delta_0':0.2,'Delta_c':0.2,'epsilon':1,'wireLength':1000, 'mu_lead':25.0, 'Nbarrier':2,'Ebarrier':10.0, 'Gamma':0.0001, 'QD':'no', 'VD':0.4, 'dotLength':20, 'SE':'no', 'Vz':0.0, 'voltage':0.0,'smoothpot':'lorentzsigmoid', 'gamma':0,'multiband':0,'leadpos':1,'peakpos':.25};
tot=256;
VzStep = 0.002*8; 
tv=np.zeros(tot);
for ii in range(tot):
   NS_dict['Vz'] = (ii)*VzStep;
   tv[ii]=Maj.TV(NS_dict)
        
