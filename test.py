import numpy as np
import Majorana_module as Maj
import time

parameters = {'isTV':0,'a':1,'mu':.2,'alpha_R':5, 'delta0':0.2,'wireLength':1000,
             'muLead':25.0, 'barrierNum':2,'barrierE':10.0, 'dissipation':0.0001,'isDissipationVar':0,
             'isQD':0, 'qdPeak':0.4, 'qdLength':20, 'qdPeakR':0,'qdLengthR':0,
             'isSE':0, 'couplingSCSM':0.2, 'vc':0,
             'potType':0,'potPeakPos':0,'potSigma':1,'potPeak':0,'potPeakR':0,'potPeakPosR':0,'potSigmaR':0,
             'muVar':0,'muVarList':0,'muVarType':0,'scatterList':0,'N_muVar':1,
             'gVar':0,'randList':0,
             'deltaVar':0,
             'vz':0.0, 'vBias':0.0,'vBiasMin':-0.3,'vBiasMax':0.3,'vBiasNum':1001,
             # 'vz0':0,'vzNum':256,'vzStep': 0.002,'mu0':0,'muMax':1,'muStep':0.002,'muNum':0,
             'leadPos':0,'leadNum':1,
             'Q':0,
             'x':'vz','xMin':0,'xMax':2.048,'xNum':256,'xUnit':'meV',
             'alpha':-1,
             'error':0}

junction=Maj.make_NS_junction(parameters)
rs=Maj.conductance(parameters,junction)
