# -*- codingarg: utf-8 -*-
"""
Created on Wed Aug 21 14:04:02 2019

@author: hnpan
"""

import matplotlib
matplotlib.use('Agg')
#from mpi4py import MPI
import numpy as np
import TwoLead_module_2
import sys
import re
import matplotlib.pyplot as plt
arg={'t':1,'alpha':5,'Vz':1,'Delta_0':0.2,'mu':0.2,'mu_lead':10,'wireLength':100,'Nbarrier':2,'Ebarrier':2,'gamma':0.2,'lamd':0.2,'voltage':0,'varymu':'no','SE':'no','QD':'no','QD2':'no'}

TwoLead_module_2.conductance(arg)