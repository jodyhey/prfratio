"""
    reads a file with SFSs 
    runs estimators
    not current as of 8/23/2023
"""
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.optimize import minimize,minimize_scalar
from scipy.stats import chi2
# import  SFS_functions
import  SFS_functions_trimmed as SFS_functions 
import math
import warnings
warnings.filterwarnings("ignore")
import string

def getsfs(fn):
    lines = open(fn,'r').readlines()
    sfs = [0]
    for line in lines:
        if len(line) > 1 and line[0] in string.digits :
            sfs.append(int(line.strip()))
    n = len(sfs)
    sfsfolded = [0]
    for i in range(1,1+n//2):
        # print(i,n-i,sfs[i],sfs[n-i])
        # print(i,n-i)
        sfsfolded.append(sfs[i]+sfs[n-i] if (n-i != i) else sfs[i])
    return sfs,sfsfolded

def getratio(nsfs,ssfs):
    r = []
    for i,val in enumerate(nsfs):
        r.append(math.inf if val==0.0 else ssfs[i]/val)
    return r
nsfs,nsfsfolded = getsfs("neutral_cfsfs_1_-0.02.txt")
ssfs,ssfsfolded = getsfs("selected_cfsfs_1_-0.02.txt")
ratios = getratio(nsfsfolded,ssfsfolded)

n = 80
thetastart = 1000.0
gstart = -1.0
dofolded = True #False
if dofolded:
    ratios = getratio(nsfsfolded,ssfsfolded)
    neu_thetagresult = minimize(SFS_functions.NegL_SFS_Theta_Ns,np.array([thetastart,gstart]),args=(n,dofolded,nsfsfolded),method="Powell",bounds=[(thetastart/10,thetastart*10),(10*gstart,-10*gstart)])
    sel_thetagresult = minimize(SFS_functions.NegL_SFS_Theta_Ns,np.array([thetastart,gstart]),args=(n,dofolded,ssfsfolded),method="Powell",bounds=[(thetastart/10,thetastart*10),(10*gstart,-10*gstart)])
    print("Basic Fisher-Wright PRF model\n\tNeutral folded SFS: theta estimate {:.4f}  g estimate {:.4f}\n\t"
        "Selected folded SFS: theta estimate {:.4f}  g estimate {:.4f}".format(neu_thetagresult.x[0],neu_thetagresult.x[1],sel_thetagresult.x[0],sel_thetagresult.x[1]) )    
else:
    ratios = getratio(nsfs,ssfs)
    neu_thetagresult = minimize(SFS_functions.NegL_SFS_Theta_Ns,np.array([thetastart,gstart]),args=(n,dofolded,nsfs),method="Powell",bounds=[(thetastart/10,thetastart*10),(10*gstart,-10*gstart)])
    sel_thetagresult = minimize(SFS_functions.NegL_SFS_Theta_Ns,np.array([thetastart,gstart]),args=(n,dofolded,ssfs),method="Powell",bounds=[(thetastart/10,thetastart*10),(10*gstart,-10*gstart)])    
    print("Basic Fisher-Wright PRF model\n\tNeutral unfolded SFS: theta estimate {:.4f}  g estimate {:.4f}\n\t"
        "Selected unfolded SFS: theta estimate {:.4f}  g estimate {:.4f}".format(neu_thetagresult.x[0],neu_thetagresult.x[1],sel_thetagresult.x[0],sel_thetagresult.x[1]) )    


ratiothetagresult =  minimize(SFS_functions.NegL_SFSRATIO_Theta_Ns,np.array([thetastart,thetastart,gstart]),args=(n,dofolded,ratios,False),method="Powell",bounds=[(thetastart/10,thetastart*10),(thetastart/10,thetastart*10),(10*gstart,-10*gstart)])
ratiothetag0result = minimize(SFS_functions.NegL_SFSRATIO_Theta_Ns,np.array([thetastart,thetastart]),args=(n,dofolded,ratios,True),method="Powell",bounds=[(thetastart/10,thetastart*10),(thetastart/10,thetastart*10)])
thetagdelta = 2*(-ratiothetagresult.fun + ratiothetag0result.fun)
print("Ratio likeliood method\n\tFull model: Theta neutral estimate {:.4f} Theta selected estimate {:.4f} g estimate {:.4f}\n\t"
      "Nested neutral model (g=0): Theta neutral estimate {:.4f} Theta selected estimate {:.4f}\n\t -2Deltag (chi^2 value): {:.4f}"
      .format(ratiothetagresult.x[0],ratiothetagresult.x[1],ratiothetagresult.x[2],ratiothetag0result.x[0],ratiothetag0result.x[1],thetagdelta))
pass