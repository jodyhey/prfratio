"""
    generate ROC curves for basic poisson random field test of selection
    and for likelihood test of ratios of neutral and selected SFS
    user must set these variables:
    dobasicPRFROC =  True # create ROC for basic poisson random field likelihood ratio test with and without selection
    doratioPRFROC = True # create ROC for ratio of poisson variables likelihood ratio test with and without selection 
    dofolded = True # use folded distribution
    theta = 200
    n = 50 # sample size 
"""
import sys
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.optimize import minimize,minimize_scalar
from scipy.stats import chi2
import  SFS_functions
import math
import argparse
import warnings
warnings.filterwarnings("ignore")

def valstoscreenandfile(fn,thresholds,pvalues,TPrate,FPrate,args,writetoscreen=False):
    if writetoscreen:
        print("SFS_ROC.py values for theta: {}  max2Ns: {} sample size: {}".format(args.theta,args.max2Ns,args.n))
        print("{}\nT\tp\tTPrate\tFPrate".format(fn))
        for i in range(len(thresholds)):
            print("{:.2g}\t{:.5f}\t{:.4f}\t{:.4f}".format(thresholds[i],pvalues[i],TPrate[i],FPrate[i]))
    if args.outfilename != None:
        f = open(args.outfilename,'a')
        f.write("\nSFS_ROC.py values for theta: {}  max2Ns: {} sample size: {}\n".format(args.theta,args.max2Ns,args.n))
        f.write("{}\nT\tp\tTPrate\tFPrate".format(fn) + "\n")
        for i in range(len(thresholds)):
            f.write("{:.2g}\t{:.5f}\t{:.4f}\t{:.4f}".format(thresholds[i],pvalues[i],TPrate[i],FPrate[i]) + "\n")
        f.write("\n")
        f.close()

def run(args):
    np.random.seed(args.seed)
    dobasicPRFROC = True # create ROC for basic poisson random field likelihood ratio test with and without selection
    doratioPRFROC = True # create ROC for ratio of poisson variables likelihood ratio test with and without selection 
    dofolded = True # use folded distribution
    theta = args.theta
    n = args.n # sample size 
    max2Ns = args.max2Ns
    ntrials = 1000
    foldstring = "folded" if dofolded else "unfolded"
    pvalues = [1,0.999, 0.995, 0.99, 0.9, 0.5, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 
        0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001,0]
    x2thresholds = [0.0,1.5708e-6, 0.0000392704, 0.000157088, 0.0157908, 0.454936, 
            2.70554, 3.84146, 5.41189, 6.6349, 7.87944, 9.54954, 10.8276, 
            12.1157, 13.8311, 15.1367, 16.4481, 18.1893, 19.5114,math.inf]
    x2thresholds.reverse()
    pvalues.reverse()
    writetxt = args.outfilename != None
    if dobasicPRFROC:
        numg0 = ntrials//2
        rocfilename = 'basicPRF_ROC_theta{}_gmax{}_n{}_{}.pdf'.format(theta,max2Ns,n,foldstring)
        gvals_and_results = [[0.0,0] for i in range(numg0)] # half the g=2Ns values are 0
        for i in range(ntrials-numg0):
            gvals_and_results.append([np.random.random()*max2Ns,1]) # half the g=2Ns values are nonzero 
        for i in range(ntrials):
            g = gvals_and_results[i][0]
            sfs,sfsfolded = SFS_functions.simsfs(theta,g,n,None, False)
            thetastart = 100.0
            gstart = -1.0
            if dofolded:
                thetagresult = minimize(SFS_functions.NegL_SFS_Theta_Ns,np.array([thetastart,gstart]),args=(n,dofolded,sfsfolded),method="Powell",bounds=[(thetastart/10,thetastart*10),(10*gstart,-10*gstart)])
                g0result = minimize_scalar(SFS_functions.NegL_SFS_Theta_Ns,bracket=(thetastart/10,thetastart*10),args = (n,dofolded,sfsfolded),method='Brent')
            else:
                thetagresult = minimize(SFS_functions.NegL_SFS_Theta_Ns,np.array([thetastart,gstart]),args=(n,dofolded,sfs),method="Powell",bounds=[(thetastart/10,thetastart*10),(10*gstart,-10*gstart)])
                g0result = minimize_scalar(SFS_functions.NegL_SFS_Theta_Ns,bracket=(thetastart/10,thetastart*10),args = (n,dofolded,sfs),method='Brent')

            thetagdelta = 2*(-thetagresult.fun + g0result.fun)
            # if g < 0:
            #     print(thetagdelta)
            for t in x2thresholds:
                if thetagdelta > t:
                    gvals_and_results[i].append(1)
                else:
                    gvals_and_results[i].append(0)
        TPrate = []
        FPrate = []
        for ti,t in enumerate(x2thresholds):
            tc = 0
            tpc = 0
            fc = 0
            fpc = 0
            for r in gvals_and_results:
                tc += r[1] == 1
                tpc += r[1]==1 and r[2+ti]==1
                fc += r[1] == 0
                fpc += r[1] == 0 and r[2+ti] == 1
            TPrate.append(tpc/tc)
            FPrate.append(fpc/fc)
        AUC = sum([(FPrate[i+1]-FPrate[i])*(TPrate[i+1]+TPrate[i])/2 for i in range(len(TPrate)-1) ])
        plt.plot(FPrate,TPrate)
        plt.title("Poisson Random Field Likelihood Ratio Test ROC ({} SFS)".format("folded" if dofolded else "unfolded"))
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.text(0.1,0.1,"AUC = {:.3f}".format(AUC),fontsize=14)

        plt.savefig(rocfilename)
        # plt.show()
        plt.clf()
        valstoscreenandfile(rocfilename,x2thresholds,pvalues,TPrate,FPrate,args)

    if doratioPRFROC:
        SFS_functions.SSFconstant_dokuethe = False
        numg0 = ntrials//2
        rocfilename = 'ratioPRF_ROC_theta{}_gmax{}_n{}_{}.pdf'.format(theta,max2Ns,n,foldstring)
        gvals_and_results = [[0.0,0] for i in range(numg0)] # half the g=2Ns values are 0
        for i in range(ntrials-numg0):
            gvals_and_results.append([-np.random.random()*max2Ns,1]) # half the g=2Ns values are negative 
        for i in range(ntrials):
            g = gvals_and_results[i][0]
            nsfs,ssfs,ratios = SFS_functions.simsfsratio(theta,theta,1.0,n,None,dofolded,"single2Ns",[g],None,False,None)
            thetastart = 100.0
            gstart = -1.0
            ratiothetagresult =  minimize(SFS_functions.NegL_SFSRATIO_estimate_thetaS_thetaN,
                np.array([thetastart,gstart]),args=(n,dofolded,"single2Ns",True,None,False,False,ratios),method="Powell",bounds=[(thetastart/10,thetastart*10),(10*gstart,-10*gstart)])
            ratiothetag0result = minimize_scalar(SFS_functions.NegL_SFSRATIO_estimate_thetaS_thetaN,
                bracket=(thetastart/10,thetastart*10),args = (n,dofolded,"fix2Ns0",True,None,False,False,ratios),method='Brent')  
            thetagdelta = 2*(-ratiothetagresult.fun + ratiothetag0result.fun)
            for t in x2thresholds:
                if thetagdelta > t:
                    gvals_and_results[i].append(1)
                else:
                    gvals_and_results[i].append(0)
        TPrate = []
        FPrate = []
        for ti,t in enumerate(x2thresholds):
            tc = 0
            tpc = 0
            fc = 0
            fpc = 0
            for r in gvals_and_results:
                tc += r[1] == 1
                tpc += r[1]==1 and r[2+ti]==1
                fc += r[1] == 0
                fpc += r[1] == 0 and r[2+ti] == 1
            TPrate.append(tpc/tc)
            FPrate.append(fpc/fc)
        AUC = sum([(FPrate[i+1]-FPrate[i])*(TPrate[i+1]+TPrate[i])/2 for i in range(len(TPrate)-1) ])
        plt.plot(FPrate,TPrate)
        plt.title("Site Frequency Ratio Likelihood Ratio Test ROC ({} SFS)".format("folded" if dofolded else "unfolded"))
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.text(0.1,0.1,"AUC = {:.3f}".format(AUC),fontsize=14)
        plt.savefig(rocfilename)
        # plt.show()
        valstoscreenandfile(rocfilename,x2thresholds,pvalues,TPrate,FPrate,args)

def parsecommandline():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f",dest="dofolded",default = False,help = "do folded")    
    parser.add_argument("-m",dest="max2Ns",default = -100, type=float,help="optional setting for the largest 2Ns, default is -100 ")
    parser.add_argument("-n",dest="n",type = int, default = 50,help="# of sampled chromosomes  i.e. 2*(# diploid individuals)  ")
    parser.add_argument("-o",dest="outfilename",type = str, default = None,help="output text file name")
    parser.add_argument("-q",dest="theta",type=float,default = 100,help = "theta ")    
    parser.add_argument("-s",dest="seed",type = int,default = 1,help = "random number seed")  
    args  =  parser.parse_args(sys.argv[1:])   
    args.commandstring = " ".join(sys.argv[1:])
    return args

    # return parser

if __name__ == '__main__':
    """

    """
    args = parsecommandline()
    run(args)